# python test_ensemble_cohort.py --cohort adult --model_path ./models/ensemble_adult/best_model.pth
import argparse
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Local Imports ---
from agents.ensemble_agent import EnsembleAgent
from utils.state_management_closed_loop_ensemble import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from utils.realistic_scenario import RealisticMealScenario
from simglucose.patient.t1dpatient import T1DPatient

def get_cohort_patients(cohort_name):
    if cohort_name == 'adult':
        return [f'adult#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'adolescent':
        return [f'adolescent#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'child':
        return [f'child#{i:03d}' for i in range(1, 11)]
    else:
        raise ValueError(f"Unknown cohort: {cohort_name}")

def evaluate_ensemble_cohort(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"EVALUATING FIXED-WEIGHT ENSEMBLE ON {args.cohort.upper()} COHORT")
    print(f"Model Path: {args.model_path}")
    print(f"{'='*70}\n")

    # 1. SETUP
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # MATCH THE ETA USED DURING FIXED-WEIGHT TRAINING
    eta_val = 2.5 if args.cohort == 'child' else 4.0

    hyperparameters = {
        'max_timesteps_per_episode': 288, # 1 Day
        'ETA': eta_val
    }

    results_dir = f'./results/eval_ensemble_{args.cohort}'
    os.makedirs(results_dir, exist_ok=True)

    state_dim = 4
    action_dim = 1

    # 2. LOAD AGENT
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model at {args.model_path}")
        
    # Initialize the fixed-weight agent
    agent = EnsembleAgent(state_dim, action_dim, max_action=1.0, device=device)
    
    # The load function will automatically pull in the saved ensemble_weights dictionary
    agent.load(args.model_path)
    
    # Set to evaluation mode
    agent.sac_agent.actor.eval()
    agent.td3_agent.actor.eval()
    
    print(f"Loaded Ensemble Weights: {agent.ensemble_weights}")

    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()

    cohort_patients = get_cohort_patients(args.cohort)
    all_metrics = []

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # 3. EVALUATION LOOP (Patient by Patient)
    for patient_name in cohort_patients:
        print(f"Evaluating Patient: {patient_name}...")
        
        patient_obj = T1DPatient.withName(patient_name)
        bw = patient_obj._params['BW']
        
        # Fixed seed for evaluation to ensure apples-to-apples comparison
        eval_scenario = RealisticMealScenario(start_time=start_time, patient=patient_obj, seed=seed)
        
        env_id = f'simglucose/eval-ens-{patient_name.replace("#", "-")}-v0'
        try:
            register(
                id=env_id,
                entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                max_episode_steps=hyperparameters['max_timesteps_per_episode'],
                kwargs={"patient_name": patient_name, "custom_scenario": eval_scenario}
            )
        except gymnasium.error.Error:
            pass
            
        env = gymnasium.make(env_id)
        i_max = float(env.action_space.high[0])

        # --- WARMUP PHASE ---
        obs_array, _ = env.reset(seed=seed)
        manager.reset()
        for _ in range(hyperparameters['max_timesteps_per_episode']):
            u_state = manager.get_full_state(obs_array[0], bw)
            n_state = manager.get_normalized_state(u_state)
            
            # Unpack 2 values for fixed-weight agent
            action, _ = agent.select_action(n_state, evaluate=True)
            insulin_dose = i_max * np.exp(hyperparameters['ETA'] * (action - 1.0))
            safe_action = safety_layer.apply(insulin_dose, u_state)
            
            manager.insulin_history.append(safe_action[0])
            obs_array, _, done, truncated, _ = env.step(safe_action)
            if done or truncated: break

        # --- OFFICIAL EVALUATION PHASE ---
        obs_array, _ = env.reset(seed=seed + 1)
        manager.reset()
        
        glucose_history = [obs_array[0]]
        insulin_history = []
        
        unnormalized_state = manager.get_full_state(obs_array[0], bw)
        current_state = manager.get_normalized_state(unnormalized_state)

        for t in range(hyperparameters['max_timesteps_per_episode']):
            # Unpack 2 values for fixed-weight agent
            action, _ = agent.select_action(current_state, evaluate=True)
            
            insulin_dose = i_max * np.exp(hyperparameters['ETA'] * (action - 1.0))
            safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
            insulin_history.append(safe_action[0])
            manager.insulin_history.append(safe_action[0])
            
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            
            glucose_history.append(next_obs_array[0])
            
            unnormalized_state = manager.get_full_state(next_obs_array[0], bw)
            current_state = manager.get_normalized_state(unnormalized_state)
            
            if terminated or truncated:
                break
        
        env.close()

        if len(insulin_history) < len(glucose_history):
            insulin_history.append(0.0)

        glucose_arr = np.array(glucose_history)
        
        # Calculate Clinical Metrics
        tir = np.sum((glucose_arr >= 70) & (glucose_arr <= 180)) / len(glucose_arr) * 100
        hypo = np.sum(glucose_arr < 70) / len(glucose_arr) * 100
        severe_hypo = np.sum(glucose_arr < 54) / len(glucose_arr) * 100
        hyper = np.sum(glucose_arr > 180) / len(glucose_arr) * 100
        mean_bg = np.mean(glucose_arr)
        
        all_metrics.append({
            'Patient': patient_name,
            'TIR (%)': round(tir, 2),
            'Hypo (%)': round(hypo, 2),
            'Severe Hypo (%)': round(severe_hypo, 2),
            'Hyper (%)': round(hyper, 2),
            'Mean BG': round(mean_bg, 2)
        })

        # --- PLOTTING ---
        time_axis = np.arange(len(glucose_history)) * 5 / 60 # Convert to hours
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Time (Hours)', fontsize=12)
        ax1.set_ylabel('Blood Glucose (mg/dL)', color='blue', fontsize=12)
        ax1.plot(time_axis, glucose_history, color='blue', label='Glucose', linewidth=2)
        ax1.axhline(y=180, color='red', linestyle='--', alpha=0.5, label='Hyper Limit (180)')
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Hypo Limit (70)')
        ax1.fill_between(time_axis, 70, 180, color='green', alpha=0.1, label='Target Range')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(40, max(300, max(glucose_history) + 20))

        ax2 = ax1.twinx()
        ax2.set_ylabel('Insulin Dose (U / 5min)', color='orange', fontsize=12)
        ax2.bar(time_axis, insulin_history, width=0.08, color='orange', alpha=0.6, label='Insulin')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim(0, max(2.0, max(insulin_history) * 2))

        # Show the static weights used for this evaluation in the plot title
        sac_w = agent.ensemble_weights['sac']
        td3_w = agent.ensemble_weights['td3']
        fig.suptitle(f'Ensemble Eval: {patient_name} (TIR: {tir:.1f}% | W: SAC {sac_w:.1f}, TD3 {td3_w:.1f})', fontsize=14, weight='bold')
        fig.tight_layout()
        
        plot_path = os.path.join(results_dir, f'plot_{patient_name.replace("#", "-")}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # 4. SAVE SUMMARY CSV
    df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, 'ensemble_evaluation_summary.csv')
    df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\nPlots and summary saved to: {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Fixed-Weight Ensemble RL System")
    parser.add_argument('--cohort', type=str, choices=['child', 'adolescent', 'adult'], required=True)
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_model.pth")
    parser.add_argument('--seed', type=int, default=100, help="Random seed for deterministic evaluation")
    
    args = parser.parse_args()
    evaluate_ensemble_cohort(args)