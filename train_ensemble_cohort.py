import argparse
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import os
import random
import collections

# --- Local Imports ---
from agents.ensemble_agent import EnsembleAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management_closed_loop_ensemble import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from utils.realistic_scenario import RealisticMealScenario
from simglucose.patient.t1dpatient import T1DPatient

def get_cohort_patients(cohort_name):
    if cohort_name == 'adult': return [f'adult#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'adolescent': return [f'adolescent#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'child': return [f'child#{i:03d}' for i in range(1, 11)]
    else: raise ValueError(f"Unknown cohort: {cohort_name}")

def train_ensemble(args):
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting TRAINABLE ENSEMBLE on {args.cohort.upper()} cohort using {device}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Note: ETA is completely removed as we now use physiological clinical mapping
    hyperparameters = {
        'max_episodes': 600,
        'max_timesteps_per_episode': 288,
        'batch_size': 256,
        'replay_buffer_size': 1000000,
        'learning_starts': 2500
    }

    model_dir = f'./models/trainable_ensemble_{args.cohort}'
    os.makedirs(model_dir, exist_ok=True)

    # 2. ENVIRONMENT INITIALIZATION
    cohort_patients = get_cohort_patients(args.cohort)
    envs, patient_bws = {}, {}
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    for patient_name in cohort_patients:
        patient_obj = T1DPatient.withName(patient_name)
        meal_scenario = RealisticMealScenario(start_time=start_time, patient=patient_obj, seed=seed)
        env_id = f'simglucose/ens-{patient_name.replace("#", "-")}-v0'
        
        try:
            register(id=env_id, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                     max_episode_steps=hyperparameters['max_timesteps_per_episode'],
                     kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario})
        except gymnasium.error.Error: pass
            
        env = gymnasium.make(env_id)
        envs[patient_name] = env
        patient_bws[patient_name] = patient_obj._params['BW']

    state_dim, action_dim = 4, 1

    # 3. INITIALIZE TRAINABLE ENSEMBLE AGENT
    agent = EnsembleAgent(state_dim, action_dim, max_action=1.0, device=device)
    manager = StateRewardManager(state_dim) 
    # safety_layer = SafetyLayer()
    # Pass the cohort argument to the SafetyLayer
    safety_layer = SafetyLayer(cohort=args.cohort)
    replay_buffer = ReplayBuffer(hyperparameters['replay_buffer_size'])

    # 4. TRAINING LOOP
    total_timesteps = 0
    
    # Checkpointing Logic: Track moving average across the cohort size
    recent_rewards = collections.deque(maxlen=len(cohort_patients))
    best_avg_reward = -float('inf')

    print("\n" + "="*70)
    print(f"--- TRAINING META-CONTROLLER ENSEMBLE FOR {args.cohort.upper()} ---")
    print("="*70)

    for i_episode in range(1, hyperparameters['max_episodes'] + 1):
        current_patient = random.choice(cohort_patients)
        env = envs[current_patient]
        bw = patient_bws[current_patient]

        obs_array, _ = env.reset(seed=seed + i_episode)
        manager.reset()

        u_state = manager.get_full_state(obs_array[0], bw)
        c_state = manager.get_normalized_state(u_state)
        
        episode_reward, episode_steps = 0, 0
        sac_weight_tracker = []

        for t in range(hyperparameters['max_timesteps_per_episode']):
            if total_timesteps < hyperparameters['learning_starts']:
                raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
            else:
                raw_action, w_sac, w_td3 = agent.select_action(c_state, evaluate=False)
                sac_weight_tracker.append(w_sac)

            # --- FINAL TUNED EXTENDED BOLUS LIMITS ---
            if args.cohort == 'child':
                clinical_max = 0.25      
            elif args.cohort == 'adolescent':
                clinical_max = 0.75      # Slightly raised from 0.5 
            else:
                clinical_max = 2.0       # Raised from 1.0 to fix Adult Hyperglycemia
                
            normalized_action = (raw_action[0] + 1.0) / 2.0  # Use action[0] in test script
            
            # For adults, we remove the squaring math so they get linear, powerful boluses
            if args.cohort == 'adult':
                insulin_dose = np.array([normalized_action * clinical_max])
            else:
                # Children and Adolescents keep the squared math for micro-dosing precision
                insulin_dose = np.array([(normalized_action ** 2) * clinical_max])
            
            # Apply safety rules
            safe_action = safety_layer.apply(insulin_dose, u_state)
            manager.insulin_history.append(safe_action[0])
            
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            next_u_state = manager.get_full_state(next_obs_array[0], bw)
            next_c_state = manager.get_normalized_state(next_u_state)
            reward = manager.get_reward(u_state)
            
            replay_buffer.push(c_state, raw_action, reward, next_c_state, done)
            
            c_state = next_c_state
            u_state = next_u_state
            episode_reward += reward
            total_timesteps += 1
            episode_steps += 1

            if total_timesteps > hyperparameters['learning_starts']:
                agent.update(replay_buffer, hyperparameters['batch_size'])

            if done: break
        
        # Add this episode's reward to our rolling memory
        recent_rewards.append(episode_reward)
        
        # -------------------------------------------------------------
        # LOGGING AND ROLLING AVERAGE CHECKPOINTING
        # -------------------------------------------------------------
        is_new_best = False
        improvement = 0.0
        current_avg = np.mean(recent_rewards)

        # Only check for saves AFTER warmup AND after we have a full queue of 10 patients
        if total_timesteps > hyperparameters['learning_starts'] and len(recent_rewards) == len(cohort_patients):
            if current_avg > best_avg_reward:
                is_new_best = True
                improvement = current_avg - best_avg_reward if best_avg_reward != -float('inf') else 0
                best_avg_reward = current_avg
                agent.save(os.path.join(model_dir, "best_model.pth"))

        if i_episode % 10 == 0 or i_episode == 1 or is_new_best:
            avg_sac_w = np.mean(sac_weight_tracker) if sac_weight_tracker else 0.5
            log_msg = (f"[{args.cohort.upper():^10}] Ep {i_episode:03d} | "
                       f"Patient: {current_patient:12s} | "
                       f"Reward: {episode_reward:9.2f} | "
                       f"Rolling Avg: {current_avg:9.2f} | "
                       f"Ep Steps: {episode_steps:<3} | "
                       f"Avg SAC Wt: {avg_sac_w:.2f} | "
                       f"Total Steps: {total_timesteps:<6}")
            
            if is_new_best:
                log_msg += f" | 🌟 NEW BEST AVG! (+{improvement:.2f})"
            print(log_msg)

    agent.save(os.path.join(model_dir, "model_final.pth"))
    print("\n" + "="*70)
    print(f"Training Complete! Final model saved to {model_dir}/model_final.pth")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Meta-Controller Ensemble Agent")
    parser.add_argument('--cohort', type=str, choices=['child', 'adolescent', 'adult'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_ensemble(args)