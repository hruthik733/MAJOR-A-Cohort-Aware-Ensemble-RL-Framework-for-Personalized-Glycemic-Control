import argparse
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import os
import random
import pandas as pd

# --- Local Imports ---
from agents.sac_baseline import SACBaselineAgent
from agents.td3_baseline import TD3BaselineAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management_closed_loop_ensemble import StateRewardManager
from utils.safety2_closed_loop import SafetyLayer
from utils.realistic_scenario import RealisticMealScenario
from simglucose.patient.t1dpatient import T1DPatient

def get_cohort_patients(cohort_name):
    """Returns the list of patient IDs for the selected cohort."""
    if cohort_name == 'adult':
        return [f'adult#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'adolescent':
        return [f'adolescent#{i:03d}' for i in range(1, 11)]
    elif cohort_name == 'child':
        return [f'child#{i:03d}' for i in range(1, 11)]
    else:
        raise ValueError(f"Unknown cohort: {cohort_name}")

def train_cohort(args):
    # =================================================================
    # 1. SETUP AND CONFIGURATION
    # =================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting {args.agent.upper()} training on {args.cohort.upper()} cohort using {device}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hyperparameters = {
        'max_episodes': 500,  # Increased for cohort training
        'max_timesteps_per_episode': 288,  # 24 hours at 5-min intervals
        'batch_size': 256,
        'replay_buffer_size': 1000000,
        'learning_starts': 2000,
        'ETA': 4.0
    }

    AGENT_NAME = f'{args.agent}_baseline_{args.cohort}'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # =================================================================
    # 2. ENVIRONMENT INITIALIZATION FOR COHORT
    # =================================================================
    cohort_patients = get_cohort_patients(args.cohort)
    envs = {}
    patient_bws = {}
    patient_i_max = {}

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    print(f"Initializing environments for {len(cohort_patients)} {args.cohort} patients...")
    for patient_name in cohort_patients:
        patient_obj = T1DPatient.withName(patient_name)
        
        # Fully closed-loop Artificial Pancreas (AP) system: realistic meals but NO meal announcements to the agent
        meal_scenario = RealisticMealScenario(start_time=start_time, patient=patient_obj, seed=seed)
        clean_patient_name = patient_name.replace('#', '-')
        env_id = f'simglucose/{args.agent}-{clean_patient_name}-v0'
        
        try:
            register(
                id=env_id,
                entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                max_episode_steps=hyperparameters['max_timesteps_per_episode'],
                kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
            )
        except gymnasium.error.Error:
            pass # Environment already registered
            
        env = gymnasium.make(env_id)
        envs[patient_name] = env
        
        # Extract physiological data for state context mapping
        patient_bws[patient_name] = patient_obj._params['BW']
        patient_i_max[patient_name] = float(env.action_space.high[0])

    # State Dimension is 4: [Glucose, Rate_of_Change, IOB, Body_Weight]
    state_dim = 4 
    action_dim = 1

    # =================================================================
    # 3. AGENT & UTILS INITIALIZATION
    # =================================================================
    if args.agent == 'sac':
        agent = SACBaselineAgent(state_dim, action_dim, max_action=1.0, device=device)
    elif args.agent == 'td3':
        agent = TD3BaselineAgent(state_dim, action_dim, max_action=1.0, device=device)
    else:
        raise ValueError("Invalid agent type selected.")

    # Initialize manager with the new 4D state expectation for normalization
    manager = StateRewardManager(state_dim) 
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(hyperparameters['replay_buffer_size'])

    # =================================================================
    # 4. TRAINING LOOP
    # =================================================================
    total_timesteps_taken = 0
    training_rewards_history = []
    best_reward = -float('inf')

    print("\n" + "="*70)
    print(f"--- STARTING TRAINING FOR {args.cohort.upper()} COHORT ---")
    print("="*70)

    for i_episode in range(1, hyperparameters['max_episodes'] + 1):
        # Dynamically sample a patient to force cohort-level generalization
        current_patient = random.choice(cohort_patients)
        env = envs[current_patient]
        bw = patient_bws[current_patient]
        i_max = patient_i_max[current_patient]

        obs_array, info = env.reset(seed=seed + i_episode)
        manager.reset()

        # -------------------------------------------------------------
        # STATE CONSTRUCTION (Clean 4D Implementation)
        # -------------------------------------------------------------
        # Get fully formed 4D unnormalized state directly from manager
        unnormalized_state = manager.get_full_state(obs_array[0], bw)
        # Normalize the 4D state for the neural networks
        current_state = manager.get_normalized_state(unnormalized_state)
        
        episode_reward = 0
        episode_timesteps = 0

        for t in range(hyperparameters['max_timesteps_per_episode']):
            # Exploration vs Exploitation
            if total_timesteps_taken < hyperparameters['learning_starts']:
                raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
            else:
                raw_action = agent.select_action(current_state, evaluate=False)

            # Map neural network output to insulin scale
            insulin_dose = i_max * np.exp(hyperparameters['ETA'] * (raw_action - 1.0))
            
            # Pass the 4D unnormalized state directly to the updated safety layer
            safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            # ---------------------------------------------------------
            # NEXT STATE CONSTRUCTION & REWARD
            # ---------------------------------------------------------
            next_unnormalized_state = manager.get_full_state(next_obs_array[0], bw)
            next_state = manager.get_normalized_state(next_unnormalized_state)
            
            # Pass the 4D state directly to the updated reward function
            reward = manager.get_reward(unnormalized_state)
            
            # Push transition to shared replay buffer
            replay_buffer.push(current_state, raw_action, reward, next_state, done)
            
            # Advance state
            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1
            episode_timesteps += 1

            # Train the network
            if total_timesteps_taken > hyperparameters['learning_starts']:
                agent.update(replay_buffer, hyperparameters['batch_size'])

            if done:
                break
        
        training_rewards_history.append(episode_reward)
        
        # -------------------------------------------------------------
        # LOGGING AND CHECKPOINTING
        # -------------------------------------------------------------
        is_new_best = False
        improvement = 0.0

        # Check for model improvement (only after warmup period)
        if total_timesteps_taken > hyperparameters['learning_starts'] and episode_reward > best_reward:
            is_new_best = True
            improvement = episode_reward - best_reward if best_reward != -float('inf') else 0
            best_reward = episode_reward
            
            best_model_path = os.path.join(model_dir, "best_model.pth")
            agent.save(best_model_path)

        # Print log if it's episode 1, a multiple of 10, OR if we just hit a new best score
        if i_episode % 10 == 0 or i_episode == 1 or is_new_best:
            log_msg = (f"[{args.cohort.upper():^10}] Ep {i_episode:03d} | "
                       f"Patient: {current_patient:12s} | "
                       f"Reward: {episode_reward:9.2f} | "
                       f"Ep Steps: {episode_timesteps:<3} | "
                       f"Total Steps: {total_timesteps_taken:<6}")
            
            # If it's a new best, append the star and improvement amount to the same line
            if is_new_best:
                log_msg += f" | 🌟 NEW BEST! (+{improvement:.2f})"
                
            print(log_msg)

    # Save final model state
    final_model_path = os.path.join(model_dir, "model_final.pth")
    agent.save(final_model_path)
    print("\n" + "="*70)
    print(f"Training Complete! Final model saved to {final_model_path}")
    print(f"Highest historical reward achieved: {best_reward:.2f}")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train fully closed-loop AP System Cohort RL Agent")
    parser.add_argument('--agent', type=str, choices=['sac', 'td3'], required=True, help="Baseline agent to train (sac or td3)")
    parser.add_argument('--cohort', type=str, choices=['child', 'adolescent', 'adult'], required=True, help="Target patient cohort")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    train_cohort(args)