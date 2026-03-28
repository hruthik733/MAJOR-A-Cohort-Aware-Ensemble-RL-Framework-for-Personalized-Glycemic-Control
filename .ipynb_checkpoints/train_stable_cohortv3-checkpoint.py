import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from tqdm import tqdm

# --- Local Imports ---
# Make sure you are using the correct, updated files
from agents.sac_agent_mdn import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management_v3 import StateRewardManager
from utils.safety_v3 import SafetyLayer
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def main():
    # --- 1. Reproducibility ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- 2. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Hyperparameters ---
    max_episodes = 2000
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1_000_000
    max_timesteps_per_episode = 288
    # Use a long, safe exploration phase to gather good data before learning
    learning_starts = 50000 
    
    # --- 4. Define the patient cohort ---
    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]

    # --- 5. Setup Directories and Paths ---
    AGENT_NAME = 'sac_mdn_cohort_stable'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    actor_path = f'{model_dir}/actor_adult_cohort.pth'

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # --- 6. Environment Registration ---
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
    env_id = 'simglucose/adult-cohort-v0'
    try:
        register(
            id=env_id,
            entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
            max_episode_steps=max_timesteps_per_episode,
            kwargs={"patient_name": adult_patients, "custom_scenario": meal_scenario}
        )
    except gymnasium.error.Error:
        pass

    # --- 7. Initialization ---
    state_dim = 12  # Using the new 12-dimensional state
    action_dim = 1

    temp_env = gymnasium.make(env_id)
    agent = SACAgent(temp_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha, device)
    temp_env.close()
    
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)
    total_timesteps_taken = 0

    # --- 8. Main Training Loop ---
    print("\n" + "="*50)
    print("--- Starting Stable Training for Adult Cohort Model ---")
    print("="*50)
    
    for i_episode in range(1, max_episodes + 1):
        # Re-create the environment to ensure a new random patient is sampled
        env = gymnasium.make(env_id)
        env.action_space.seed(SEED + i_episode)
        
        obs_array, info = env.reset(seed=SEED + i_episode)
        
        patient_name_for_episode = info.get('patient_name', 'Unknown')
        manager.reset()

        current_sim_time = env.unwrapped.env.env.time
        upcoming_carbs = env.unwrapped.env.custom_scenario.get_action(current_sim_time).meal
        unnormalized_state = manager.get_full_state(obs_array[0], current_sim_time, upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        
        episode_reward = 0
        glucose_history = [unnormalized_state[0]]

        for t in range(max_timesteps_per_episode):
            if total_timesteps_taken < learning_starts:
                # Use small, safe random actions during the long exploration phase
                action = np.array([np.random.uniform(low=0, high=1.0)])
            else:
                action = agent.select_action(current_state)

            safe_action = safety_layer.apply(action, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            current_sim_time = env.unwrapped.env.env.time
            upcoming_carbs = env.unwrapped.env.custom_scenario.get_action(current_sim_time).meal
            next_unnormalized_state = manager.get_full_state(next_obs_array[0], current_sim_time, upcoming_carbs)
            next_state = manager.get_normalized_state(next_unnormalized_state)

            reward = manager.get_reward(unnormalized_state)
            replay_buffer.push(current_state, safe_action, reward, next_state, done)

            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1
            glucose_history.append(unnormalized_state[0])

            if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

            if done:
                break
        
        env.close()

        # Log performance periodically
        if i_episode % 25 == 0:
            tir = np.sum((np.array(glucose_history) >= 70) & (np.array(glucose_history) <= 180)) / len(glucose_history) * 100
            print(f"Episode {i_episode}/{max_episodes} | Patient: {patient_name_for_episode} | Length: {t+1} | Reward: {episode_reward:.2f} | TIR: {tir:.2f}%")

    print("--- Training Finished ---")
    torch.save(agent.actor.state_dict(), actor_path)
    print(f"Saved trained cohort model to {actor_path}")

    # --- 9. Systematic Evaluation ---
    print("\n" + "="*50)
    print("--- Starting Systematic Evaluation ---")
    print("="*50)
    
    all_patient_results = []
    # Create a dummy env for the evaluation agent constructor
    eval_env_dummy = gymnasium.make(env_id)
    eval_agent = SACAgent(eval_env_dummy, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha, device)
    eval_env_dummy.close()
    
    eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    eval_agent.actor.eval()

    for patient_name in adult_patients:
        print(f"\n--- Evaluating on Patient: {patient_name} ---")
        meal_times = [7 * 60, 12 * 60, 18 * 60]
        meal_carbs = [45, 70, 80]
        eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
        
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario, patient_name=patient_name)
        
        manager.reset()
        obs_array, info = eval_env.reset()
        
        current_sim_time = eval_env.unwrapped.env.env.time
        upcoming_carbs = eval_scenario.get_action(current_sim_time).meal
        unnormalized_state = manager.get_full_state(obs_array[0], current_sim_time, upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history = [unnormalized_state[0]]

        for t in range(max_timesteps_per_episode):
            with torch.no_grad():
                action = eval_agent.select_action(current_state)

            safe_action = safety_layer.apply(action, unnormalized_state)
            manager.insulin_history.append(safe_action[0])
            obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
            
            current_sim_time = eval_env.unwrapped.env.env.time
            upcoming_carbs = eval_scenario.get_action(current_sim_time).meal
            unnormalized_state = manager.get_full_state(obs_array[0], current_sim_time, upcoming_carbs)
            current_state = manager.get_normalized_state(unnormalized_state)
            glucose_history.append(unnormalized_state[0])

            if terminated or truncated:
                break
        
        eval_env.close()

        # Metrics and Plotting
        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        # ... (calculate other metrics as needed) ...
        
        all_patient_results.append({"Patient": patient_name, "Time in Range (%)": time_in_range})
        
        plt.figure(figsize=(15, 6))
        # ... (plotting code from previous version) ...
        plot_path = f'{results_dir}/evaluation_plot_{patient_name.replace("#", "-")}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to {plot_path}")

    # --- 10. Final Summary ---
    print("\n" + "="*56)
    print("---           OVERALL PERFORMANCE SUMMARY           ---")
    print("="*56)
    results_df = pd.DataFrame(all_patient_results).set_index('Patient')
    print(results_df)
    print("\n--- Average Performance Across All Patients ---")
    print(results_df.mean().to_string())

if __name__ == '__main__':
    main()