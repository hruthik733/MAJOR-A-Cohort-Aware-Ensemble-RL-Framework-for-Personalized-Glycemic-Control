# train_all_adults_safety3.py (but overall summary printing is missing-upto friday- reason a bug in plotting graph upwards line)
# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import random
# import pandas as pd
# import multiprocessing as mp

# # --- Local Imports ---
# from agents.sac_agent_mdn import SACAgent
# from utils.replay_buffer import ReplayBuffer
# from utils.state_management2 import StateRewardManager
# from utils.safety2 import SafetyLayer
# import simglucose.simulation.scenario_gen as scgen
# from simglucose.simulation.scenario import CustomScenario

# def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
#     """
#     This function contains the entire training and evaluation loop for a single patient.
#     It's designed to be run as a separate process on a specific GPU.
#     """
#     # 1. SETUP AND CONFIGURATION
#     # =================================================================
#     device = torch.device(f"cuda:{gpu_id}")
#     print(f"Process for {patient_name} starting on device: {device}")

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.set_device(device)
#     torch.cuda.manual_seed_all(seed)

#     # Unpack hyperparameters
#     max_episodes = hyperparameters['max_episodes']
#     lr = hyperparameters['lr']
#     gamma_val = hyperparameters['gamma_val']
#     tau = hyperparameters['tau']
#     alpha = hyperparameters['alpha']
#     batch_size = hyperparameters['batch_size']
#     n_latent_var = hyperparameters['n_latent_var']
#     replay_buffer_size = hyperparameters['replay_buffer_size']
#     max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
#     learning_starts = hyperparameters['learning_starts']
#     ETA = hyperparameters['ETA']

#     AGENT_NAME = 'sac_personalized_exp'
#     model_dir = f'./models/{AGENT_NAME}'
#     results_dir = f'./results/{AGENT_NAME}'
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(results_dir, exist_ok=True)
#     actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
#     now = datetime.now()
#     start_time = datetime.combine(now.date(), datetime.min.time())

#     meal_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)
#     clean_patient_name = patient_name.replace('#', '-')
#     env_id = f'simglucose/{clean_patient_name}-v0'
    
#     try:
#         register(
#             id=env_id,
#             entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#             max_episode_steps=max_timesteps_per_episode,
#             kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
#         )
#     except gymnasium.error.Error:
#         pass

#     env = gymnasium.make(env_id)
#     env.action_space.seed(seed)
    
#     state_dim = 4
#     action_dim = 1
#     i_max = float(env.action_space.high[0])

#     # Correctly initialize the agent with numeric parameters, not the env object
#     agent = SACAgent(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         max_action=1.0,  # The agent's internal max_action is 1.0 due to Tanh
#         n_latent_var=n_latent_var,
#         lr=lr,
#         gamma=gamma_val,
#         tau=tau,
#         alpha=alpha,
#         device=device
#     )

#     manager = StateRewardManager(state_dim)
#     safety_layer = SafetyLayer()
#     replay_buffer = ReplayBuffer(replay_buffer_size)

#     # 2. TRAINING LOOP
#     # =================================================================
#     print(f"--- Starting Training for Patient: {patient_name} on {device} ---")
#     total_timesteps_taken = 0
#     for i_episode in range(1, max_episodes + 1):
#         obs_array, info = env.reset(seed=seed + i_episode)
#         episode_scenario = info.get('scenario')
#         manager.reset()

#         current_sim_time = env.unwrapped.env.env.time
#         # ! FIX: Use .meal instead of .CHO
#         upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#         unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#         current_state = manager.get_normalized_state(unnormalized_state)
#         episode_reward = 0

#         for t in range(max_timesteps_per_episode):
#             if total_timesteps_taken < learning_starts:
#                 raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
#             else:
#                 raw_action = agent.select_action(current_state)

#             insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
#             safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
#             manager.insulin_history.append(safe_action[0])
#             next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
#             done = terminated or truncated

#             current_sim_time = env.unwrapped.env.env.time
#             upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
#             next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
#             next_state = manager.get_normalized_state(next_unnormalized_state)
#             reward = manager.get_reward(unnormalized_state)
            
#             replay_buffer.push(current_state, raw_action, reward, next_state, done)
            
#             current_state = next_state
#             unnormalized_state = next_unnormalized_state
#             episode_reward += reward
#             total_timesteps_taken += 1

#             if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
#                 agent.update(replay_buffer, batch_size)

#             if done:
#                 break
        
#         if i_episode % 50 == 0:
#             print(f"[{patient_name} on {device}] Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")
    
#     env.close()
#     print(f"--- Training Finished for {patient_name} ---")
#     torch.save(agent.actor.state_dict(), actor_path)
#     print(f"Saved trained model to {actor_path}")


#     # 3. EVALUATION LOOP
#     # =================================================================
#     print(f"\n--- Starting Evaluation for {patient_name} ---")
#     meal_times = [7 * 60, 12 * 60, 18 * 60]
#     meal_carbs = [45, 70, 80]
#     eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
#     eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    
#     eval_agent = SACAgent(
#         state_dim=state_dim, action_dim=action_dim, max_action=1.0,
#         n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau,
#         alpha=alpha, device=device
#     )
#     eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
#     eval_agent.actor.eval()

#     manager.reset()
#     obs_array, info = eval_env.reset()

#     current_sim_time = eval_env.unwrapped.env.env.time
#     upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
#     unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#     current_state = manager.get_normalized_state(unnormalized_state)
#     glucose_history = [obs_array[0]]

#     for t in range(max_timesteps_per_episode):
#         with torch.no_grad():
#             raw_action = eval_agent.select_action(current_state)

#         insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
#         safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
        
#         manager.insulin_history.append(safe_action[0])
#         obs_array, _, terminated, truncated, _ = env.step(safe_action)

#         current_sim_time = eval_env.unwrapped.env.env.time
#         upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
#         unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#         current_state = manager.get_normalized_state(unnormalized_state)
#         glucose_history.append(obs_array[0])

#         if terminated or truncated:
#             break
            
#     eval_env.close()
    
#     # Generate and save plot
#     plt.figure(figsize=(15, 6))
#     time_axis_minutes = np.arange(len(glucose_history)) * 5
#     plt.plot(time_axis_minutes, glucose_history, label='SAC Agent Glucose')
#     plt.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold')
#     plt.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold')
#     for meal_time, meal_amount in zip(meal_times, meal_carbs):
#         plt.axvline(x=meal_time, color='black', linestyle='--', label=f'Meal ({meal_amount}g)')
#     plt.title(f'Performance for {patient_name}')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Blood Glucose (mg/dL)')
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())
#     plt.grid(True)
#     plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"Saved evaluation plot to {plot_path}")

#     # Return results for aggregation
#     glucose_history = np.array(glucose_history)
#     time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
#     time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
#     time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
#     mean_glucose = np.mean(glucose_history)

#     return {
#         "Patient": patient_name,
#         "Mean Glucose (mg/dL)": mean_glucose,
#         "Time in Range (%)": time_in_range,
#         "Time Hypo (%)": time_hypo,
#         "Time Hyper (%)": time_hyper,
#     }

# if __name__ == '__main__':
#     try:
#         mp.set_start_method('spawn')
#     except RuntimeError:
#         pass

#     hyperparameters = {
#         'max_episodes': 200, 'lr': 2e-4, 'gamma_val': 0.99, 'tau': 0.005, 'alpha': 0.2,
#         'batch_size': 256, 'n_latent_var': 256, 'replay_buffer_size': 1000000,
#         'max_timesteps_per_episode': 288, 'learning_starts': 1500, 'ETA': 4.0
#     }

#     adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
#     tasks = []
#     for i, patient_name in enumerate(adult_patients):
#         gpu_id = i % 2
#         tasks.append((patient_name, gpu_id, 42 + i, hyperparameters))

#     print("\n" + "="*50)
#     print(f"Starting parallel training for {len(tasks)} patients on 2 GPUs...")
#     print("="*50)
    
#     with mp.Pool(processes=len(tasks)) as pool:
#         pool.starmap(train_and_evaluate_patient, tasks)

#     print("\n" + "="*50)
#     print("--- ALL TRAINING PROCESSES COMPLETE ---")
#     print("="*50)















##############
# train_all_adults_safety3.py (with text summary and overall summary plot)
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import multiprocessing as mp

# --- Local Imports ---
from agents.sac_agent_mdn import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management2 import StateRewardManager
from utils.safety2 import SafetyLayer
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def train_and_evaluate_patient(patient_name, gpu_id, seed, hyperparameters):
    """
    This function contains the entire training and evaluation loop for a single patient.
    It's designed to be run as a separate process on a specific GPU.
    """
    # 1. SETUP AND CONFIGURATION
    # =================================================================
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process for {patient_name} starting on device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)

    # Unpack hyperparameters
    max_episodes = hyperparameters['max_episodes']
    lr = hyperparameters['lr']
    gamma_val = hyperparameters['gamma_val']
    tau = hyperparameters['tau']
    alpha = hyperparameters['alpha']
    batch_size = hyperparameters['batch_size']
    n_latent_var = hyperparameters['n_latent_var']
    replay_buffer_size = hyperparameters['replay_buffer_size']
    max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
    learning_starts = hyperparameters['learning_starts']
    ETA = hyperparameters['ETA']

    AGENT_NAME = 'sac_personalized_exp'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'
    
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=seed)
    clean_patient_name = patient_name.replace('#', '-')
    env_id = f'simglucose/{clean_patient_name}-v0'
    
    try:
        register(
            id=env_id,
            entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
            max_episode_steps=max_timesteps_per_episode,
            kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
        )
    except gymnasium.error.Error:
        pass

    env = gymnasium.make(env_id)
    env.action_space.seed(seed)
    
    state_dim = 4
    action_dim = 1
    i_max = float(env.action_space.high[0])

    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, max_action=1.0,
        n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau,
        alpha=alpha, device=device
    )

    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # 2. TRAINING LOOP
    # =================================================================
    print(f"--- Starting Training for Patient: {patient_name} on {device} ---")
    total_timesteps_taken = 0
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset(seed=seed + i_episode)
        episode_scenario = info.get('scenario')
        manager.reset()

        current_sim_time = env.unwrapped.env.env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0

        for t in range(max_timesteps_per_episode):
            if total_timesteps_taken < learning_starts:
                raw_action = np.random.uniform(low=-1.0, high=1.0, size=(action_dim,))
            else:
                raw_action = agent.select_action(current_state)

            insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
            safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            current_sim_time = env.unwrapped.env.env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
            next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
            next_state = manager.get_normalized_state(next_unnormalized_state)
            reward = manager.get_reward(unnormalized_state)
            
            replay_buffer.push(current_state, raw_action, reward, next_state, done)
            
            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1

            if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

            if done:
                break
        
        if i_episode % 50 == 0:
            print(f"[{patient_name} on {device}] Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")
    
    env.close()
    print(f"--- Training Finished for {patient_name} ---")
    torch.save(agent.actor.state_dict(), actor_path)
    print(f"Saved trained model to {actor_path}")


    # 3. EVALUATION LOOP
    # =================================================================
    print(f"\n--- Starting Evaluation for {patient_name} ---")
    meal_times = [7 * 60, 12 * 60, 18 * 60]
    meal_carbs = [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
    try:
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    except gymnasium.error.Error:
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
    
    eval_agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, max_action=1.0,
        n_latent_var=n_latent_var, lr=lr, gamma=gamma_val, tau=tau,
        alpha=alpha, device=device
    )
    eval_agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    eval_agent.actor.eval()

    manager.reset()
    obs_array, info = eval_env.reset()

    current_sim_time = eval_env.unwrapped.env.env.time
    upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
    unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]

    for t in range(max_timesteps_per_episode):
        with torch.no_grad():
            raw_action = eval_agent.select_action(current_state)

        insulin_dose = i_max * np.exp(ETA * (raw_action - 1.0))
        safe_action = safety_layer.apply(insulin_dose, unnormalized_state)
        
        manager.insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)

        current_sim_time = eval_env.unwrapped.env.env.time
        upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])

        if terminated or truncated:
            break
            
    eval_env.close()
    
    # Generate and save individual plot
    plt.figure(figsize=(15, 6))
    time_axis_minutes = np.arange(len(glucose_history)) * 5
    plt.plot(time_axis_minutes, glucose_history, label='SAC Agent Glucose')
    plt.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold')
    plt.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold')
    
    meal_labels_seen = set()
    for meal_time, meal_amount in zip(meal_times, meal_carbs):
        label = f'Meal ({meal_amount}g)'
        if label not in meal_labels_seen:
            plt.axvline(x=meal_time, color='black', linestyle='--', label=label)
            meal_labels_seen.add(label)
        else:
            plt.axvline(x=meal_time, color='black', linestyle='--')

    plt.title(f'Performance for {patient_name}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Blood Glucose (mg/dL)')
    plt.legend()
    plt.grid(True)
    plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved evaluation plot to {plot_path}")

    # Calculate metrics and return results for aggregation
    glucose_history = np.array(glucose_history)
    time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
    time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
    time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
    mean_glucose = np.mean(glucose_history)

    return {
        "Patient": patient_name,
        "Mean Glucose (mg/dL)": mean_glucose,
        "Time in Range (%)": time_in_range,
        "Time Hypo (%)": time_hypo,
        "Time Hyper (%)": time_hyper,
    }

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    hyperparameters = {
        'max_episodes': 200, 'lr': 2e-4, 'gamma_val': 0.99, 'tau': 0.005, 'alpha': 0.2,
        'batch_size': 256, 'n_latent_var': 256, 'replay_buffer_size': 1000000,
        'max_timesteps_per_episode': 288, 'learning_starts': 1500, 'ETA': 4.0
    }
    results_dir = f'./results/sac_personalized_exp'
    os.makedirs(results_dir, exist_ok=True)

    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    tasks = []
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("This script requires at least one GPU.")

    for i, patient_name in enumerate(adult_patients):
        gpu_id = i % num_gpus
        tasks.append((patient_name, gpu_id, 42 + i, hyperparameters))

    print("\n" + "="*50)
    print(f"Starting parallel training for {len(tasks)} patients on {num_gpus} GPUs...")
    print("="*50)
    
    num_processes = min(len(tasks), mp.cpu_count()) 
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(train_and_evaluate_patient, tasks)

    print("\n" + "="*50)
    print("--- ALL TRAINING PROCESSES COMPLETE ---")
    print("="*50)

    # 4. AGGREGATE, DISPLAY, AND PLOT FINAL RESULTS
    # =================================================================
    if results:
        results_df = pd.DataFrame(results)
        column_order = [
            "Patient", "Mean Glucose (mg/dL)", "Time in Range (%)",
            "Time Hypo (%)", "Time Hyper (%)"
        ]
        results_df = results_df[column_order]
        average_metrics = results_df.select_dtypes(include=np.number).mean()

        print("\n" + "="*70)
        print("--- OVERALL EVALUATION SUMMARY ---")
        print("="*70)

        print("\n--- Performance per Patient ---")
        print(results_df.to_string(index=False))

        print("\n\n--- Average Performance Across All Patients ---")
        print(f"Mean Glucose (mg/dL): {average_metrics['Mean Glucose (mg/dL)']:.2f}")
        print(f"Time in Range (%):    {average_metrics['Time in Range (%)']:.2f}")
        print(f"Time Hypo (%):        {average_metrics['Time Hypo (%)']:.2f}")
        print(f"Time Hyper (%):       {average_metrics['Time Hyper (%)']:.2f}")
        
        summary_path = os.path.join(results_dir, 'overall_summary.csv')
        results_df.to_csv(summary_path, index=False, float_format='%.2f')
        print(f"\n✅ Saved overall summary table to {summary_path}")
        print("\n" + "="*70)

        # ############################################################### #
        # ############# NEW SECTION: OVERALL SUMMARY PLOT ############# #
        # ############################################################### #
        print("\n--- Generating Overall Summary Plot ---")
        try:
            # Set 'Patient' as the index for plotting
            plot_data = results_df.set_index('Patient')

            # Select only the percentage columns for the stacked bar chart
            time_metrics_df = plot_data[['Time in Range (%)', 'Time Hypo (%)', 'Time Hyper (%)']]
            
            # Create the stacked bar chart
            ax = time_metrics_df.plot(
                kind='bar',
                stacked=True,
                figsize=(16, 9),
                color={'Time in Range (%)': 'green', 'Time Hypo (%)': 'orange', 'Time Hyper (%)': 'red'},
                edgecolor='black',
                width=0.8
            )
            
            # --- Formatting the plot for clarity ---
            plt.title('Glycemic Control Summary Across All Patients', fontsize=18, weight='bold')
            plt.ylabel('Percentage of Time (%)', fontsize=14)
            plt.xlabel('Patient ID', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12) # Rotate labels for better readability
            plt.yticks(fontsize=12)
            plt.ylim(0, 100) # Y-axis from 0 to 100%
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add a horizontal line for the 70% TIR target
            plt.axhline(y=70, color='blue', linestyle='--', label='70% Time in Range Target')
            
            plt.legend(title='Glycemic Range', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout() # Adjust layout to prevent labels/legend overlapping
            
            # Save the figure
            summary_plot_path = os.path.join(results_dir, 'overall_summary_plot.png')
            plt.savefig(summary_plot_path)
            plt.close()
            print(f"✅ Saved overall summary plot to {summary_plot_path}")

        except Exception as e:
            print(f"\n⚠️ Could not generate summary plot. Error: {e}")

    else:
        print("\n⚠️ No results were generated to summarize.")