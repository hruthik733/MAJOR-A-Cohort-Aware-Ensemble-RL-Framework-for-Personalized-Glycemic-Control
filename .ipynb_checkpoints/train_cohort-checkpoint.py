import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

from agents.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.safety import SafetyLayer
from utils.state_management import StateRewardManager
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario
from gymnasium.vector import AsyncVectorEnv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env(patient_name, env_seed, meal_scenario, max_timesteps):
    def _init():
        clean_patient_name = patient_name.replace("#", "-")
        env_id = f'simglucose/{clean_patient_name}-v0'
        try:
            register(
                id=env_id,
                entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                max_episode_steps=max_timesteps,
                kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
            )
        except gymnasium.error.Error:
            # Env already registered
            pass
        env = gymnasium.make(env_id)
        env.action_space.seed(env_seed)
        return env
    return _init

def main():
    max_episodes = 2000
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1000000
    max_timesteps_per_episode = 288
    learning_starts = 1000
    num_patients = 10
    adult_patients = [f'adult#{i:03d}' for i in range(1, num_patients + 1)]
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available! Exiting.")
        return

    AGENT_NAME = 'sac'
    model_dir = f'./models/{AGENT_NAME}_cohort'
    results_dir = f'./results/{AGENT_NAME}_cohort'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create vectorized envs and scenarios list
    env_fns = []
    meal_scenarios = []
    for idx, patient_name in enumerate(adult_patients):
        scenario = scgen.RandomScenario(start_time=start_time, seed=SEED + idx)
        meal_scenarios.append(scenario)
        env_fns.append(make_env(patient_name, SEED + idx, scenario, max_timesteps_per_episode))

    vec_env = AsyncVectorEnv(env_fns)
    num_envs = len(env_fns)

    state_dim = 4
    action_dim = 1

    agent = SACAgent(vec_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    managers = [StateRewardManager(state_dim) for _ in range(num_envs)]
    safety_layers = [SafetyLayer() for _ in range(num_envs)]

    episode_rewards = []
    total_timesteps = 0
    env_episode_lens = [[] for _ in range(num_envs)]
    env_final_glucose = [[] for _ in range(num_envs)]

    # Initial reset
    obs_arrays, infos = vec_env.reset(seed=SEED)
    # We expect infos to be dict of arrays
    current_scenarios = meal_scenarios

    unnormalized_states = []
    current_states = []
    for i in range(num_envs):
        managers[i].reset()
        # Access sim_time from infos dict, example: infos['time'][i]
        sim_time = infos['time'][i] if 'time' in infos else 0
        upcoming_carbs = current_scenarios[i].get_action(sim_time).meal if current_scenarios[i] else 0
        unnorm_state = managers[i].get_full_state(obs_arrays[i][0], upcoming_carbs)
        norm_state = managers[i].get_normalized_state(unnorm_state)
        unnormalized_states.append(unnorm_state)
        current_states.append(norm_state)
    unnormalized_states = np.stack(unnormalized_states)
    current_states = np.stack(current_states)

    for ep in range(1, max_episodes + 1):
        if ep > 1:
            obs_arrays, infos = vec_env.reset(seed=SEED + ep)
            current_scenarios = meal_scenarios
            for i in range(num_envs):
                managers[i].reset()
            unnormalized_states = []
            current_states = []
            for i in range(num_envs):
                sim_time = infos['time'][i] if 'time' in infos else 0
                upcoming_carbs = current_scenarios[i].get_action(sim_time).meal if current_scenarios[i] else 0
                unnorm_state = managers[i].get_full_state(obs_arrays[i][0], upcoming_carbs)
                norm_state = managers[i].get_normalized_state(unnorm_state)
                unnormalized_states.append(unnorm_state)
                current_states.append(norm_state)
            unnormalized_states = np.stack(unnormalized_states)
            current_states = np.stack(current_states)

        cumulative_episode_reward = np.zeros(num_envs)
        dones = np.array([False] * num_envs)

        for t in range(max_timesteps_per_episode):
            actions = []
            for i in range(num_envs):
                if total_timesteps < learning_starts:
                    action = np.array([np.random.uniform(0, 0.5)])
                else:
                    action = agent.select_action(current_states[i])
                action = safety_layers[i].apply(action, unnormalized_states[i])
                clipped_action = np.clip(action, 0, 5.0)
                actions.append(clipped_action)
            actions = np.stack(actions)

            next_obs_arrays, _, terminations, truncations, infos = vec_env.step(actions)
            dones = np.logical_or(terminations, truncations)

            next_unnormalized_states = []
            next_normalized_states = []
            rewards = []

            for i in range(num_envs):
                if not dones[i]:
                    managers[i].insulin_history.append(actions[i][0])
                sim_time = infos['time'][i] if 'time' in infos else 0
                upcoming_carbs = current_scenarios[i].get_action(sim_time).meal if current_scenarios[i] else 0
                next_unnorm_state = managers[i].get_full_state(next_obs_arrays[i][0], upcoming_carbs)
                next_norm_state = managers[i].get_normalized_state(next_unnorm_state)
                reward = managers[i].get_reward(unnormalized_states[i])

                next_unnormalized_states.append(next_unnorm_state)
                next_normalized_states.append(next_norm_state)
                rewards.append(reward)

            next_unnormalized_states = np.stack(next_unnormalized_states)
            next_normalized_states = np.stack(next_normalized_states)
            rewards = np.array(rewards)

            for i in range(num_envs):
                replay_buffer.push(current_states[i], actions[i], rewards[i], next_normalized_states[i], dones[i])

            current_states = next_normalized_states
            unnormalized_states = next_unnormalized_states
            cumulative_episode_reward += rewards
            total_timesteps += num_envs

            if total_timesteps > learning_starts and len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
            if all(dones):
                break

        for i in range(num_envs):
            env_episode_lens[i].append(t + 1)
            env_final_glucose[i].append(unnormalized_states[i][0])

        episode_rewards.append(np.mean(cumulative_episode_reward))
        if ep % 50 == 0:
            print(f"Episode {ep}/{max_episodes} | Mean Reward: {np.mean(cumulative_episode_reward):.2f}")

    model_path = f"{model_dir}/actor_cohort.pth"
    torch.save(agent.actor.state_dict(), model_path)
    print(f"Saved shared cohort model to {model_path}")

    # Evaluation per patient sequentially
    all_patient_results = []
    for idx, patient_name in enumerate(adult_patients):
        print(f"\nEvaluating on {patient_name}...")
        eval_scenario = CustomScenario(start_time=start_time, scenario=[(7*60, 45), (12*60, 70), (18*60, 80)])
        clean_patient_name = patient_name.replace('#', '-')
        env_id = f'simglucose/{clean_patient_name}-v0'
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)
        eval_manager = StateRewardManager(state_dim)
        eval_safety = SafetyLayer()
        eval_agent = SACAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
        eval_agent.actor.load_state_dict(torch.load(model_path))

        obs_array, _ = eval_env.reset()
        eval_manager.reset()
        sim_time = 0
        upcoming_carbs = eval_scenario.get_action(sim_time).CHO if eval_scenario else 0
        unnormalized_state = eval_manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = eval_manager.get_normalized_state(unnormalized_state)
        glucose_history = [obs_array[0]]

        for t in range(max_timesteps_per_episode):
            action = eval_agent.select_action(current_state)
            safe_action = eval_safety.apply(action, unnormalized_state)
            clipped_action = np.clip(safe_action, 0, 5.0)
            eval_manager.insulin_history.append(clipped_action[0])
            obs_array, _, terminated, truncated, _ = eval_env.step(clipped_action)
            # eval_env may or may not return 'info' containing time, so manually simulate sim_time here if needed
            sim_time += 3  # assuming steps correspond to 3 minutes or add better timing if you have it
            upcoming_carbs = eval_scenario.get_action(sim_time).CHO if eval_scenario else 0
            unnormalized_state = eval_manager.get_full_state(obs_array[0], upcoming_carbs)
            current_state = eval_manager.get_normalized_state(unnormalized_state)
            glucose_history.append(obs_array[0])
            if terminated or truncated:
                break

        eval_env.close()

        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
        time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
        mean_glucose = np.mean(glucose_history)

        print(f"Mean Glucose: {mean_glucose:.2f} mg/dL")
        print(f"Time in Range: {time_in_range:.2f}% | Hypo: {time_hypo:.2f}% | Hyper: {time_hyper:.2f}%")
        patient_summary = {
            "Patient": patient_name,
            "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range,
            "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper,
        }
        all_patient_results.append(patient_summary)

        plt.figure(figsize=(15, 6))
        plt.plot(glucose_history, label='SAC Agent')
        plt.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia Threshold')
        plt.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia Threshold')
        plt.axhline(y=140, color='g', linestyle='-', label='Target')
        plt.title(f'SAC Agent Performance for {patient_name}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Blood Glucose (mg/dL)')
        plt.legend()
        plt.grid(True)
        plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to {plot_path}")

    results_df = pd.DataFrame(all_patient_results)
    results_df.set_index('Patient', inplace=True)

    print("\n--- Results Per Patient ---")
    print(results_df)

    print("\n--- Average Performance ---")
    print(results_df.mean().to_string())

if __name__ == '__main__':
    main()
