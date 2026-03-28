# utils/replay_buffer.py

import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size)])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)





# import torch
# import numpy as np

# class VectorizedReplayBuffer:
#     def __init__(self, num_envs, capacity, state_dim, action_dim, device):
#         self.num_envs = num_envs
#         self.capacity = capacity
#         self.device = device
        
#         # Pre-allocate memory on the CPU for efficiency
#         self.states = np.zeros((capacity, num_envs, state_dim), dtype=np.float32)
#         self.actions = np.zeros((capacity, num_envs, action_dim), dtype=np.float32)
#         self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
#         self.next_states = np.zeros((capacity, num_envs, state_dim), dtype=np.float32)
#         self.dones = np.zeros((capacity, num_envs), dtype=np.float32)

#         self.pos = 0
#         self.size = 0

#     def push(self, state, action, reward, next_state, done):
#         # state, action, etc., are now arrays of shape [num_envs, feature_dim]
#         self.states[self.pos] = state
#         self.actions[self.pos] = action
#         self.rewards[self.pos] = reward
#         self.next_states[self.pos] = next_state
#         self.dones[self.pos] = done
        
#         self.pos = (self.pos + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)

#     def sample(self, batch_size):
#         # Randomly sample time step indices
#         indices = np.random.randint(0, self.size, size=batch_size // self.num_envs)
        
#         # Sample from all environments at those time steps and reshape
#         # This creates a flat batch of [batch_size, feature_dim]
#         states_batch = torch.FloatTensor(self.states[indices].reshape(-1, self.states.shape[-1])).to(self.device)
#         actions_batch = torch.FloatTensor(self.actions[indices].reshape(-1, self.actions.shape[-1])).to(self.device)
#         rewards_batch = torch.FloatTensor(self.rewards[indices].reshape(-1)).to(self.device)
#         next_states_batch = torch.FloatTensor(self.next_states[indices].reshape(-1, self.next_states.shape[-1])).to(self.device)
#         dones_batch = torch.FloatTensor(self.dones[indices].reshape(-1)).to(self.device)
        
#         return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch

#     def __len__(self):
#         # Return total number of transitions stored across all environments
#         return self.size * self.num_envs