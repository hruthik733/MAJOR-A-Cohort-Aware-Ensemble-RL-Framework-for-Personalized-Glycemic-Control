import torch
import torch.nn as nn
import numpy as np
import os

from agents.sac_baseline import SACBaselineAgent
from agents.td3_baseline import TD3BaselineAgent

class MetaController(nn.Module):
    """
    A small neural network that looks at the physiological state and 
    outputs the optimal weights for the SAC and TD3 agents.
    """
    def __init__(self, state_dim):
        super(MetaController, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1) # Ensures weights sum to 1.0
        )
        
    def forward(self, state):
        raw_weights = self.net(state)
        # THE SAFETY CLAMP: 
        # Squeeze the [0.0, 1.0] output into a [0.2, 0.8] range.
        # This guarantees the agent ALWAYS uses at least 20% of the other network,
        # preventing it from finding a fatal "local optima" of 100% SAC.
        clamped_weights = 0.2 + (raw_weights * 0.6) 
        return clamped_weights

class EnsembleAgent:
    """
    State-of-the-art Trainable Ensemble Agent.
    Dynamically learns how to mix SAC and TD3 policies using a Meta-Controller.
    """
    def __init__(self, state_dim, action_dim, max_action, device, 
                 sac_kwargs=None, td3_kwargs=None):
        
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        
        sac_kwargs = sac_kwargs or {}
        td3_kwargs = td3_kwargs or {}

        # 1. Initialize Base Agents
        self.sac_agent = SACBaselineAgent(state_dim, action_dim, max_action, device, **sac_kwargs)
        self.td3_agent = TD3BaselineAgent(state_dim, action_dim, max_action, device, **td3_kwargs)
        
        # 2. Initialize Trainable Meta-Controller
        self.meta_controller = MetaController(state_dim).to(device)
        self.meta_optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        """
        Uses the Meta-Controller to dynamically weight the actions.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 1. Get base actions
        sac_action = self.sac_agent.select_action(state, evaluate=evaluate)
        td3_action = self.td3_agent.select_action(state, evaluate=evaluate)
        
        # 2. Get learned weights from Meta-Controller
        with torch.no_grad():
            weights = self.meta_controller(state_tensor).cpu().numpy()[0]
            
        w_sac, w_td3 = weights[0], weights[1]
        
        # 3. Combine actions
        action = (w_sac * sac_action) + (w_td3 * td3_action)
        
        return action, w_sac, w_td3

    def update(self, replay_buffer, batch_size):
        """
        Trains the base agents and the Meta-Controller.
        """
        # 1. Update Base Agents standardly
        self.sac_agent.update(replay_buffer, batch_size)
        self.td3_agent.update(replay_buffer, batch_size)
        
        # 2. Train the Meta-Controller
        state, _, _, _, _ = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        
        # Get deterministic actions from base agents (no gradients to base networks here)
        with torch.no_grad():
            sac_mean, _ = self.sac_agent.actor(state)
            a_sac = torch.tanh(sac_mean) * self.max_action
            a_td3 = self.td3_agent.actor(state)
            
        # Get differentiable weights from Meta-Controller
        weights = self.meta_controller(state)
        w_sac = weights[:, 0].unsqueeze(1)
        w_td3 = weights[:, 1].unsqueeze(1)
        
        # Compute differentiable combined action
        a_ens = (w_sac * a_sac) + (w_td3 * a_td3)
        
        # Evaluate how good this combined action is using SAC's Critic.
        # Gradients MUST flow backward through here to reach the Meta-Controller.
        q1, q2 = self.sac_agent.critic(state, a_ens)
        q_ens = torch.min(q1, q2)
            
        # Meta-Loss: We want to MAXIMIZE Q, so we minimize negative Q
        meta_loss = -q_ens.mean()
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

    def save(self, filepath):
        torch.save({
            'sac_actor': self.sac_agent.actor.state_dict(),
            'sac_critic': self.sac_agent.critic.state_dict(),
            'td3_actor': self.td3_agent.actor.state_dict(),
            'td3_critic': self.td3_agent.critic.state_dict(),
            'meta_controller': self.meta_controller.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.sac_agent.actor.load_state_dict(checkpoint['sac_actor'])
        self.sac_agent.critic.load_state_dict(checkpoint['sac_critic'])
        self.td3_agent.actor.load_state_dict(checkpoint['td3_actor'])
        self.td3_agent.critic.load_state_dict(checkpoint['td3_critic'])
        self.meta_controller.load_state_dict(checkpoint['meta_controller'])