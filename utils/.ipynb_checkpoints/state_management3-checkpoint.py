import collections
import numpy as np
from scipy.stats import gamma

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    """
    Pharmacokinetics/Pharmacodynamics insulin absorption curve.
    This function models how insulin is absorbed by the body over time.
    """
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

class StateRewardManager:
    """
    Manages the state representation, normalization, and reward calculation
    for the reinforcement learning agent.
    """
    def __init__(self, state_dim):
        # Store state_dim as an instance attribute
        self.state_dim = state_dim

        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        
        # Calculate the PK/PD curve parameters once
        _, self.F_k = get_pkpd_discount_factors(
            t_peak=55, t_end=480, n_steps=160
        )
        
        # FIXED NORMALIZATION PARAMETERS to ensure stable state representation
        # These are stable estimates of the mean and standard deviation of the state vector.
        # State: [glucose, rate, iob, carbs]
        self.state_mean = np.array([140.0, 0.0, 5.0, 20.0])
        self.state_std = np.array([40.0, 1.5, 3.0, 30.0])
        
        # Call reset() to set initial history values
        self.reset()

    def get_normalized_state(self, state):
        """
        Normalizes the state using the fixed mean and standard deviation.
        This provides a consistent and stable input for the agent.
        """
        return (state - self.state_mean) / self.state_std

    def calculate_iob(self):
        """
        Calculates Insulin On Board (IOB) from history and the PK/PD curve.
        """
        insulin_array = np.array(list(self.insulin_history)[::-1])
        iob = np.sum(insulin_array * (1 - self.F_k))
        return iob

    def get_full_state(self, observation, upcoming_carbs=0):
        """
        Constructs the full state vector from the current glucose observation.
        State vector: [glucose, rate_of_change, insulin_on_board, upcoming_carbs]
        """
        self.glucose_history.append(observation)
        
        if len(self.glucose_history) == 2:
            # Glucose rate of change (mg/dL per minute), assuming 5-minute intervals
            rate = (self.glucose_history[1] - self.glucose_history[0]) / 5.0
        else:
            rate = 0.0
            
        iob = self.calculate_iob()
        return np.array([observation, rate, iob, upcoming_carbs])

    def get_reward(self, state):
        """
        A tiered, "flat-top" reward function to encourage staying within a safe range.
        - High, flat reward for the optimal zone (90-140).
        - Small positive reward for the acceptable zone (70-180).
        - Penalties outside the safe range.
        """
        g, rate, iob, _ = state
        reward = 0

        # Define glycemic zones
        optimal_zone = (90 <= g <= 140)
        acceptable_zone = (70 <= g < 90) or (140 < g <= 180)

        # 1. Assign reward based on the current zone
        if optimal_zone:
            # High reward for being in the best range.
            # The agent has no incentive to make risky changes if it's already here.
            reward = 1.0
        elif acceptable_zone:
            # Smaller positive reward to encourage entering the optimal zone.
            reward = 0.1
        else:
            # Quadratic penalty for being outside the safe range (70-180).
            if g < 70:
                reward = -0.01 * (70 - g)**2
            elif g > 180:
                reward = -0.001 * (180 - g)**2

        # 2. Add extra penalty for severe hypoglycemia
        if g < 60:
            reward -= 0.2 * (60 - g)**2

        # 3. Penalize high IOB and rate of change to promote stability
        reward -= 0.5 * iob       # Slightly increased IOB penalty
        reward -= 0.2 * rate**2   # Slightly increased rate penalty

        return reward

    def reset(self):
        """
        Resets the history for a new episode.
        """
        self.glucose_history.clear()
        # Start with a safe, neutral glucose level
        self.glucose_history.append(140)
        
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0)