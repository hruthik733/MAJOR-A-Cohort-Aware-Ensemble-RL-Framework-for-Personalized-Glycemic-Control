# utils/state_management_closed_loop_ensemble.py
# Enhanced reward function optimized for Ensemble SAC-TD3 agent

# utils/state_management_closed_loop_ensemble.py
# Enhanced reward function optimized for Ensemble SAC-TD3 agent (Cohort Ready)

import collections
import numpy as np
from scipy.stats import gamma

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    """
    Pharmacokinetics/Pharmacodynamics insulin absorption curve
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
    Enhanced State and Reward Manager for Ensemble Agent
    
    Key Improvements:
    1. Multi-zone reward structure for better glycemic control
    2. Adaptive penalties based on glucose trends
    3. Reward shaping for stable glucose trajectories
    4. Enhanced hypoglycemia prevention
    """
    
    def __init__(self, state_dim):
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        
        # Enhanced tracking for ensemble - MUST be initialized BEFORE reset()
        self.glucose_zone_history = collections.deque(maxlen=10)
        self.reward_history = collections.deque(maxlen=10)
        
        _, self.F_k = get_pkpd_discount_factors(
            t_peak=55, t_end=480, n_steps=160
        )
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.ones(state_dim)
        self.n_observations = 0
        
        # Now call reset after all attributes are initialized
        self.reset()
        
    def update_normalization_stats(self, state):
        self.n_observations += 1
        old_mean = self.running_state_mean.copy()
        self.running_state_mean += (state - self.running_state_mean) / self.n_observations
        self.running_state_std += (state - old_mean) * (state - self.running_state_mean)
    
    def get_normalized_state(self, state):
        self.update_normalization_stats(state)
        std = np.sqrt(self.running_state_std / (self.n_observations if self.n_observations > 1 else 1))
        return (state - self.running_state_mean) / (std + 1e-8)
    
    def calculate_iob(self):
        """
        Insulin On Board from history and PK/PD curve
        """
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))
    
    def get_full_state(self, observation, bw):
        """
        observation: glucose level (mg/dL)
        bw: patient body weight (kg)
        Returns 4-dimensional state: [glucose, rate_of_change, insulin_on_board, body_weight]
        """
        self.glucose_history.append(observation)
        rate = (
            (self.glucose_history[1] - self.glucose_history[0]) / 3.0
            if len(self.glucose_history) == 2
            else 0.0
        )
        iob = self.calculate_iob()
        return np.array([observation, rate, iob, bw])
    
    def _classify_glucose_zone(self, glucose):
        """
        Classify glucose into zones for zone-based reward shaping
        """
        if glucose < 54:
            return 'severe_hypo'
        elif glucose < 70:
            return 'hypo'
        elif glucose < 80:
            return 'low_normal'
        elif glucose <= 140:
            return 'target'
        elif glucose <= 180:
            return 'high_normal'
        elif glucose <= 250:
            return 'hyper'
        else:
            return 'severe_hyper'
    
    def get_reward(self, state):
        """
        Enhanced Multi-Zone Reward Function for Ensemble SAC-TD3
        """
        # Unpack the 4-dimensional state
        g, rate, iob, bw = state
        reward = 0.0
        
        # Track glucose zone for adaptive learning
        zone = self._classify_glucose_zone(g)
        self.glucose_zone_history.append(zone)
        
        # ===================================================================
        # 1. PRIMARY REWARD: Multi-Zone Gaussian Shaping
        # ===================================================================
        # Target zone (80-140 mg/dL): Maximum reward
        if 80 <= g <= 140:
            # Optimal zone with Gaussian peak at 110 mg/dL
            target_glucose = 110
            sigma = 25.0
            base_reward = 20.0 * np.exp(-0.5 * ((g - target_glucose) / sigma) ** 2)
            reward += base_reward
            
            # Additional bonus for being in tight control (90-130)
            if 90 <= g <= 130:
                reward += 10.0
        
        # Extended target zone (70-180 mg/dL): Good control
        elif 70 <= g <= 180:
            # Moderate reward with softer Gaussian
            target_glucose = 110
            sigma = 40.0
            reward += 12.0 * np.exp(-0.5 * ((g - target_glucose) / sigma) ** 2)
        
        # ===================================================================
        # 2. CRITICAL: Hypoglycemia Prevention (HIGHEST PRIORITY)
        # ===================================================================
        if g < 54:
            # Severe hypoglycemia: catastrophic penalty
            reward -= 500 * (1 + (54 - g) / 10)
            # Extra penalty if still dropping
            if rate < 0:
                reward -= 200 * abs(rate)
        
        elif g < 70:
            # Hypoglycemia: very strong penalty
            severity = (70 - g) / 10
            reward -= 250 * (1 + severity)
            
            # Worse if rapidly dropping
            if rate < -1.0:
                reward -= 150 * abs(rate)
            # Small bonus if recovering (rising)
            elif rate > 0.5:
                reward += 30 * rate
        
        elif g < 80:
            # Low normal: moderate penalty, encourage rise
            reward -= 50 * (80 - g) / 10
            # Bonus for rising toward target
            if rate > 0:
                reward += 20 * rate
        
        # ===================================================================
        # 3. Hyperglycemia Penalties (Progressive)
        # ===================================================================
        if g > 250:
            # Severe hyperglycemia: strong penalty
            reward -= min((g - 250) * 0.8, 150)
            # Bonus if dropping toward target
            if rate < -1.0:
                reward += 40 * abs(rate)
        
        elif g > 180:
            # Moderate hyperglycemia: scaled penalty
            reward -= min((g - 180) * 0.4, 80)
            # Bonus for correcting downward
            if rate < -0.5:
                reward += 25 * abs(rate)
        
        elif g > 140:
            # High normal: gentle penalty
            reward -= (g - 140) * 0.2
        
        # ===================================================================
        # 4. STABILITY BONUS: Reward Smooth Glucose Control
        # ===================================================================
        # Penalize rapid glucose changes (volatility)
        if abs(rate) > 3.0:
            # Excessive volatility penalty
            reward -= 15 * (abs(rate) - 3.0)
        elif abs(rate) < 1.5 and 70 <= g <= 180:
            # Stability bonus for smooth control in target range
            reward += 15 * (1.5 - abs(rate))
        
        # ===================================================================
        # 5. TREND REWARD: Encourage Movement Toward Target
        # ===================================================================
        target_glucose = 110
        distance_from_target = abs(g - target_glucose)
        
        # If glucose is moving toward target, give bonus
        if g < target_glucose and rate > 0:
            # Rising toward target from below
            trend_bonus = min(10 * rate / (1 + distance_from_target / 50), 25)
            reward += trend_bonus
        elif g > target_glucose and rate < 0:
            # Falling toward target from above
            trend_bonus = min(10 * abs(rate) / (1 + distance_from_target / 50), 25)
            reward += trend_bonus
        
        # Penalty if moving away from target
        if g < target_glucose and rate < -0.5:
            # Dropping further below target
            reward -= 20 * abs(rate)
        elif g > target_glucose and rate > 0.5:
            # Rising further above target
            reward -= 15 * rate
        
        # ===================================================================
        # 6. INSULIN ON BOARD (IOB) Management
        # ===================================================================
        # Penalize excessive IOB to prevent insulin stacking
        if iob > 10:
            # High IOB: risk of future hypoglycemia
            reward -= 1.5 * ((iob - 10) ** 1.5)
        elif iob > 5 and g < 100:
            # Moderate IOB with low glucose: dangerous
            reward -= 2.0 * (iob - 5) * (100 - g) / 20
        
        # Small penalty for very high IOB regardless of glucose
        if iob > 15:
            reward -= 0.8 * (iob ** 2)
        
        # ===================================================================
        # 7. CONSISTENCY BONUS: Reward Staying in Target Zone
        # ===================================================================
        # Bonus for consecutive timesteps in good control
        if len(self.glucose_zone_history) >= 5:
            recent_zones = list(self.glucose_zone_history)[-5:]
            target_zones = ['target', 'low_normal', 'high_normal']
            if all(z in target_zones for z in recent_zones):
                # Consistent good control bonus
                reward += 20
        
        # ===================================================================
        # 8. RECOVERY BONUS: Reward Quick Recovery from Extremes
        # ===================================================================
        if len(self.glucose_zone_history) >= 2:
            prev_zone = self.glucose_zone_history[-2]
            curr_zone = zone
            
            # Reward recovery from hypoglycemia
            if prev_zone in ['severe_hypo', 'hypo'] and curr_zone in ['low_normal', 'target']:
                reward += 50
            
            # Reward correction from hyperglycemia
            if prev_zone in ['severe_hyper', 'hyper'] and curr_zone in ['high_normal', 'target']:
                reward += 30
        
        # ===================================================================
        # 9. FINAL REWARD SHAPING: Clip and Normalize
        # ===================================================================
        # Store reward for adaptive learning
        self.reward_history.append(reward)
        
        # Clip extreme rewards for stable learning
        reward = np.clip(reward, -500, 100)
        
        return reward
    
    def get_reward_statistics(self):
        """
        Get statistics about recent rewards for monitoring
        """
        if len(self.reward_history) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        rewards = np.array(self.reward_history)
        return {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards)
        }
    
    def reset(self):
        """Reset all histories"""
        self.glucose_history.clear()
        for _ in range(2):
            self.glucose_history.append(140)
        
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0)
        
        self.glucose_zone_history.clear()
        self.reward_history.clear()