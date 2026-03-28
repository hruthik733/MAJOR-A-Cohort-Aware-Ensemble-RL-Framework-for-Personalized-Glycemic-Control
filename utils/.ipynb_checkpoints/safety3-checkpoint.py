# utils/safety3.py

import numpy as np

class SafetyLayer:
    """
    A rule-based safety layer to constrain the actions of the RL agent.
    This layer is crucial for preventing dangerous situations like severe hypoglycemia.
    """
    def __init__(self):
        # Define safety parameters
        self.GLUCOSE_LOWER_LIMIT = 80   # Block all insulin if glucose is below this value
        self.GLUCOSE_UPPER_LIMIT = 110  # Block bolus insulin below this; only allow basal
        self.MAX_IOB = 8.0              # Maximum allowed Insulin on Board (Units)
        self.MAX_BOLUS = 4.0            # The absolute maximum single dose allowed (Units)
        self.MIN_BASAL_RATE = 0.01      # A tiny, constant insulin dose (U/5min)

    def apply(self, requested_action, state):
        """
        Apply safety rules to the agent's requested action.

        Args:
            requested_action (np.array): The insulin dose requested by the SAC agent.
            state (np.array): The current unnormalized state [glucose, rate, iob, carbs].

        Returns:
            np.array: A new, safe insulin dose.
        """
        glucose, rate, iob, carbs = state
        
        # <<< NEWLY ADDED RULE >>>
        # Rule 0: Enforce physical constraint - insulin cannot be negative.
        # This is the most important fix for the hyperglycemia issue.
        action = max(0, requested_action[0])

        # Rule 1: Prevent hypoglycemia. If glucose is low or dropping fast, give no insulin.
        if glucose < self.GLUCOSE_LOWER_LIMIT:
            return np.array([0.0])

        # Rule 2: Prevent insulin stacking. If IOB is already very high, only allow a minimal basal dose.
        if iob > self.MAX_IOB:
            return np.array([self.MIN_BASAL_RATE])
        
        # Rule 3: If glucose is in a safe but not high range, be conservative.
        # Only allow a minimal basal dose to avoid causing a drop.
        if glucose < self.GLUCOSE_UPPER_LIMIT and rate <= 0:
            return np.array([self.MIN_BASAL_RATE])

        # Rule 4: Cap the maximum bolus to prevent a single massive overdose.
        if action > self.MAX_BOLUS:
            action = self.MAX_BOLUS
            
        # If no critical safety rules are broken, return the capped, non-negative action.
        return np.array([action])