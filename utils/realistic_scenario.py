import numpy as np
from scipy.stats import truncnorm
from simglucose.simulation.scenario import Scenario
from simglucose.patient.t1dpatient import T1DPatient
from datetime import datetime
from collections import namedtuple

# Define the namedtuple with 'meal', which is what env.py expects
PATIENT_ACTION = namedtuple('patient_action', ['meal', 'insulin'])

class RealisticMealScenario(Scenario):
    """
    A realistic meal scenario generator based on the schema from Wang et al. [60]
    as described in Biomedicines 2024, 12, 2143.
    """
    def __init__(self, start_time, patient: T1DPatient, seed=None):
        super().__init__(start_time)
        self.patient = patient
        self.random_gen = np.random.RandomState(seed)
        self.meals = self._generate_daily_meals()

    def _generate_daily_meals(self):
        # (This internal logic is correct and remains unchanged)
        p = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        lo = np.array([5, 9, 10, 14, 16, 20]) * 60
        up = np.array([9, 10, 14, 16, 20, 23]) * 60
        mu_t = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        sigma_t = np.array([60, 30, 60, 30, 60, 30])
        
        bw = self.patient._params['BW']
        mu_a_factors = np.array([0.7, 0.15, 1.1, 0.15, 1.25, 0.15])
        mu_a = mu_a_factors * bw
        sigma_a = mu_a * 0.15

        generated_meals = []
        for k in range(6):
            if self.random_gen.rand() <= p[k]:
                amount = self.random_gen.normal(mu_a[k], sigma_a[k])
                amount = round(max(0, amount))
                a, b = (lo[k] - mu_t[k]) / sigma_t[k], (up[k] - mu_t[k]) / sigma_t[k]
                time = truncnorm.rvs(a, b, loc=mu_t[k], scale=sigma_t[k], random_state=self.random_gen)
                time = round(time)
                if amount > 0:
                    generated_meals.append((time, amount))
        
        generated_meals.sort(key=lambda x: x[0])
        return generated_meals

    def get_action(self, t):
        """
        t is a datetime object from the simulator.
        This method must return a patient_action object with a 'meal' field.
        """
        current_time_in_day = t.hour * 60 + t.minute
        
        for meal_time, meal_amount in self.meals:
            if abs(current_time_in_day - meal_time) < 1e-9:
                # Return a PATIENT_ACTION with the 'meal' keyword
                return PATIENT_ACTION(meal=meal_amount, insulin=0)
        
        # Return a PATIENT_ACTION with the 'meal' keyword
        return PATIENT_ACTION(meal=0, insulin=0)

    def reset(self):
        self.meals = self._generate_daily_meals()