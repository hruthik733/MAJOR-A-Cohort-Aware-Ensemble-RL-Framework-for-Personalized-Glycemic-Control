# Ensemble SAC-TD3: Experimental Results

This directory contains the training logs, performance plots, and detailed metrics for the **Ensemble SAC-TD3** agent trained on 10 distinct virtual patients using the `simglucose` environment.

### 🚀 Key Achievement: Fully Closed-Loop Control
The primary milestone of this experiment is the successful implementation of a **Fully Closed-Loop Artificial Pancreas** that operates **without meal intimation**. 

Unlike hybrid-closed loops that require patients to announce carbohydrates manually, this agent:
* **Detects and Reacts:** Automatically responds to blood glucose fluctuations caused by realistic meal disturbances.
* **No Manual Inputs:** Operates solely on CGM (Continuous Glucose Monitor) data, eliminating the burden of carb counting.
* **Realistic Simulation:** Validated against physiological models that simulate unannounced meal intake and metabolic absorption variability.

---
Here are the results for the Cohort models for each class (child, adult, adolescent)
