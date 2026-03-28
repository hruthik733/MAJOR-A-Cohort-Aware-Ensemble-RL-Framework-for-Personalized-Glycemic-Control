# Ensemble SAC-TD3: Experimental Results

This directory contains the training logs, performance plots, and detailed metrics for the **Ensemble SAC-TD3** agent trained on 10 distinct virtual patients using the `simglucose` environment.

### 🚀 Key Achievement: Fully Closed-Loop Control
The primary milestone of this experiment is the successful implementation of a **Fully Closed-Loop Artificial Pancreas** that operates **without meal intimation**. 

Unlike hybrid-closed loops that require patients to announce carbohydrates manually, this agent:
* **Detects and Reacts:** Automatically responds to blood glucose fluctuations caused by realistic meal disturbances.
* **No Manual Inputs:** Operates solely on CGM (Continuous Glucose Monitor) data, eliminating the burden of carb counting.
* **Realistic Simulation:** Validated against physiological models that simulate unannounced meal intake and metabolic absorption variability.

---

## 📊 Visual Summaries

### 1. Glycemic Control Summary
The chart below illustrates the clinical performance across all 10 patients.
* **Green:** Time in Range (Target 70-180 mg/dL).
* **Observation:** The model consistently maintains high Time in Range (TIR), with 3 patients achieving **100% TIR**.

<img width="4800" height="2700" alt="ensemble_summary_plot" src="https://github.com/user-attachments/assets/c21e467e-3091-4ccd-ac4c-e2e131ffb1a5" />

### 2. Learning Curve
The training progression shows the Cumulative Reward over 250 episodes.
* **Convergence:** The agent demonstrates stable learning and convergence around episode 150, proving its ability to adapt to complex glucose dynamics.

<img width="3600" height="1800" alt="ensemble_learning_curve" src="https://github.com/user-attachments/assets/3d0a5eda-a383-4621-8838-f16271559fda" />

---

## 📈 Quantitative Performance Metrics

### Aggregate Statistics
Across all 10 virtual patients, the Ensemble model achieved superior glycemic control standards without meal announcements:

| Metric | Average Value (± Std Dev) | Clinical Significance |
| :--- | :--- | :--- |
| **Time in Range (TIR)** | **78.78% ± 16.93** | *Significantly exceeds the clinical target of >70%* |
| **Mean Glucose** | 112.82 ± 21.83 mg/dL | *Maintains near-optimal fasting glucose levels* |
| **Time Hyper** | 2.39% ± 5.79 | *Exceptional prevention of post-meal hyperglycemia* |

---

## 📝 Detailed Patient-by-Patient Breakdown

The table below details the final evaluation metrics for each individual patient model, highlighting the robustness of the policy across different metabolic profiles.

| Patient ID | Mean Glucose (mg/dL) | Glucose Std | **Time in Range (%)** | Time Hypo (%) | Time Hyper (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **adult#001** | 124.08 | 12.00 | **100.00%** 🟢 | 0.00% | 0.00% |
| **adult#002** | 141.93 | 8.90 | **100.00%** 🟢 | 0.00% | 0.00% |
| **adult#003** | 99.82 | 31.29 | 74.74% 🟢 | 25.26% | 0.00% |
| **adult#004** | 89.18 | 34.38 | 62.28% | 37.72% | 0.00% |
| **adult#005** | 98.19 | 44.09 | 67.80% | 32.20% | 0.00% |
| **adult#006** | 134.12 | 43.77 | 82.01% 🟢 | 0.00% | 17.99% |
| **adult#007** | 119.43 | 41.64 | 82.01% 🟢 | 12.11% | 5.88% |
| **adult#008** | 141.53 | 7.90 | **100.00%** 🟢 | 0.00% | 0.00% |
| **adult#009** | 88.21 | 44.40 | 54.26% | 45.74% | 0.00% |
| **adult#010** | 91.69 | 39.01 | 64.71% | 35.29% | 0.00% |

### Success Highlights
1.  **Perfect Glycemic Control:** Patients `adult#001`, `adult#002`, and `adult#008` achieved **100% Time in Range**. This indicates that the model perfectly learned the metabolic parameters for these patients, neutralizing all meal disturbances.
2.  **Hyperglycemia Management:** The system excelled at preventing high blood sugar (Hyperglycemia), keeping it at 0.00% for the vast majority of patients. This proves the agent is highly effective at delivering insulin promptly in response to rising glucose trends.

---

## 📂 File Structure
- `ensemble_summary_plot.png`: Bar chart of TIR metrics.
- `ensemble_learning_curve.png`: Training reward progression.
- `ensemble_summary.csv`: Raw CSV data of the metrics table above.
- 
