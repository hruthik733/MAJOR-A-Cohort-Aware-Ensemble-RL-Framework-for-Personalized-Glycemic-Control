# Personalized Diabetes Management using Reinforcement Learning

> Extension work from MINI02 to MAJOR Project — A fully closed-loop Artificial Pancreas (AP) system powered by a Trainable Ensemble Reinforcement Learning agent.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Overall Workflow Diagram](#overall-workflow-diagram)
4. [Component Details](#component-details)
   - [Environment & Simulation](#environment--simulation)
   - [State Representation](#state-representation)
   - [Ensemble Agent (Meta-Controller)](#ensemble-agent-meta-controller)
   - [Reward Function](#reward-function)
   - [Safety Layer](#safety-layer)
   - [Replay Buffer](#replay-buffer)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Project Structure](#project-structure)
8. [Installation & Requirements](#installation--requirements)
9. [Usage](#usage)
10. [Clinical Metrics](#clinical-metrics)

---

## Project Overview

This project implements a **Personalized Closed-Loop Artificial Pancreas** system for Type 1 Diabetes (T1D) management. The system uses **Reinforcement Learning (RL)** to learn optimal continuous insulin dosing strategies for different patient cohorts (children, adolescents, adults) without relying on meal announcements.

**Key innovation:** A **Trainable Ensemble Agent** that dynamically combines the strengths of two RL algorithms — **SAC (Soft Actor-Critic)** and **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** — via a learned **Meta-Controller**, resulting in more robust and personalized insulin dosing.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERSONALIZED AP SYSTEM                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │  simglucose  │    │           ENSEMBLE AGENT                 │   │
│  │  Simulator   │    │  ┌────────────┐    ┌────────────────┐    │   │
│  │              │◄───┤  │ SAC Agent  │    │   TD3 Agent    │    │   │
│  │  T1D Patient │    │  │ (Stochastic│    │(Deterministic) │    │   │
│  │  + Realistic │    │  │  Policy)   │    │                │    │   │
│  │  Meal Scen.  │    │  └─────┬──────┘    └───────┬────────┘    │   │
│  └──────┬───────┘    │        │  action_sac  action_td3 │       │   │
│         │            │        └──────────┬───────────────┘       │   │
│   obs   │            │               ┌───▼──────────────┐        │   │
│ (glucose│            │               │  Meta-Controller │        │   │
│  mg/dL) │            │               │  (learns weights │        │   │
│         │            │               │   w_sac, w_td3)  │        │   │
│         │            │               └────────┬─────────┘        │   │
│         │            │                        │ blended action    │   │
│         │            └────────────────────────┼──────────────────┘   │
│         │                                     │                      │
│         │         ┌───────────────────────┐   │                      │
│         │         │    State Manager      │   │                      │
│         ├────────►│ [glucose, RoC, IOB,   │   │                      │
│         │         │  body_weight]         │   │                      │
│         │         └──────────┬────────────┘   │                      │
│         │                    │ 4D state        │                      │
│         │         ┌──────────▼────────────┐   │                      │
│         │         │    Safety Layer       │◄──┘                      │
│         │         │  (cohort-aware rules) │                          │
│         │         └──────────┬────────────┘                          │
│         │                    │ safe insulin dose                     │
│         └────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Overall Workflow Diagram

### Training Workflow (`train_ensemble_cohort.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘

  CLI: python train_ensemble_cohort.py --cohort [child|adolescent|adult] --seed 42
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 1: SETUP                                             │
│  • Detect device (CUDA / CPU)                              │
│  • Set random seeds (Python, NumPy, PyTorch)               │
│  • Define hyperparameters:                                  │
│      max_episodes=600, timesteps_per_ep=288                │
│      batch_size=256, replay_buffer=1M, warmup=2500         │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 2: ENVIRONMENT INITIALIZATION (10 patients)          │
│  • For each patient in cohort (e.g., adult#001…adult#010)  │
│      ├─ Load T1DPatient physiological parameters           │
│      ├─ Create RealisticMealScenario                       │
│      │    └─ Generates 6 meals/day based on body weight    │
│      │       (probabilistic timing, truncated-normal)      │
│      ├─ Register Gymnasium env (simglucose)                │
│      └─ Store env & patient body weight (BW)               │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 3: INITIALIZE COMPONENTS                             │
│  • EnsembleAgent(state_dim=4, action_dim=1)                │
│      ├─ SACBaselineAgent   (stochastic policy)             │
│      ├─ TD3BaselineAgent   (deterministic policy)          │
│      └─ MetaController     (weight mixer, 64-unit MLP)     │
│  • StateRewardManager  (state construction + reward)       │
│  • SafetyLayer(cohort=args.cohort)                         │
│  • ReplayBuffer(capacity=1,000,000)                        │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│  STEP 4: TRAINING LOOP  (600 episodes)                     │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Per Episode:                                        │  │
│  │  1. Sample a random patient from cohort              │  │
│  │  2. Reset environment & StateRewardManager           │  │
│  │  3. Construct initial 4D state:                      │  │
│  │     [glucose, rate_of_change, IOB, body_weight]      │  │
│  │  4. Normalize state (online running statistics)      │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                  │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Per Timestep (288 steps = 24 hours, 5-min interval):│  │
│  │                                                      │  │
│  │  if total_steps < 2500 (warmup):                     │  │
│  │      action = random ∈ [-1, 1]                       │  │
│  │  else:                                               │  │
│  │      (raw_action, w_sac, w_td3) =                    │  │
│  │          EnsembleAgent.select_action(state)          │  │
│  │      ├─ SAC samples stochastic action                │  │
│  │      ├─ TD3 samples deterministic action + noise     │  │
│  │      └─ Meta-Controller blends: w_sac·a_sac +        │  │
│  │                                 w_td3·a_td3          │  │
│  │                                                      │  │
│  │  Map action → insulin_dose (cohort-specific scaling):│  │
│  │      Adult:       dose = norm_action * 2.0  (linear) │  │
│  │      Adolescent:  dose = norm_action² * 0.75         │  │
│  │      Child:       dose = norm_action² * 0.25         │  │
│  │                                                      │  │
│  │  Apply SafetyLayer rules:                            │  │
│  │      ├─ Hard stop if glucose < 80 mg/dL              │  │
│  │      ├─ Predictive stop if BG-in-20min < 75          │  │
│  │      ├─ IOB stacking cutoff (cohort-specific)        │  │
│  │      └─ Halve dose if dropping fast (RoC < -1.5)     │  │
│  │                                                      │  │
│  │  Step simulation → next glucose observation          │  │
│  │  Build next 4D state & compute reward                │  │
│  │  Push (state, action, reward, next_state, done)      │  │
│  │      → ReplayBuffer                                  │  │
│  │                                                      │  │
│  │  if total_steps > 2500 (learning phase):             │  │
│  │      EnsembleAgent.update(buffer, batch=256)         │  │
│  │      ├─ Update SAC (actor + critic + target)         │  │
│  │      ├─ Update TD3 (critic every step,               │  │
│  │      │              actor every 2 steps)             │  │
│  │      └─ Update MetaController:                       │  │
│  │           meta_loss = -min(Q(s, a_ens))              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Checkpointing:                                            │
│  • Rolling average over last 10 episodes                   │
│  • Save best_model.pth when avg improves                   │
│  • Save model_final.pth at episode 600                     │
└────────────────────────────────────────────────────────────┘
```

### Evaluation Workflow (`test_ensemble_cohort.py`)

```
  CLI: python test_ensemble_cohort.py --cohort adult --model_path ./models/...
                          │
                          ▼
  Load EnsembleAgent weights (SAC + TD3 + MetaController)
                          │
                          ▼
  For each of 10 patients in cohort:
  ├─ Warmup phase (normalize state statistics)
  ├─ Evaluation phase (288 timesteps = 1 day)
  │   ├─ Select deterministic action (evaluate=True)
  │   └─ Record glucose & insulin traces
  │
  ├─ Compute Clinical Metrics:
  │   ├─ TIR   – Time In Range (70–180 mg/dL)
  │   ├─ Hypo  – Time Below Range (< 70 mg/dL)
  │   ├─ Severe Hypo (< 54 mg/dL)
  │   ├─ Hyper – Time Above Range (> 180 mg/dL)
  │   └─ Mean Blood Glucose
  │
  └─ Save dual-axis plots (glucose + insulin) & CSV summary
```

---

## Component Details

### Environment & Simulation

| Component | Description |
|-----------|-------------|
| **Simulator** | `simglucose` — FDA-accepted UVA/Padova T1D simulator |
| **Patient Cohorts** | 10 adults, 10 adolescents, 10 children (30 virtual patients total) |
| **Simulation step** | 5 minutes per timestep; 288 steps = 24 hours |
| **Meal Scenario** | `RealisticMealScenario` — probabilistic 6-meal-per-day generator (breakfast, mid-morning snack, lunch, afternoon snack, dinner, evening snack). Meal timing follows truncated-normal distributions; meal size scales with patient body weight. Based on Wang et al. (Biomedicines 2024). |
| **Closed-loop** | No meal announcements to agent — fully autonomous AP mode |

### State Representation

The agent observes a **4-dimensional state vector** at each timestep:

| Dimension | Variable | Description |
|-----------|----------|-------------|
| 1 | `glucose` | Current blood glucose (mg/dL) |
| 2 | `rate_of_change` | Glucose velocity (mg/dL/min), computed over last 5 min |
| 3 | `IOB` | Insulin On Board — active insulin computed via PK/PD gamma curve (peak at 55 min) |
| 4 | `body_weight` | Patient body weight (kg) — personalizes dosing scale |

States are **online-normalized** using a running mean/std (Welford's algorithm) updated every timestep.

### Ensemble Agent (Meta-Controller)

```
              ┌──────────────────────────────────────┐
 4D State ───►│         EnsembleAgent                │
              │                                      │
              │  ┌─────────────┐  ┌───────────────┐  │
              │  │ SAC Actor   │  │  TD3 Actor    │  │
              │  │ (stochastic)│  │(deterministic)│  │
              │  └──────┬──────┘  └───────┬───────┘  │
              │   a_sac │           a_td3 │           │
              │         └────────┬────────┘           │
              │                  │                    │
              │  ┌───────────────▼──────────────────┐ │
              │  │      MetaController               │ │
              │  │  Linear(4→64) → ReLU              │ │
              │  │  Linear(64→2) → Softmax → Clamp   │ │
              │  │  weights ∈ [0.2, 0.8]             │ │
              │  └───────────────┬──────────────────┘ │
              │            [w_sac, w_td3]              │
              │                  │                    │
              │   a_ens = w_sac·a_sac + w_td3·a_td3   │
              └──────────────────┬───────────────────┘
                                 │
                           final action
```

**Meta-Controller Training:**
- Loss: `meta_loss = -mean(min(Q1, Q2)(state, a_ens))`
- Maximizes expected Q-value of the blended action
- Weight clamping to `[0.2, 0.8]` prevents degenerate single-agent collapse

### Reward Function

The reward function uses a **9-component multi-zone shaping** design:

| Component | Description |
|-----------|-------------|
| 1. Primary Gaussian | Peak reward at 110 mg/dL (σ=25), bonus for 90–130 range |
| 2. Extended Zone | Moderate reward for 70–180 mg/dL (σ=40) |
| 3. Hypoglycemia Penalty | Severe: –500× at <54; Strong: –250× at <70 |
| 4. Hyperglycemia Penalty | Progressive: gentle at >140, strong at >250 |
| 5. Stability Bonus | +15 for |RoC| < 1.5 in range; –15 for |RoC| > 3.0 |
| 6. Trend Reward | Bonus for moving toward 110 mg/dL |
| 7. IOB Management | Penalty for IOB > 10 U; severe penalty for IOB > 15 U |
| 8. Consistency Bonus | +20 if last 5 steps all in good zones |
| 9. Recovery Bonus | +50 for recovery from hypo; +30 from hyper |

Final reward is clipped to `[-500, 100]` for training stability.

### Safety Layer

Cohort-aware hard-safety rules applied **after** the agent's action:

| Rule | Trigger | Action |
|------|---------|--------|
| Hard Hypo Cutoff | glucose < 80 mg/dL | Dose = 0 |
| Predictive Suspension | predicted BG in 20 min < 75 mg/dL | Dose = 0 |
| IOB Stack Cutoff | IOB ≥ max_safe_iob | Dose = 0 |
| Gentle Brakes | rate_of_change < -1.5 mg/dL/min | Dose × 0.5 |

**Cohort-specific IOB limits:**

| Cohort | Max Safe IOB |
|--------|-------------|
| Child | 1.0 U |
| Adolescent | 2.0 U |
| Adult | 4.0 U |

### Replay Buffer

- Capacity: **1,000,000** transitions
- Stores: `(state, action, reward, next_state, done)` tuples
- Sampling: uniform random sampling
- Shared across all patients in the cohort within one training run

---

## Training Pipeline

```
train_ensemble_cohort.py
│
├── get_cohort_patients()          # Build list of 10 patient IDs
├── T1DPatient.withName()          # Load physiological parameters
├── RealisticMealScenario()        # Generate daily meal plan
├── gymnasium.make()               # Create simulation environment
│
├── EnsembleAgent()
│   ├── SACBaselineAgent()         # Actor (stochastic), Twin Critic
│   ├── TD3BaselineAgent()         # Actor (det.), Twin Critic, Target networks
│   └── MetaController()           # 4→64→2 MLP, Softmax + Clamp
│
├── StateRewardManager()           # State construction + reward calculation
├── SafetyLayer(cohort)            # Hard safety rules
└── ReplayBuffer(1M)               # Experience storage

Training Loop (600 episodes):
  └── Random patient selection per episode
      └── 288 timesteps / episode
          ├── Warmup: random actions (< 2500 steps)
          ├── Exploitation: EnsembleAgent.select_action()
          ├── Cohort-specific dose scaling
          ├── SafetyLayer.apply()
          ├── Simulation step
          ├── StateRewardManager.get_reward()
          ├── ReplayBuffer.push()
          └── EnsembleAgent.update() [after warmup]
              ├── SAC update (actor + critic + target)
              ├── TD3 update (critic + delayed actor + targets)
              └── MetaController update (meta_loss backprop)
```

---

## Evaluation Pipeline

```
test_ensemble_cohort.py
│
├── Load EnsembleAgent from checkpoint
├── Set evaluate=True (deterministic mode)
│
└── For each patient (10 patients):
    ├── Warmup phase: run 1 day to stabilize normalization stats
    ├── Evaluation phase: run 1 day, record traces
    │
    └── Compute metrics:
        ├── TIR   – Time In Range 70–180 mg/dL (%)
        ├── Hypo  – Time < 70 mg/dL (%)
        ├── Severe Hypo – Time < 54 mg/dL (%)
        ├── Hyper – Time > 180 mg/dL (%)
        └── Mean Blood Glucose (mg/dL)
```

---

## Project Structure

```
.
├── train_ensemble_cohort.py      # Main: train Trainable Ensemble agent
├── train_cohort.py               # Train individual SAC or TD3 baseline agent
├── test_ensemble_cohort.py       # Evaluate ensemble model on cohort
├── test_cohort.py                # Evaluate baseline model on cohort
├── test_trainable_ensemble.py    # Additional ensemble evaluation script
├── check_gpu.py                  # GPU availability check
│
├── agents/
│   ├── ensemble_agent.py         # EnsembleAgent + MetaController
│   ├── sac_baseline.py           # SAC Actor, Twin Critic, update logic
│   └── td3_baseline.py           # TD3 Actor, Twin Critic, delayed update
│
├── utils/
│   ├── replay_buffer.py          # Experience replay (deque-based)
│   ├── realistic_scenario.py     # Probabilistic meal scenario generator
│   ├── safety2_closed_loop.py    # Cohort-aware safety layer
│   └── state_management_closed_loop_ensemble.py  # State + reward manager
│
├── simglucose/                   # UVA/Padova T1D simulation engine
│   └── ...
│
├── models/                       # Saved model checkpoints (auto-created)
│   └── trainable_ensemble_<cohort>/
│       ├── best_model.pth
│       └── model_final.pth
│
├── results/                      # Evaluation plots and CSVs (auto-created)
│   └── eval_ensemble_<cohort>/
│       ├── plot_<patient>.png
│       └── ensemble_evaluation_summary.csv
│
├── logs/                         # Training logs
├── requirements_j.txt            # Python dependencies
└── README.md                     # This file
```

---

## Installation & Requirements

**Python 3.8+** is required.

Key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.7.1 | Deep learning framework |
| `gymnasium` | 0.29.1 | RL environment interface |
| `simglucose` | 0.2.11 | T1D glucose simulation |
| `numpy` | 2.2.6 | Numerical computing |
| `scipy` | 1.15.3 | PK/PD curve computation |
| `pandas` | 2.3.1 | Results export |
| `matplotlib` | 3.10.3 | Evaluation plots |

Install dependencies:

```bash
pip install -r requirements_j.txt
```

---

## Usage

### Train the Ensemble Agent

```bash
# Train on the adult cohort
python train_ensemble_cohort.py --cohort adult --seed 42

# Train on the adolescent cohort
python train_ensemble_cohort.py --cohort adolescent --seed 42

# Train on the child cohort
python train_ensemble_cohort.py --cohort child --seed 42
```

### Train Baseline Agents (SAC or TD3)

```bash
# Train SAC on adult cohort
python train_cohort.py --agent sac --cohort adult --seed 42

# Train TD3 on adolescent cohort
python train_cohort.py --agent td3 --cohort adolescent --seed 42
```

### Evaluate the Ensemble Model

```bash
python test_ensemble_cohort.py \
    --cohort adult \
    --model_path ./models/trainable_ensemble_adult/best_model.pth \
    --seed 100
```
 
Output is saved to `./results/eval_ensemble_<cohort>/`:
- `ensemble_evaluation_summary.csv` — per-patient clinical metrics
- `plot_<patient_name>.png` — dual-axis glucose + insulin trace

---

## Clinical Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **TIR** (Time In Range) | ≥ 70% | % time with BG in 70–180 mg/dL |
| **Hypo** | < 4% | % time with BG < 70 mg/dL |
| **Severe Hypo** | < 1% | % time with BG < 54 mg/dL |
| **Hyper** | < 25% | % time with BG > 180 mg/dL |
| **Mean BG** | ~110–140 mg/dL | Average blood glucose over evaluation period |

Clinical targets are based on the **International Consensus on Time in Range** (Battelino et al., 2019).

---

## References

- Battelino, T. et al. (2019). *Clinical Targets for Continuous Glucose Monitoring Data Interpretation*. Diabetes Care.
- Wang, Y. et al. (2024). *Biomedicines*, 12, 2143. (Meal scenario model)
- UVA/Padova T1DMS Simulator — used via the `simglucose` package.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic*. ICML.
- Fujimoto, S. et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*. ICML.
