# Turbofan Engine RUL Prediction
### XGBoost · MLflow · Databricks | Regression & Classification

---

## Overview

This project builds two machine learning models to predict the **Remaining Useful Life (RUL)** of turbofan engines using multivariate sensor time-series data. It is the direct continuation of the [Predictive Maintenance Data Pipeline](https://github.com/vbharathmohan/bharath-portfolio/tree/main/cmapss-pipeline) project - the Silver layer Delta table produced by that pipeline is the input to this ML pipeline.

Two complementary models are built:
- **Regression model**: predicts exact RUL in cycles, useful for scheduling precise maintenance windows
- **Classification model**: assigns engines to health zones (Critical / Warning / Moderate / Healthy), useful as a real-time alert system

Both models are tracked and registered using **MLflow** on Databricks, following production ML engineering practices.

This problem maps directly to real-world industrial use cases — replace "turbofan engine" with any rotating equipment in an oil refinery, power plant, or manufacturing facility and the pipeline is identical.

---

## Companion Project

This project depends on the data engineering pipeline:
→ **[Predictive Maintenance Data Pipeline](https://github.com/vbharathmohan/bharath-portfolio/tree/main/cmapss-pipeline)**

The Silver layer table `cmapss_project.silver.silver_train` produced by that pipeline feeds directly into this project's preprocessing notebook.

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**
- Source: [Kaggle — NASA CMAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- 4 sub-datasets (FD001–FD004) with increasing complexity
- 708 engines total across training and test sets
- ~160,000 training rows | 707 test rows (one per engine, last observed cycle)

| Dataset | Engines | Conditions | Fault Modes | Test RMSE | Test Accuracy |
|---------|---------|------------|-------------|-----------|---------------|
| FD001   | 100     | 1          | 1 (HPC)     | 16.03     | 65.0%         |
| FD002   | 259     | 6          | 1 (HPC)     | 17.62     | 73.0%         |
| FD003   | 100     | 1          | 2 (HPC+Fan) | 15.49     | 73.0%         |
| FD004   | 248     | 6          | 2 (HPC+Fan) | 19.72     | 66.5%         |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Databricks Community Edition** | Unified ML platform |
| **XGBoost** | Gradient boosted tree models (regression + classification) |
| **MLflow** | Experiment tracking, model versioning, model registry |
| **PySpark / Delta Lake** | Data loading from Silver layer |
| **Scikit-learn** | Metrics, preprocessing, train/val split |
| **Pandas / NumPy** | Feature manipulation |
| **Matplotlib** | Visualizations |

---

## Project Structure

```
cmapss-ml/
├── README.md
├── notebooks/
│   ├── 04_ml_preprocessing.py      - feature prep, RUL cap, zone labels
│   ├── 05_ml_regression.py         - XGBoost regressor + MLflow
│   └── 06_ml_classification.py     - XGBoost classifier + MLflow
└── assets/
    ├── confusion_matrix.png
    ├── predicted_vs_actual.png
    └── feature_importance.png
```

---

## ML Pipeline

### Preprocessing (Notebook 04)
- Loaded `silver_train` and `silver_test` Delta tables (160,359 rows × 70 columns)
- Filled rolling std null values with 0 — first 9 cycles of each engine have no window history, 0 is semantically correct (no volatility observed yet)
- **RUL capping at 125 cycles** — standard CMAPSS practice. Engines with RUL > 125 are all operationally "healthy" with no urgency distinction. Capping focuses the model on the critical degradation window
- Created zone labels from capped RUL for classification target:
  - `1_Critical` = RUL 0–25
  - `2_Warning`  = RUL 26–50
  - `3_Moderate` = RUL 51–100
  - `4_Healthy`  = RUL 101–125
- Encoded `dataset_id` as integer feature (FD001=1 … FD004=4)
- Final feature set: 65 features (21 raw sensors + 3 op settings + 40 rolling features + dataset_id)
- Saved as `gold_ml_train` (160,359 rows) and `gold_ml_test` (707 rows) Delta tables

### Regression Model (Notebook 05)
**Task:** Predict exact RUL value as a continuous number

**Feature Engineering decisions:**
- Retained all 21 sensors including near-constant ones (sensor_1, 5, 16, 18, 19) - these carry real signal in FD002/FD004 multi-condition datasets and tree models handle uninformative features naturally
- Rolling mean (5, 10, 30 cycles) and rolling std (10 cycles) on 10 key sensors - captures both trend and volatility
- Operating settings included - critical for decoupling condition changes from degradation in multi-condition datasets

**Two runs tracked in MLflow:**

| Run | RMSE (Val) | RMSE (Test) | MAE (Test) | R² (Test) |
|-----|-----------|------------|-----------|----------|
| Baseline (100 trees, depth 6, lr 0.1) | 18.05 | 18.45 | 13.72 | 0.808 |
| Tuned (500 trees, depth 8, lr 0.05)   | 16.38 | **17.90** | **13.12** | **0.819** |

**Key finding — Feature Importance:**
The top features by importance were sensor_5, sensor_1, and sensor_13 — not sensor_2 as EDA on single-condition FD001 suggested. This is because FD002 and FD004 together comprise 72% of training engines, and sensor_1/5 show significant variance under multiple operating conditions. The model weights features that are informative across the majority of the training distribution. Sensor_2 ranked 20th — its clean degradation signal in FD001 is obscured by operating condition noise in multi-condition datasets.

### Classification Model (Notebook 06)
**Task:** Predict which health zone an engine is currently in

**Two runs tracked in MLflow:**

| Run | Accuracy (Val) | Accuracy (Test) | F1 (Test) |
|-----|--------------|----------------|----------|
| Baseline (100 trees, depth 6) | 0.769 | 0.682 | 0.656 |
| Tuned (500 trees, depth 8)    | 0.811 | **0.696** | **0.682** |

**Per-class F1 (Tuned model, test set):**

| Zone | Precision | Recall | F1 |
|------|-----------|--------|----|
| 1_Critical | 0.91 | 0.87 | **0.89** |
| 2_Warning  | 0.54 | 0.55 | 0.55 |
| 3_Moderate | 0.63 | 0.41 | 0.50 |
| 4_Healthy  | 0.68 | 0.89 | **0.77** |

---

## Key Findings

**1. Critical zone safety profile is strong**
Of 143 engines actually in the Critical zone, only 1 was misclassified as Healthy — the most dangerous error type. 125 were correctly identified. The model's mistakes are almost entirely in adjacent zones, which is the expected and acceptable error pattern for a maintenance alert system.

**2. Operating conditions matter more than fault complexity**
Both models perform worse on FD002/FD004 (6 conditions) than FD001/FD003 (1 condition), regardless of fault mode. RMSE gap: 16.03 (FD001) vs 19.72 (FD004). Sensor signals reflect both operating state changes and degradation simultaneously — decoupling these two effects is the core challenge of real-world industrial sensor ML.

**3. Warning and Moderate zones are the hardest to classify**
F1 scores of 0.55 and 0.50 respectively. These middle zones sit in the gradual degradation curve where sensor signals are changing but not yet distinctly failure-like. This is a known challenge in RUL classification — the transition from healthy to degrading is continuous, not discrete.

**4. Regression and classification serve different operational needs**
The regression model (RMSE 17.9) is best for maintenance scheduling — "this engine has approximately 23 cycles left, schedule downtime in 2 weeks." The classifier is best as a real-time alert system — "3 engines just entered Critical zone, escalate immediately." In production, both would run in parallel.

---

## MLflow Experiment Tracking

All runs are logged to two MLflow experiments:
- `/cmapss_rul_regression` — 3 runs (baseline, tuned, registration)
- `/cmapss_rul_classification` — 3 runs (baseline, tuned, registration)

Registered models:
- `cmapss_rul_regressor` — XGBoost regression model, version 1
- `cmapss_rul_classifier` — XGBoost classification model, version 1

---

## Real-World Deployment Context

In a production refinery setting, this system would operate as follows:

```
Sensor data arrives every cycle (real-time stream)
          ↓
Preprocessing pipeline normalizes + computes rolling features
          ↓
Regression model  → "Engine 47 has ~18 cycles remaining"
          ↓
Classifier model  → "Engine 47 is in WARNING zone"
          ↓
Alert system      → Notify maintenance team if zone = Critical or Warning
          ↓
Maintenance team schedules intervention before failure occurs
```

The 707-row test set evaluates point-in-time accuracy. Full `silver_test` (~105K rows) would be used for continuous real-time scoring across the entire engine fleet.

---

## References

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO.
