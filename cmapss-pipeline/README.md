# Predictive Maintenance Data Pipeline
### Medallion Architecture on Databricks | PySpark · Delta Lake · Unity Catalog · Lakeview BI

---

## Overview

This project builds a production-style **end-to-end data engineering pipeline** on Databricks using the NASA CMAPSS Turbofan Engine Degradation dataset. The pipeline ingests raw multi-sensor time-series data from 4 sub-datasets, transforms it through a **Bronze → Silver → Gold medallion architecture**, and delivers a Lakeview BI dashboard revealing fleet health insights and sensor degradation patterns.

The domain mirrors real-world industrial use cases - **predictive maintenance in oil refineries, manufacturing plants, and aerospace** - where sensor data from equipment must be collected, cleaned, and made analytics-ready before any intelligent decision-making can happen.

The Silver layer output of this pipeline feeds directly into a companion ML project: **[Turbofan Engine RUL Prediction](#)**.

---

## Architecture

```
NASA CMAPSS Raw .txt Files (12 files)
            │
            ▼
      ┌─────────────┐
      │ BRONZE LAYER│  Raw ingestion, schema assignment, metadata tagging
      │  12 Delta   │  No transformations (data preserved as-is)
      │   Tables    │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │ SILVER LAYER│  Cleaned, unified across 4 sub-datasets
      │  2 Delta    │  RUL computed, normalized, rolling features engineered
      │   Tables    │  ML-ready output
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │  GOLD LAYER │  Aggregated analytics tables
      │  4 Delta    │  Business-level insights for BI consumption
      │   Tables    │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │  LAKEVIEW   │  Interactive BI dashboard
      │  DASHBOARD  │  Fleet health, sensor trends, fault mode analysis
      └─────────────┘
```

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**
- Source: [Kaggle - NASA CMAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- 4 sub-datasets (FD001–FD004) with increasing complexity
- Each dataset contains multivariate time-series sensor readings from a fleet of turbofan engines
- Engines run from healthy state to failure, the objective is to predict Remaining Useful Life (RUL)

| Dataset | Engines (Train) | Engines (Test) | Operating Conditions | Fault Modes |
|---------|----------------|----------------|----------------------|-------------|
| FD001   | 100            | 100            | 1 (Sea Level)        | 1 (HPC Degradation) |
| FD002   | 260            | 259            | 6                    | 1 (HPC Degradation) |
| FD003   | 100            | 100            | 1 (Sea Level)        | 2 (HPC + Fan) |
| FD004   | 248            | 249            | 6                    | 2 (HPC + Fan) |

**Total rows ingested:** ~160,000 training rows across all 4 datasets

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Databricks Community Edition** | Unified analytics platform |
| **Apache Spark / PySpark** | Distributed data processing |
| **Delta Lake** | ACID-compliant storage layer for all tables |
| **Unity Catalog** | Catalog and schema governance (3-layer namespace) |
| **Databricks Volumes** | Landing zone for raw source files |
| **Lakeview Dashboard + Genie AI** | BI layer and natural language analytics |
| **Python (matplotlib)** | EDA visualizations in Bronze layer |

---

## Project Structure

```
cmapss-pipeline/
├── README.md
├── notebooks/
│   ├── 00_setup.py                  - Catalog, schema, volume creation
│   ├── 01_bronze_ingestion.py        - Raw ingestion + EDA
│   ├── 02_silver_transformation.py   - Cleaning, RUL, features
│   └── 03_gold_analytics.py          - Aggregations for BI
└── assets/
    └── dashboard_screenshot.png
```

---

## Pipeline Detail

### 00 — Setup
Created a Unity Catalog with 4 schemas representing each pipeline layer:
- `cmapss_project.source_raw` — Databricks Volume for raw file landing
- `cmapss_project.bronze` — Raw ingested Delta tables
- `cmapss_project.silver` — Cleaned and feature-engineered tables
- `cmapss_project.gold` — Aggregated analytics tables

### 01 — Bronze Layer: Raw Ingestion + EDA
- Ingested all 12 raw `.txt` files (train, test, RUL × 4 datasets) from the Unity Catalog Volume
- Assigned column names from NASA documentation (unit_id, cycle, 3 op settings, 21 sensors)
- Added metadata columns: `dataset_id` and `data_split` to preserve data lineage
- Handled trailing null columns from space-separated raw format
- Assigned `unit_id` to RUL ground truth files using `monotonically_increasing_id()` for downstream joins

**EDA findings from Bronze:**
- Zero null values across all 160,000+ rows - dataset is simulation data, inherently clean
- Sensors 1, 5, 16, 18, 19 showed near-zero standard deviation in FD001 and FD003 (single operating condition datasets) but significant variance in FD002 and FD004. Retained all sensors since dropping would lose signal in multi-condition datasets
- Engine lifespan peaks around 175–200 cycles across all datasets with a right-skewed distribution
- FD003 and FD004 contain outlier engines surviving 500+ cycles due to dual-fault degradation complexity
- Sensor degradation trends are clearly visible in single-condition datasets (FD001/FD003) but masked by operating condition switching in FD002/FD004

### 02 — Silver Layer: Transformation + Feature Engineering
- Unioned all 4 training Bronze tables into a single DataFrame with `dataset_id` preserved
- **RUL computation:** `RUL = max_cycle_per_engine − current_cycle` using Spark Window functions partitioned by `unit_id` AND `dataset_id` (critical, since same unit_id numbers exist across datasets)
- **Normalization:** Min-max scaling per sensor per `dataset_id` - normalized within groups to prevent cross-dataset range distortion. Train statistics used to scale test data (no data leakage)
- **Rolling window features** on 10 key sensors:
  - Rolling mean (5, 10, 30 cycle windows) for smoothed degradation trend
  - Rolling std (10 cycle window) for sensor volatility signal (erratic readings precede failure)
- For test data: joined RUL ground truth from Bronze RUL tables on last observed cycle per engine
- Rolling std nulls in first row of each engine are intentional. Preserved in Silver, handled at ML layer

**Output:** `silver_train` (~160K rows) and `silver_test` (~105K rows), both ML-ready Delta tables

### 03 — Gold Layer: Analytics Aggregations
Four aggregated Delta tables created for BI consumption:

| Table | Description |
|-------|-------------|
| `gold_fleet_health` | Per-engine summary: total cycles, initial RUL, fault mode, avg sensor readings |
| `gold_sensor_trends` | Avg sensor readings bucketed by RUL range (0-25, 26-50, 51-100, 101-150, 150+) |
| `gold_operating_conditions` | Op setting combinations vs average RUL for multi-condition datasets |
| `gold_fault_comparison` | Lifespan statistics (avg, min, max, p25, p75) grouped by fault mode |

---

## Key Findings

**1. Sensor 2 is the strongest degradation indicator in single-condition datasets**
Average sensor_2 reading rises from 0.35 at 150+ cycles RUL to 0.68 in the critical 0-25 cycle window — a clean, monotonic degradation signal that directly informs the RUL prediction model.

**2. Dual-fault engines show higher lifespan variance, not shorter lives**
Average lifespan: Dual Fault = 246.6 cycles vs Single Fault = 206.5 cycles. This is counterintuitive — dual fault engines live longer on average but with much higher variance (max 534 vs 370 cycles), suggesting more varied degradation paths rather than faster failure.

**3. Operating conditions mask degradation signals**
In FD002 and FD004 (6 operating conditions), sensor trends across RUL buckets are less clear because sensors respond to both operating state changes and degradation simultaneously. This is the core challenge of real-world industrial sensor analytics.

**4. Fleet lifespan follows a right-skewed distribution peaking at ~175–200 cycles**
Consistent across all 4 datasets, with a long tail of outlier engines surviving 350–500+ cycles. This distribution informs RUL bucketing strategy for downstream classification tasks.

---

## Dashboard

The Lakeview BI dashboard built on Gold tables delivers 5 visualizations:

- **Average Engine Lifespan by Dataset**: fleet lifespan comparison across all 4 sub-datasets
- **Lifespan Range by Fault Mode**: min/max spread comparison between single and dual fault engines
- **Sensor 2 Degradation by RUL Bucket**: clear staircase degradation pattern as failure approaches
- **Sensor 12 Degradation by RUL Bucket**: secondary degradation signal confirmation
- **Engine Lifespan Distribution**: fleet-wide histogram showing lifespan spread per dataset

---

## What's Next

The Silver layer of this pipeline feeds directly into the companion ML project where an XGBoost model is trained on `silver_train` to predict Remaining Useful Life, tracked and versioned using MLflow on Databricks.

→ **[Turbofan Engine RUL Prediction](#)**

---

## References

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO.
