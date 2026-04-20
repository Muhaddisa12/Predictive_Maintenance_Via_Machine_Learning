# Predictive Maintenance Using Machine Learning
### Applied to Industrial Sensor Data | AI4I 2020 Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/XGBoost-Best%20Model-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/F1%20Score-0.703-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.978-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-success?style=for-the-badge"/>
</p>

---

## Overview

Unplanned machine failure in manufacturing costs billions annually in downtime, repairs, and lost production. This project builds a machine learning system that uses real-time sensor readings — temperature, rotational speed, torque, and tool wear — to **predict machine failure before it happens**, enabling maintenance teams to act proactively instead of reactively.

This project was built to demonstrate the application of AI/ML to real industrial problems, directly relevant to large-scale chemical and manufacturing operations.

---

## Business Problem

| Without Predictive Maintenance | With This System |
|---|---|
| Failures discovered after damage | Failures predicted hours/days ahead |
| Reactive, costly emergency repairs | Planned, cheaper scheduled maintenance |
| Unplanned production downtime | Maximised equipment uptime |
| No data-driven decision making | Evidence-based maintenance scheduling |

> **At scale (10,000 machines): XGBoost catches 259 of 339 failures per year, saving an estimated $7M+ annually compared to no predictive system.**

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository — AI4I 2020](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) |
| Samples | 10,000 industrial sensor readings |
| Features | 9 (5 raw sensors + 3 engineered + machine type) |
| Target | Binary: `Machine failure` (1) or `Normal` (0) |
| Class imbalance | ~96.6% Normal / ~3.4% Failure |
| Challenge | Severe class imbalance requiring special handling |

### Sensor Features

| Feature | Unit | Description |
|---|---|---|
| Air temperature | K | Ambient temperature around machine |
| Process temperature | K | Internal operating temperature |
| Rotational speed | rpm | Spindle rotation speed |
| Torque | Nm | Rotational force applied |
| Tool wear | min | Accumulated tool usage time |
| Type | L/M/H | Machine quality tier |

### Failure Modes in Dataset

| Code | Name | Description |
|---|---|---|
| TWF | Tool Wear Failure | Tool exceeds wear threshold (200–240 min) |
| HDF | Heat Dissipation Failure | Insufficient temperature differential at low RPM |
| PWF | Power Failure | Power consumption outside safe range [3500W–9000W] |
| OSF | Overstrain Failure | Torque × Tool wear exceeds machine-tier threshold |
| RNF | Random Failure | Random 0.1% chance — unpredictable by any model |

---

## Project Structure

```
predictive-maintenance/
│
├── predictive_maintenance_IVL.ipynb   ← Main analysis notebook
├── predictive_maintenance.csv         ← Dataset
├── best_model_xgboost.pkl             ← Saved XGBoost model
├── scaler.pkl                         ← Feature scaler
├── label_encoder.pkl                  ← Type encoder
├── feature_cols.json                  ← Feature list for deployment
├── app.py                             ← Streamlit deployment app
│
├── README.md                          ← You are here
│
└── docs/
    ├── 01_eda.md                      ← Exploratory Data Analysis
    ├── 02_feature_engineering.md      ← Feature Engineering
    ├── 03_model_balanced_rf.md        ← Balanced Random Forest
    ├── 04_model_xgboost.md            ← XGBoost (Best Model)
    ├── 05_model_rf_threshold.md       ← RF + Threshold Tuning
    ├── 06_model_ensemble.md           ← Soft Voting Ensemble
    ├── 07_model_comparison.md         ← Full Model Comparison
    └── 08_cost_analysis.md            ← Business & Cost Analysis
```

---

## Methodology

```
Raw Data (10,000 rows)
        │
        ▼
   Data Loading & Cleaning
   (column sanitisation, null checks)
        │
        ▼
   Exploratory Data Analysis
   (distributions, correlations, failure patterns)
        │
        ▼
   Feature Engineering
   (Temp_diff, Power, Wear_Torque, Type encoding)
        │
        ▼
   Train / Test Split (80/20, stratified)
        │
        ├──► Balanced Random Forest
        ├──► XGBoost (scale_pos_weight)
        ├──► RF + SMOTETomek + Threshold Tuning
        └──► Soft Voting Ensemble
                │
                ▼
        Model Evaluation
        (Confusion Matrix, ROC, PR curves, CV)
                │
                ▼
        SHAP Explainability
                │
                ▼
        Business Impact Analysis
        (Operational metrics, Cost-benefit)
                │
                ▼
        Model Saved → Streamlit App
```

---

## Results Summary

### Model Performance

| Model | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|
| Balanced Random Forest | 0.941 | 0.234 | 0.375 | 0.974 |
| **XGBoost**  | **0.765** | **0.650** | **0.703** | **0.978** |
| RF + Threshold Tuning | 1.000 | 0.034 | 0.066 | 0.974 |
| Soft Voting Ensemble | 0.794 | 0.587 | 0.675 | 0.976 |

> **XGBoost selected as production model** — best F1 score and highest precision, making it the most operationally trustworthy.

### Business Impact (10,000 machines/year)

| Model | Failures Caught | False Alarms | Total Cost (USD) |
|---|---|---|---|
| XGBoost | 259 / 339 | 140 | $8,140,000 |
| Ensemble | 269 / 339 | 189 | $7,189,000 |
| Balanced RF | 319 / 339 | 1,050 | $3,050,000 |

> *Cost model: $100,000 per missed failure, $1,000 per false alarm.*
> See [full cost analysis →](docs/08_cost_analysis.md)

---

## Key Findings

1. **Torque and Tool Wear are the strongest failure predictors** — confirmed by both XGBoost feature importance and SHAP values
2. **Engineered feature `Wear_Torque` (Torque × Tool Wear)** outperforms both raw features in predictive power, directly mapping to Overstrain Failure mechanics
3. **High recall ≠ good model** — Balanced RF achieves 94% recall but generates 1,050 false alarms/year, causing alert fatigue that defeats the system's purpose
4. **Machine type L (Low quality) has highest failure rate** — type encoding adds measurable predictive value
5. **SHAP analysis confirms domain logic** — the model's reasoning matches engineering knowledge, making it explainable and trustworthy for industrial deployment

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Deployment | Streamlit, Joblib |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-ivl.git
cd predictive-maintenance-ivl

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook predictive_maintenance_IVL.ipynb

# Launch the Streamlit app
streamlit run app.py
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
streamlit>=1.20.0
joblib>=1.2.0
```

---

## Detailed Documentation

| Section | Link |
|---|---|
| Exploratory Data Analysis | [docs/01_eda.md](docs/01_eda.md) |
| Feature Engineering | [docs/02_feature_engineering.md](docs/02_feature_engineering.md) |
| Balanced Random Forest | [docs/03_model_balanced_rf.md](docs/03_model_balanced_rf.md) |
| XGBoost (Best Model) | [docs/04_model_xgboost.md](docs/04_model_xgboost.md) |
| RF + Threshold Tuning | [docs/05_model_rf_threshold.md](docs/05_model_rf_threshold.md) |
| Soft Voting Ensemble | [docs/06_model_ensemble.md](docs/06_model_ensemble.md) |
| Model Comparison | [docs/07_model_comparison.md](docs/07_model_comparison.md) |
| Business & Cost Analysis | [docs/08_cost_analysis.md](docs/08_cost_analysis.md) |

---

## Author

**Muhaddisa**
BS Artificial Intelligence — 6th Semester
Shifa Tameer-e-Millat University, Pakistan

---

*Dataset: Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository.*
