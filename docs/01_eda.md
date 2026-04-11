# Exploratory Data Analysis

[← Back to README](../README.md)

---

## Overview

Before building any model, the data must be fully understood. This section covers distribution analysis, class imbalance investigation, sensor behaviour by failure class, and correlation analysis. Every modeling decision made later is justified by findings here.

---

## 1. Dataset at a Glance

```python
df = pd.read_csv('predictive_maintenance.csv')
print(f'Dataset shape: {df.shape}')
print(df.dtypes)
print(df.isnull().sum())
df.describe().round(2)
```

**Output:**
```
Dataset shape: (10000, 14)

Columns: UDI, Product ID, Type, Air temperature [K], Process temperature [K],
         Rotational speed [rpm], Torque [Nm], Tool wear [min],
         Machine failure, TWF, HDF, PWF, OSF, RNF

Missing values: 0 across all columns
```

**Key observations:**
- 10,000 clean rows with no missing values
- Mix of numerical sensors and categorical machine type
- 5 individual failure mode flags alongside the main binary target

---

## 2. Class Distribution — The Imbalance Problem

```python
counts = df['Machine failure'].value_counts()
pcts   = df['Machine failure'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Bar chart + Pie chart (see notebook for full code)
```

![Class Distribution](../assets/plots/class_distribution.png)

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 9,661 | 96.61% |
| Failure (1) | 339 | 3.39% |

> **Imbalance ratio: 28.5:1**
>
> This is the core challenge. A naive model that always predicts "No Failure" would achieve 96.6% accuracy — but would catch **zero failures**. This is why accuracy is a misleading metric here and why F1-Score and AUC-ROC are used instead.

---

## 3. Machine Type Distribution and Failure Rates

```python
# Failure rate per machine type
failure_by_type = df.groupby('Type')['Machine failure'].mean() * 100
```

![Machine Type Analysis](../assets/plots/machine_type_distribution.png)

| Type | Description | Count | Failure Rate |
|---|---|---|---|
| L | Low quality | ~6,000 | Highest |
| M | Medium quality | ~3,000 | Medium |
| H | High quality | ~1,000 | Lowest |

> **Finding:** Machine Type is a meaningful predictor. Low-quality machines fail significantly more often. This is why `Type` is encoded and included as a feature rather than dropped.

---

## 4. Sensor Feature Distributions

```python
sensor_cols = [
    'Air_temperature__K', 'Process_temperature__K',
    'Rotational_speed__rpm', 'Torque__Nm', 'Tool_wear__min'
]

for col in sensor_cols:
    plt.hist(df[df['Machine_failure']==0][col], bins=40, alpha=0.6, label='Normal')
    plt.hist(df[df['Machine_failure']==1][col], bins=40, alpha=0.6, label='Failure')
```

![Sensor Distributions](../assets/plots/sensor_distributions.png)

**Key observations per sensor:**

| Sensor | Observation |
|---|---|
| Air temperature | Similar distributions — weak individual predictor |
| Process temperature | Slightly higher in failures |
| Rotational speed | **Failures concentrate at LOWER speeds** |
| Torque | **Failures concentrate at HIGHER torque** |
| Tool wear | **Failures increase as wear accumulates** |

---

## 5. Boxplots — Sensor Ranges by Failure Class

```python
sns.boxplot(data=df.astype({'Machine_failure': str}),
            x='Machine_failure', y=col,
            palette={'0': '#4C9BE8', '1': '#E86B4C'})
```

![Sensor Boxplots](../assets/plots/sensor_boxplots.png)

The boxplots confirm the histograms with cleaner separation:
- **Rotational speed:** Failure median is noticeably lower — machines running slower are under more strain
- **Torque:** Failure IQR sits higher — excessive load is a primary failure driver
- **Tool wear:** Wide range for failures — both early and late wear can trigger failure depending on torque

---

## 6. Correlation Heatmap

```python
corr = df[sensor_cols + ['Machine_failure']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r')
```

![Correlation Heatmap](../assets/plots/correlation_heatmap.png)

**Correlation with `Machine_failure` (strongest first):**

| Feature | Correlation |
|---|---|
| Torque | +0.19 |
| Tool wear | +0.10 |
| Rotational speed | -0.19 |
| Process temperature | +0.07 |
| Air temperature | +0.04 |

> **Note:** Individual correlations are weak — this is expected. Machine failure is a complex multi-factor event. This justifies using ensemble tree methods rather than linear models.

**Notable feature correlations:**
- Air temp and Process temp are highly correlated (+0.88) — their **difference** is more informative than either individually → `Temp_diff` engineered feature
- Torque and Rotational speed are negatively correlated (-0.87) — higher torque at lower RPM drives overstrain → `Wear_Torque` engineered feature

---

## 7. Failure Mode Breakdown

```python
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
mode_counts = df[failure_modes].sum().sort_values(ascending=False)
```

![Failure Mode Distribution](../assets/plots/failure_mode_distribution.png)

| Failure Mode | Count | Root Cause |
|---|---|---|
| OSF — Overstrain | ~78 | Torque × Tool wear threshold exceeded |
| HDF — Heat Dissipation | ~76 | Insufficient air-process temp differential |
| PWF — Power | ~65 | Power (Torque × RPM) outside [3500, 9000] W |
| TWF — Tool Wear | ~46 | Tool wear exceeds 200–240 min threshold |
| RNF — Random | ~19 | Completely random — unpredictable |

> **Implication for feature engineering:** OSF and HDF are the most common failures, and both are directly computable from raw sensor readings. This motivates creating `Wear_Torque` (for OSF) and `Temp_diff` (for HDF) as explicit features.

---

## Summary of EDA Findings

| Finding | Action Taken |
|---|---|
| Severe class imbalance (28.5:1) | Used BalancedRF, SMOTETomek, scale_pos_weight |
| Machine Type predicts failure rate | Encoded as ordinal feature |
| Torque + Tool wear drive failure | Created `Wear_Torque` feature |
| Temp differential more informative than raw temps | Created `Temp_diff` feature |
| Power = Torque × RPM maps to PWF | Created `Power` feature |
| Individual correlations weak | Used ensemble methods (RF, XGBoost) |

---

[Next: Feature Engineering →](02_feature_engineering.md)
