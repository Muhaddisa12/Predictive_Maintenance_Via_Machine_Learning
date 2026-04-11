# Feature Engineering

[← Back to README](../README.md) | [← EDA](01_eda.md)

---

## Overview

Raw sensor readings alone do not capture the full physics of machine failure. By combining domain knowledge of how each failure mode works, we create three engineered features that significantly improve predictive power. Each feature directly encodes a known failure mechanism from the dataset documentation.

---

## Column Cleaning

XGBoost and many sklearn tools reject feature names containing `[`, `]`, or `<`. The original dataset has column names like `'Air temperature [K]'`. We sanitise all names first:

```python
df.columns = (df.columns
              .str.replace('[', '_', regex=False)
              .str.replace(']', '_', regex=False)
              .str.replace(' ', '_', regex=False)
              .str.strip('_'))
```

**Before → After:**
```
'Air temperature [K]'       →  'Air_temperature__K'
'Process temperature [K]'   →  'Process_temperature__K'
'Rotational speed [rpm]'    →  'Rotational_speed__rpm'
'Torque [Nm]'               →  'Torque__Nm'
'Tool wear [min]'           →  'Tool_wear__min'
'Machine failure'           →  'Machine_failure'
```

---

## Feature 1 — Machine Type Encoding

```python
le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])
# H=0, L=1, M=2  (alphabetical order)
```

**Why:** The `Type` column (L/M/H — Low/Medium/High quality) has different failure thresholds in the dataset definition. Low-quality machines have tighter limits and higher observed failure rates. Dropping this column loses meaningful information.

| Type | Encoded | Failure Rate |
|---|---|---|
| H (High) | 0 | ~0.1% |
| L (Low)  | 1 | ~5.1% |
| M (Medium) | 2 | ~2.7% |

---

## Feature 2 — Temperature Difference (`Temp_diff`)

```python
df['Temp_diff'] = df['Process_temperature__K'] - df['Air_temperature__K']
```

**Why:** Heat Dissipation Failure (HDF) — the second most common failure mode — is triggered when the machine cannot cool itself. The cooling capacity is determined by *how much cooler* the surrounding air is compared to the process temperature. When this gap is too small at low RPM, heat builds up and HDF occurs.

Using `Process_temp` and `Air_temp` separately gives the model less information than providing their difference directly.

![Temp Diff Distribution](../assets/plots/engineered_features.png)

**Distribution insight:** Normal machines maintain a larger temperature differential. When `Temp_diff` is small (poor heat dissipation), failure probability rises significantly.

---

## Feature 3 — Power (`Power`)

```python
df['Power'] = df['Torque__Nm'] * (df['Rotational_speed__rpm'] * 2 * np.pi / 60)
```

**Physics:** Power (Watts) = Torque (Nm) × Angular velocity (rad/s), where ω = RPM × 2π/60.

**Why:** Power Failure (PWF) occurs when power consumption falls outside the safe operating range of **3,500W to 9,000W**. Raw torque and RPM individually don't capture this — their product does. This is a direct encoding of the failure condition.

| Power Range | Risk |
|---|---|
| < 3,500 W | PWF risk (underload) |
| 3,500 – 9,000 W | Safe operating zone |
| > 9,000 W | PWF risk (overload) |

---

## Feature 4 — Wear-Torque Interaction (`Wear_Torque`)

```python
df['Wear_Torque'] = df['Tool_wear__min'] * df['Torque__Nm']
```

**Why:** Overstrain Failure (OSF) — the most common failure mode — is triggered when the product of tool wear and torque exceeds a machine-type-specific threshold. For example, for Type L machines: `Tool_wear × Torque > 11,000 Nm·min`. This interaction term directly encodes the overstrain failure condition.

| Machine Type | OSF Threshold (Torque × Tool Wear) |
|---|---|
| L (Low) | > 11,000 Nm·min |
| M (Medium) | > 12,000 Nm·min |
| H (High) | > 13,000 Nm·min |

---

## Final Feature Set

```python
feature_cols = [
    'Type_encoded',           # Machine quality tier
    'Air_temperature__K',     # Raw sensor
    'Process_temperature__K', # Raw sensor
    'Rotational_speed__rpm',  # Raw sensor
    'Torque__Nm',             # Raw sensor
    'Tool_wear__min',         # Raw sensor
    'Temp_diff',              # Engineered → HDF risk
    'Power',                  # Engineered → PWF risk
    'Wear_Torque'             # Engineered → OSF risk
]

X = df[feature_cols]
y = df['Machine_failure']
```

---

## Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Stratified split** ensures both train and test sets preserve the original 96.6%/3.4% class ratio. Without stratification, the small test set could end up with disproportionate failure counts, making evaluation unreliable.

```
Training set : 8,000 samples  | 271 failures (3.39%)
Test set     : 2,000 samples  | 68 failures  (3.40%)
```

---

## Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

> **Important:** The scaler is **fitted on training data only** and then applied to test data. Fitting on the full dataset would constitute data leakage — the model would have indirect knowledge of test set statistics.

Scaling is used for the Logistic Regression component inside the Ensemble. Tree-based models (XGBoost, Random Forest) are scale-invariant and do not require normalisation.

---

## Impact of Engineered Features

After model training, SHAP analysis and XGBoost feature importance both confirm that the engineered features rank among the most important predictors — validating that domain knowledge meaningfully improves model performance beyond raw sensors alone.

| Feature | Type | Maps to Failure Mode |
|---|---|---|
| Wear_Torque | Engineered | OSF — most common failure |
| Power | Engineered | PWF — power boundary condition |
| Temp_diff | Engineered | HDF — heat dissipation capacity |
| Type_encoded | Encoded categorical | All failure modes (type-specific thresholds) |

---

[Next: Balanced Random Forest →](03_model_balanced_rf.md)
