# Model 1 — Balanced Random Forest

[← Back to README](../README.md) | [← Feature Engineering](02_feature_engineering.md) | [Next: XGBoost →](04_model_xgboost.md)

---

## What Is a Balanced Random Forest?

A standard Random Forest trained on imbalanced data will be dominated by the majority class (Normal). It learns to predict "Normal" almost always, achieving high accuracy but near-zero recall on failures.

`BalancedRandomForestClassifier` from `imbalanced-learn` solves this by **undersampling the majority class for each tree** in the forest. Each decision tree is built on a balanced bootstrap sample, forcing every tree to learn from failures equally — unlike a standard forest where failures are rarely seen.

---

## Implementation

```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=200,      # 200 trees in the forest
    max_depth=10,          # Prevent overfitting
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples per leaf
    sampling_strategy='auto',  # Undersample majority to match minority
    replacement=True,      # Sampling with replacement
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

brf.fit(X_train, y_train)
y_pred_brf  = brf.predict(X_test)
y_proba_brf = brf.predict_proba(X_test)[:, 1]
```

**Key parameter — `sampling_strategy='auto'`:**
For each tree, the majority class is undersampled to have the same number of samples as the minority class. This creates balanced bootstrap samples without generating any synthetic data.

---

## Results

| Metric | Score |
|---|---|
| **Recall** | **0.941** |
| Precision | 0.234 |
| F1 Score | 0.375 |
| AUC-ROC | 0.974 |

```
Classification Report:
              precision    recall  f1-score   support
           0       0.99      0.81      0.89      1932
           1       0.23      0.94      0.37        68
    accuracy                           0.82      2000
```

---

## Confusion Matrix

![Balanced RF Confusion Matrix](../assets/plots/cm_balanced_rf.png)

```
Actual\Predicted   Normal    Failure
Normal               1564       368
Failure                 4        64
```

| Metric | Value | Meaning |
|---|---|---|
| True Positives (TP) | 64 | Failures correctly predicted |
| False Negatives (FN) | 4 | Failures missed — dangerous |
| False Positives (FP) | 368 | False alarms — costly |
| True Negatives (TN) | 1564 | Normal correctly identified |

---

## Strengths

- **Highest recall (0.941)** — catches 94% of all failures
- Only **4 missed failures** out of 68 in the test set
- Built-in imbalance handling — no preprocessing required
- AUC-ROC of 0.974 shows excellent discriminative ability at all thresholds

---

## Weaknesses

**Precision is only 0.234** — for every real failure caught, the model generates **3.3 false alarms**.

At scale across 10,000 machines:
- Failures caught: **319** out of 339
- False alarms per year: **1,050**
- Alerts per day: **~3.8**

> A maintenance team receiving 3–4 false alerts daily will quickly lose confidence in the system. Alert fatigue causes engineers to ignore or deprioritise alerts — which can lead to the very failures the system was built to prevent. This is called the **precision-recall tradeoff** and it has real operational consequences.

---

## When to Use This Model

✅ Use Balanced RF when:
- Missing a failure has **catastrophic, irreversible consequences** (safety-critical equipment)
- False alarms are cheap (fully automated inspection line, no human time required)
- The cost of a missed failure is orders of magnitude higher than the cost of a false alarm

❌ Do not use for:
- Standard manufacturing operations where maintenance team time is limited
- Any context where alert fatigue is a concern
- Long-term deployment where team trust in the system matters

---

## AUC-ROC Curve

The AUC-ROC of **0.974** means that in 97.4% of random pairings of a failure instance and a normal instance, the model correctly assigns a higher probability score to the failure. This is a strong discriminative score.

![ROC Curve - All Models](../assets/plots/roc_curves.png)

---

[Next: XGBoost →](04_model_xgboost.md)
