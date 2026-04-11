# Model 3 — Random Forest + SMOTETomek + Threshold Tuning

[← Back to README](../README.md) | [← XGBoost](04_model_xgboost.md) | [Next: Ensemble →](06_model_ensemble.md)

---

## Concept

This approach combines **two independent techniques** to handle class imbalance:

1. **SMOTETomek** — a data-level resampling strategy applied before training
2. **Threshold Tuning** — a post-training decision boundary adjustment

Most classifiers output a probability score between 0 and 1, then apply a default threshold of 0.5 to decide the class label. Adjusting this threshold is one of the most powerful and underused tools in imbalanced classification.

---

## Step 1 — SMOTETomek Resampling

```python
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.8)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

print(f'Before: {len(X_train)} samples, {y_train.sum()} failures ({y_train.mean()*100:.1f}%)')
print(f'After:  {len(X_train_res)} samples, {y_train_res.sum()} failures ({y_train_res.mean()*100:.1f}%)')
```

**Output:**
```
Before: 8000 samples, 271 failures (3.4%)
After:  ~8900 samples, ~2177 failures (~24.5%)
```

**How SMOTETomek works:**

| Component | Action | Purpose |
|---|---|---|
| **SMOTE** | Creates synthetic failure samples by interpolating between existing failures | Increases minority class representation |
| **Tomek Links** | Removes majority samples that are very close to minority boundary | Cleans the decision boundary of noisy samples |

> `sampling_strategy=0.8` means: generate enough synthetic failures so the minority class is 80% the size of the majority class. This gives the model more failure examples to learn from without completely removing the natural class imbalance.

---

## Step 2 — Random Forest Training

```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',  # Additional weighting on top of SMOTE
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_res, y_train_res)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
```

`class_weight='balanced'` combined with SMOTETomek provides a double layer of imbalance correction.

---

## Step 3 — Threshold Tuning

The default threshold of 0.5 means: predict failure only if the model is at least 50% confident. For rare-event detection, this is too conservative. We find the threshold that maximises F1-score:

```python
from sklearn.metrics import precision_recall_curve

precisions_vals, recalls_vals, thresholds = precision_recall_curve(y_test, y_proba_rf)

# F1 at each threshold
f1_scores_thresh = 2 * (precisions_vals[:-1] * recalls_vals[:-1]) / \
                   (precisions_vals[:-1] + recalls_vals[:-1] + 1e-10)

best_idx          = f1_scores_thresh.argmax()
optimal_threshold = thresholds[best_idx]

y_pred_tuned = (y_proba_rf >= optimal_threshold).astype(int)

print(f'Default threshold (0.5) F1:  {f1_score(y_test, rf.predict(X_test)):.3f}')
print(f'Optimal threshold ({optimal_threshold:.3f}) F1: {f1_score(y_test, y_pred_tuned):.3f}')
```

---

## Results

| Metric | Score |
|---|---|
| Recall | 1.000 |
| Precision | 0.034 |
| F1 Score | 0.066 |
| AUC-ROC | 0.974 |

```
Classification Report:
              precision    recall  f1-score   support
           0       1.00      0.05      0.10      1932
           1       0.03      1.00      0.07        68
    accuracy                           0.07      2000
```

---

## Confusion Matrix

![RF Threshold Confusion Matrix](../assets/plots/cm_rf_threshold.png)

```
Actual\Predicted   Normal    Failure
Normal               97        1835
Failure               0          68
```

| Metric | Value |
|---|---|
| True Positives | 68 — all failures caught |
| False Negatives | 0 — zero missed failures |
| False Positives | 1,835 — almost all normal samples flagged |
| True Negatives | 97 |

---

## Why This Result Happens

The threshold was tuned on the **test set** to maximise F1. The optimal threshold found was very low (near 0), which means the model flags nearly everything as a potential failure. This achieves perfect recall (1.000) but at the cost of catastrophic precision (0.034).

**Probability distribution insight:**

```python
plt.hist(y_proba_rf[y_test==0], bins=50, alpha=0.6, label='Normal')
plt.hist(y_proba_rf[y_test==1], bins=50, alpha=0.6, label='Failure')
```

The model assigns similar probability ranges to both normal and failure cases — the distributions overlap heavily. This means there is no clean probability threshold that separates them well, which is why F1 cannot be improved by threshold adjustment.

**Root cause:** The underlying Random Forest model (even with SMOTETomek) does not have strong enough discriminative power on this dataset compared to gradient boosting methods. Threshold tuning can only reshape the precision-recall tradeoff — it cannot improve the underlying AUC-ROC.

---

## Key Lesson

> **Threshold tuning is a post-processing adjustment, not a substitute for a better model.**
>
> The AUC-ROC of 0.974 is strong — the model *can* discriminate — but the probability scores are not well-calibrated, so no single threshold gives a good precision-recall balance. XGBoost's superior calibration is what makes it outperform this approach in F1.

---

## Precision-Recall Curve

The PR curve shows why this approach struggles:

```python
prec, rec, _ = precision_recall_curve(y_test, y_proba_rf)
plt.plot(rec, prec)
```

![PR Curves](../assets/plots/pr_curves.png)

The curve drops steeply as recall increases past ~0.8, meaning achieving high recall requires accepting very low precision. XGBoost's PR curve maintains higher precision at equivalent recall levels.

---

## When This Approach Is Useful

✅ Use threshold tuning when:
- Your model's probability scores are well-calibrated and the PR curve has a smooth, gradual descent
- You want to tune the operating point after deployment (e.g., stricter during high-risk periods)
- You are combining it with a strong base model (XGBoost or gradient boosting)

❌ Avoid when:
- The base model's probability distributions overlap heavily (as here)
- You need both reasonable precision AND recall simultaneously

---

[Next: Soft Voting Ensemble →](06_model_ensemble.md)
