# Business & Cost Analysis

[← Back to README](../README.md) | [← Model Comparison](07_model_comparison.md)

---

## Overview

Machine learning models are evaluated by F1-score and AUC-ROC in notebooks. They are evaluated by **dollars, downtime, and operational trust** in boardrooms. This section translates model metrics into the language that matters to an industrial company like Indorama Ventures.

---

## Cost Assumptions

```python
COST_MISSED_FAILURE = 100_000   # USD per event
COST_FALSE_ALARM    = 1_000     # USD per event
```

| Cost Item | Value | Includes |
|---|---|---|
| Missed failure | $100,000 | Unplanned downtime, emergency repair, parts, lost production, potential safety incident |
| False alarm | $1,000 | Technician time, unnecessary inspection, minor parts/consumables |

> These are conservative estimates for chemical manufacturing. In high-pressure or hazardous material contexts, the true cost of a missed failure can be $500,000–$5,000,000.

---

## Operational Metrics at Scale

Scaled to **10,000 machines with 339 expected failures per year:**

```python
for name, d in op_models.items():
    cm = confusion_matrix(y_test, d['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    recall_m = tp / (tp + fn)
    prec_m   = tp / (tp + fp)

    failures_caught = int(recall_m * n_failures)
    false_alarms    = int((fp / len(y_test)) * n_machines)
```

| Model | Failures Caught | Failures Missed | False Alarms | Alerts/Day |
|---|---|---|---|---|
| XGBoost | 259 | 80 | 140 | 1.1 |
| Ensemble | 269 | 70 | 189 | 1.3 |
| Balanced RF | 319 | 20 | 1,050 | 3.8 |

---

## Annual Cost Analysis

```python
costs[name] = {
    'Cost of missed failures' : missed  * COST_MISSED_FAILURE,
    'Cost of false alarms'    : fa      * COST_FALSE_ALARM,
    'Total annual cost'       : missed  * COST_MISSED_FAILURE + fa * COST_FALSE_ALARM
}
```

### XGBoost
| Component | Value |
|---|---|
| Failures missed | 80 × $100,000 = **$8,000,000** |
| False alarms | 140 × $1,000 = **$140,000** |
| **Total annual cost** | **$8,140,000** |

### Ensemble
| Component | Value |
|---|---|
| Failures missed | 70 × $100,000 = **$7,000,000** |
| False alarms | 189 × $1,000 = **$189,000** |
| **Total annual cost** | **$7,189,000** |

### Balanced Random Forest
| Component | Value |
|---|---|
| Failures missed | 20 × $100,000 = **$2,000,000** |
| False alarms | 1,050 × $1,000 = **$1,050,000** |
| **Total annual cost** | **$3,050,000** |

---

## Cost Visualisation

![Cost Breakdown](../assets/plots/cost_breakdown.png)

![Total Cost Comparison](../assets/plots/total_cost_comparison.png)

---

## Important Interpretation Note

The cost analysis above shows Balanced RF as the "cheapest" model by total cost. However, this conclusion **depends entirely on the cost assumptions** and hides an operational reality:

> **Balanced RF generates 1,050 false alarms per year — 3.8 per day.**
>
> In practice, maintenance teams that receive constant false alerts will begin to ignore or delay acting on alerts. When a real failure occurs in this "boy who cried wolf" environment, the response time increases — which can turn a preventable failure into a catastrophic one. The $1,000 per false alarm estimate does not capture this systemic trust erosion.

This is why **XGBoost is recommended for most operations** despite a higher estimated cost under these assumptions — its 65% precision builds and maintains operational trust.

---

## Sensitivity Analysis — Breakeven Point

At what cost ratio does each model become optimal?

```
Let x = cost of missed failure
    y = cost of false alarm

Balanced RF beats XGBoost when:
  20x + 1050y < 80x + 140y
  910y < 60x
  x/y > 15.2

→ If missed failure costs more than 15× a false alarm,
  Balanced RF is the cheaper model.
```

| Ratio (Missed / False Alarm) | Optimal Model |
|---|---|
| < 15× | XGBoost |
| 15–25× | Context-dependent |
| > 25× | Balanced RF or Ensemble |

With standard manufacturing costs ($100K / $1K = **100×**), Balanced RF does appear cheapest — but this ignores alert fatigue and long-term operational trust.

---

## No-System Baseline (For Context)

Without any predictive maintenance system, all 339 failures are unplanned:

```
No system:  339 × $100,000 = $33,900,000/year
XGBoost:                      $8,140,000/year
Ensemble:                     $7,189,000/year
```

**Even the worst-performing model saves over $25 million per year** compared to purely reactive maintenance — demonstrating the strong ROI of any ML-based predictive maintenance system.

---

## Business Recommendation

### For standard manufacturing operations
**Deploy XGBoost.** It catches 76.5% of failures while generating only 1.1 alerts per day. The maintenance team will receive meaningful, trustworthy alerts that they act on promptly. Long-term, a trusted system with 65% precision is more valuable than a high-recall system that erodes team confidence.

### For safety-critical or high-value equipment
**Deploy Ensemble.** The additional 10 failures caught per year (at 10,000-machine scale) justifies the extra 49 false alarms per year when each failure risks safety incidents, environmental damage, or production losses above $250,000.

### Universal recommendation
Regardless of model choice: **set up a feedback loop.** When maintenance teams investigate an alert and find no failure, log it as a false positive. Use this data to periodically retrain the model on real operational data, improving precision over time.

---

## Summary Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│           PREDICTIVE MAINTENANCE — DEPLOYMENT SUMMARY        │
├─────────────────────┬─────────────┬─────────────────────────┤
│ Metric              │  XGBoost    │  Ensemble               │
├─────────────────────┼─────────────┼─────────────────────────┤
│ Failures caught/yr  │  259 / 339  │  269 / 339              │
│ False alarms/yr     │  140        │  189                    │
│ Alerts per day      │  1.1        │  1.3                    │
│ Alert precision     │  65%        │  59%                    │
│ F1 Score            │  0.703      │  0.675                  │
│ AUC-ROC             │  0.978      │  0.976                  │
│ Est. annual cost    │  $8.14M     │  $7.19M                 │
│ Savings vs baseline │  $25.76M    │  $26.71M                │
├─────────────────────┼─────────────┼─────────────────────────┤
│ Recommended for     │  Most ops   │  High-risk equipment    │
└─────────────────────┴─────────────┴─────────────────────────┘
```

---

[← Back to README](../README.md)
