# Streamlit Frontend — PredictMaint Dashboard

[← Back to README](../README.md) | [← Cost Analysis](08_cost_analysis.md)

---

## Overview

The frontend is a fully interactive Streamlit dashboard built for the Predictive Maintenance project. It translates the machine learning model into a usable industrial tool — where a maintenance engineer can enter live sensor readings and receive an immediate, explainable failure prediction.

The design language is **industrial monochrome**: dark carbon backgrounds, steel text tones, and amber as the single accent colour — chosen because amber is the universal colour of industrial warning systems, instrument panels, and hazard indicators. Every colour carries meaning. Nothing is decorative.

---

## Design System

| Element | Choice | Rationale |
|---|---|---|
| Font — labels & data | IBM Plex Mono | Designed for technical/engineering interfaces, reads like real instrument displays |
| Font — body text | IBM Plex Sans | Clean complement to the mono font, readable at small sizes |
| Background | `#0d0f12` | Deep carbon — reduces eye strain for long monitoring sessions |
| Accent | `#e8a23a` (amber) | Industrial warning colour — universally understood in manufacturing |
| Normal state | `#3dbc74` (green) | Safe operating signal |
| Failure state | `#dc4f4f` (red) | Immediate danger signal |
| Border/surface | `#1e2329` / `#111418` | Subtle depth without gradients |

---

## Running the App

```bash
# Ensure model files are in the same directory as app.py
# Required: best_model_xgboost.pkl, scaler.pkl, feature_cols.json

pip install -r requirements.txt
streamlit run app.py
```

**Required files in the same folder:**
```
project/
├── app.py
├── best_model_xgboost.pkl     ← Trained XGBoost model
├── scaler.pkl                 ← StandardScaler
├── feature_cols.json          ← Feature column order
└── requirements.txt
```

---

## App Structure

```
PredictMaint Dashboard
│
├── Sidebar
│   ├── Machine Configuration (Type selector)
│   ├── Sensor Sliders (5 inputs)
│   ├── RUN ANALYSIS button
│   └── Load Scenario (preset test cases)
│
└── Main Area
    ├── Tab 1: PREDICTION
    │   ├── Result Banner (Normal / Failure)
    │   ├── Risk Gauge
    │   ├── Computed Features panel
    │   ├── Sensor Range Chart
    │   ├── Feature Importance Chart
    │   ├── Failure Flag Cards (conditional)
    │   └── Prediction History
    │
    ├── Tab 2: MODEL PERFORMANCE
    │   ├── Key metric cards
    │   ├── Model comparison table
    │   └── Business impact table
    │
    └── Tab 3: ABOUT
        ├── Project description
        └── Author information
```

---

## Section 1 — Sidebar: Sensor Input Controls

![Sidebar](../assets/screenshots/sidebar.png)

The sidebar is the operator's control panel. It contains all machine configuration inputs and is always visible regardless of which tab is active.

### Machine Type Selector

```python
machine_type = st.selectbox(
    "Machine Type",
    options=['L', 'M', 'H'],
    format_func=lambda x: {
        'L': 'L — Low Quality',
        'M': 'M — Medium Quality',
        'H': 'H — High Quality'
    }[x]
)
```

Machine type determines the failure thresholds — Type L machines have the tightest Overstrain limit (11,000 Nm·min vs 13,000 for Type H).

### Sensor Sliders

```python
air_temp     = st.slider("Air Temperature [K]",    295.0, 304.0,  298.1, 0.1)
process_temp = st.slider("Process Temperature [K]", 305.0, 313.8,  308.6, 0.1)
rpm          = st.slider("Rotational Speed [RPM]",  1168,  2886,   1500,  1)
torque       = st.slider("Torque [Nm]",             3.8,   76.6,   40.0,  0.1)
tool_wear    = st.slider("Tool Wear [min]",         0,     253,    80,    1)
```

All slider ranges are set to the exact min/max values from the AI4I 2020 dataset — so the user can never enter an out-of-distribution value.

### Preset Scenario Loader

```python
scenarios = {
    "Healthy Machine"  : (298.1, 308.6, 1551, 42.8,  0,   'M'),
    "Overstrain Risk"  : (298.2, 308.7, 1400, 52.0,  230, 'L'),
    "Heat Dissipation" : (300.0, 307.0, 1300, 38.0,  80,  'M'),
    "Power Failure"    : (298.5, 309.0, 800,  20.0,  50,  'H'),
    "Borderline Case"  : (299.0, 309.5, 1420, 46.0,  180, 'L'),
}
```

Five preset scenarios are available for immediate testing. Each represents a distinct failure condition — useful for demonstrating the app to an audience.

### Model Info Footer

The sidebar footer permanently displays the active model's key metrics:
```
MODEL: XGBoost
F1: 0.703  |  AUC: 0.978
DATASET: UCI AI4I 2020
```

---

## Section 2 — Prediction Result Banner

![Result Banner — Normal](../assets/screenshots/banner_normal.png)
*Normal operating condition — green border, 5.2% probability*

![Result Banner — Failure](../assets/screenshots/banner_failure.png)
*Failure predicted — red border, 87.4% probability*

The banner is the most prominent element after pressing RUN ANALYSIS. It communicates the result immediately and unambiguously — no scrolling required.

```python
if pred == 1:
    banner_bg     = '#1a0a0a'
    banner_border = '#dc4f4f'
    status_color  = '#dc4f4f'
    status_text   = 'FAILURE PREDICTED'
else:
    banner_bg     = '#0a1a12'
    banner_border = '#3dbc74'
    status_color  = '#3dbc74'
    status_text   = 'NORMAL OPERATION'
```

The banner shows two values simultaneously:
- The binary prediction (NORMAL / FAILURE) in large text
- The exact failure probability percentage

---

## Section 3 — Risk Gauge

![Risk Gauge](../assets/screenshots/gauge.png)

The circular gauge gives an immediate visual sense of how close to failure the machine is — even if the binary prediction says "normal", a 45% probability reading tells a different story than 2%.

```python
def make_gauge(prob):
    color = '#3dbc74' if prob < 0.3 else ('#e8a23a' if prob < 0.6 else '#dc4f4f')

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color, 'thickness': 0.22},
            'steps': [
                {'range': [0,  30],  'color': '#0d1a14'},  # safe zone
                {'range': [30, 60],  'color': '#1a1608'},  # caution zone
                {'range': [60, 100], 'color': '#1a0a0a'},  # danger zone
            ]
        }
    ))
```

**Colour thresholds:**

| Range | Colour | Meaning |
|---|---|---|
| 0 – 30% | Green `#3dbc74` | Safe operating range |
| 30 – 60% | Amber `#e8a23a` | Caution — monitor closely |
| 60 – 100% | Red `#dc4f4f` | High risk — maintenance recommended |

---

## Section 4 — Sensor Range Chart

![Sensor Range Chart](../assets/screenshots/sensor_ranges.png)

This chart answers the question: *which of the five sensors is in the danger zone?*

Each bar shows where the current reading sits as a percentile within the full dataset range (0 = minimum ever recorded, 100 = maximum ever recorded). The vertical dotted lines at 20% and 80% mark the normal operating corridor.

```python
def make_sensor_ranges(air_temp, process_temp, rpm, torque, tool_wear):
    mins = [295,  305,   1168, 3.8,  0]
    maxs = [304,  313.8, 2886, 76.6, 253]

    normalised = [(v - mn) / (mx - mn) * 100
                  for v, mn, mx in zip(values, mins, maxs)]

    # Color: green if 20-80%, amber if 10-90%, red if outside
    colors = [
        '#3dbc74' if 20 <= n <= 80 else
        '#e8a23a' if 10 <= n <= 90 else
        '#dc4f4f'
        for n in normalised
    ]
```

A sensor bar appearing red immediately tells the operator which reading is causing the concern — before they even read the failure flag cards.

---

## Section 5 — Computed Features Panel

![Computed Features](../assets/screenshots/computed_features.png)

The three engineered features — Temp_diff, Power, and Wear×Torque — are computed live from the slider inputs and displayed here. This makes the feature engineering transparent and explainable.

```python
temp_diff   = process_temp - air_temp
power       = torque * (rpm * 2 * np.pi / 60)
wear_torque = tool_wear * torque
```

An operator can directly compare these computed values against the known failure thresholds:
- Temp_diff < 8.6K at RPM < 1380 → HDF risk
- Power outside 3,500–9,000W → PWF risk
- Wear×Torque > 11,000 Nm·min (Type L) → OSF risk

---

## Section 6 — Feature Importance Chart

![Feature Importance](../assets/screenshots/feature_importance.png)

Horizontal bar chart showing the XGBoost feature importance scores. Engineered features are highlighted in amber, raw sensor features in dark blue — making it visually clear that the domain-informed features carry the most predictive weight.

```python
feature_importance = {
    'Wear_Torque'           : 0.31,
    'Torque__Nm'            : 0.18,
    'Rotational_speed__rpm' : 0.15,
    'Power'                 : 0.12,
    'Tool_wear__min'        : 0.10,
    'Temp_diff'             : 0.07,
    ...
}

colors = [
    '#e8a23a' if f in ['Wear_Torque', 'Power', 'Temp_diff']
    else '#2a3a5c'
    for f in feats
]
```

---

## Section 7 — Failure Flag Cards

![Failure Flags](../assets/screenshots/failure_flags.png)

When a physical failure threshold is crossed, a dedicated card appears for each triggered failure mode. These only appear when relevant — a healthy machine shows no cards.

```python
flags = []
if wear_torque > 11000:
    flags.append(('OSF', 'Overstrain Failure',
                  f'Wear×Torque = {wear_torque:,.0f} > 11,000 threshold',
                  '#dc4f4f'))
if temp_diff < 8.6 and rpm < 1380:
    flags.append(('HDF', 'Heat Dissipation Failure',
                  f'Temp diff = {temp_diff:.1f}K at {rpm} RPM',
                  '#dc4f4f'))
if power < 3500 or power > 9000:
    flags.append(('PWF', 'Power Failure',
                  f'Power = {power:,.0f}W (safe: 3,500–9,000W)',
                  '#dc4f4f'))
if tool_wear >= 200:
    flags.append(('TWF', 'Tool Wear Failure',
                  f'Tool wear = {tool_wear} min (threshold: 200–240 min)',
                  '#dc4f4f'))
```

Each card shows: the failure mode code, the full name, and the exact numerical reason why the threshold was exceeded. This makes the prediction explainable — the engineer knows what to fix, not just that something is wrong.

---

## Section 8 — Prediction History

All predictions made during the current session are stored and displayed in a table. Failure rows are highlighted in red, normal rows in green — allowing the operator to quickly see patterns across multiple sensor readings.

```python
if 'history' not in st.session_state:
    st.session_state.history = []

# After each prediction:
st.session_state.history.append({
    'Air Temp': air_temp, 'Process Temp': process_temp,
    'RPM': rpm, 'Torque': torque, 'Tool Wear': tool_wear,
    'Type': machine_type,
    'Probability': round(prob * 100, 1),
    'Prediction': 'FAILURE' if pred == 1 else 'NORMAL'
})
```

---

## Section 9 — Model Performance Tab

![Model Performance Tab](../assets/screenshots/model_performance_tab.png)

The second tab shows four key metric cards at the top, followed by the full model comparison table and business impact analysis — accessible without leaving the app.

```python
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("F1 Score",  "0.703")
with c2: st.metric("AUC-ROC",   "0.978")
with c3: st.metric("Precision", "0.650")
with c4: st.metric("Recall",    "0.765")
```

---

## Predicted Scenarios — Visual Reference

### Scenario 1: Healthy Machine

| Input | Value |
|---|---|
| Machine Type | M |
| Air Temp | 298.1 K |
| Process Temp | 308.6 K |
| RPM | 1,551 |
| Torque | 42.8 Nm |
| Tool Wear | 0 min |

**Result:** Normal — ~2–4% probability. No failure flags. All sensor bars green.

![Scenario 1](../assets/screenshots/scenario_healthy.png)

---

### Scenario 2: Overstrain Failure (OSF)

| Input | Value |
|---|---|
| Machine Type | L |
| RPM | 1,400 |
| Torque | 52.0 Nm |
| Tool Wear | 230 min |

**Wear × Torque = 230 × 52 = 11,960 Nm·min** → exceeds Type L threshold of 11,000

**Result:** Failure predicted. OSF flag card appears. Tool wear and torque bars in red zone.

![Scenario 2](../assets/screenshots/scenario_osf.png)

---

### Scenario 3: Heat Dissipation Failure (HDF)

| Input | Value |
|---|---|
| Air Temp | 300.0 K |
| Process Temp | 307.0 K |
| RPM | 1,300 |

**Temp diff = 7.0K** (below 8.6K threshold) **at 1,300 RPM** (below 1,380 threshold)

**Result:** Failure predicted. HDF flag card appears. RPM bar in red zone.

![Scenario 3](../assets/screenshots/scenario_hdf.png)

---


### Scenario 4: Borderline Case

| Input | Value |
|---|---|
| Machine Type | L |
| RPM | 1,420 |
| Torque | 46.0 Nm |
| Tool Wear | 180 min |

**Wear × Torque = 8,280 Nm·min** — below threshold. No physical flags fire.

**Result:** The model may return 20–40% probability without triggering a binary failure prediction — demonstrating that the probability score carries information beyond the yes/no output.

![Scenario 4](../assets/screenshots/scenario_borderline.png)

---

## Generating Screenshots

To add screenshots to this markdown file, run the app, load each scenario, and save the relevant screenshots to `assets/screenshots/`. The filenames used in this document are:

```
assets/screenshots/
├── sidebar.png
├── banner_normal.png
├── banner_failure.png
├── gauge.png
├── sensor_ranges.png
├── computed_features.png
├── feature_importance.png
├── failure_flags.png
├── prediction_history.png
├── model_performance_tab.png
├── scenario_healthy.png
├── scenario_osf.png
├── scenario_hdf.png
├── scenario_pwf.png
└── scenario_borderline.png
```

On Windows, use `Win + Shift + S` to take region screenshots. Save directly into the `assets/screenshots/` folder before pushing to GitHub.

---

## Custom Styling — Key CSS Decisions

The app overrides Streamlit's default styling using injected CSS. Key decisions:

**Typography override** — replaces Streamlit's default sans-serif with IBM Plex Mono for all labels:
```css
section[data-testid="stSidebar"] .stSlider label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
```

**Button styling** — full-width amber button with uppercase mono text:
```css
.stButton > button {
    background-color: #e8a23a !important;
    color: #0d0f12 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
```

**Tab styling** — amber underline on active tab, monospace uppercase labels:
```css
.stTabs [aria-selected="true"] {
    color: #e8a23a !important;
    border-bottom: 2px solid #e8a23a !important;
}
```

**Metric cards** — dark surface with border, monospace label:
```css
[data-testid="metric-container"] {
    background: #111418;
    border: 1px solid #1e2329;
    border-radius: 4px;
}
```

---

[← Back to README](../README.md)
