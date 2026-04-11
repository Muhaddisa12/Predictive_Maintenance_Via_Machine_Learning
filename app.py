import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictMaint — Industrial AI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f12;
    color: #c8cdd6;
}

.stApp {
    background-color: #0d0f12;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #111418;
    border-right: 1px solid #1e2329;
}

section[data-testid="stSidebar"] * {
    color: #c8cdd6 !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5a6478 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding: 4px 0;
}

.stSlider [data-testid="stThumbValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    background: #1a2035 !important;
    border: 1px solid #2a3550 !important;
    color: #e8a23a !important;
    border-radius: 3px !important;
}

/* ── Select box ── */
.stSelectbox [data-baseweb="select"] {
    background-color: #161b24 !important;
    border: 1px solid #1e2a3e !important;
    border-radius: 4px !important;
}

.stSelectbox [data-baseweb="select"] * {
    background-color: #161b24 !important;
    color: #c8cdd6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Number input ── */
.stNumberInput input {
    background-color: #161b24 !important;
    border: 1px solid #1e2a3e !important;
    color: #c8cdd6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 4px !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background-color: #e8a23a !important;
    color: #0d0f12 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #f0b84e !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #111418;
    border: 1px solid #1e2329;
    border-radius: 4px;
    padding: 16px !important;
}

[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5a6478 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #c8cdd6 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111418;
    border-bottom: 1px solid #1e2329;
    gap: 0px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5a6478 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 20px !important;
}

.stTabs [aria-selected="true"] {
    color: #e8a23a !important;
    border-bottom: 2px solid #e8a23a !important;
}

/* ── Divider ── */
hr {
    border-color: #1e2329 !important;
    margin: 8px 0 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5a6478 !important;
    background-color: #111418 !important;
    border: 1px solid #1e2329 !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f12; }
::-webkit-scrollbar-thumb { background: #1e2a3e; border-radius: 2px; }

/* ── Hide default streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper: Custom card HTML ───────────────────────────────────
def card(content_html, border_color="#1e2329", bg="#111418"):
    return f"""
    <div style="
        background:{bg};
        border:1px solid {border_color};
        border-radius:4px;
        padding:20px 24px;
        margin-bottom:12px;
    ">{content_html}</div>
    """

def mono(text, color="#e8a23a", size="13px"):
    return f'<span style="font-family:IBM Plex Mono,monospace;color:{color};font-size:{size};font-weight:600;">{text}</span>'

def label(text):
    return f'<p style="font-family:IBM Plex Mono,monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#5a6478;margin:0 0 4px 0;">{text}</p>'


# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load('best_model_xgboost.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_cols.json') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols, True
    except Exception as e:
        return None, None, None, False

model, scaler, feature_cols, model_loaded = load_model()


# ── Prediction engine ──────────────────────────────────────────
def run_prediction(air_temp, process_temp, rpm, torque, tool_wear, machine_type):
    type_map = {'H': 0, 'L': 1, 'M': 2}
    type_encoded = type_map[machine_type]

    temp_diff   = process_temp - air_temp
    power       = torque * (rpm * 2 * np.pi / 60)
    wear_torque = tool_wear * torque

    input_df = pd.DataFrame([{
        'Type_encoded'           : type_encoded,
        'Air_temperature__K'     : air_temp,
        'Process_temperature__K' : process_temp,
        'Rotational_speed__rpm'  : rpm,
        'Torque__Nm'             : torque,
        'Tool_wear__min'         : tool_wear,
        'Temp_diff'              : temp_diff,
        'Power'                  : power,
        'Wear_Torque'            : wear_torque
    }])

    if feature_cols:
        input_df = input_df[feature_cols]

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    # Failure mode flags
    flags = []
    if wear_torque > 11000:
        flags.append(('OSF', 'Overstrain Failure', f'Wear×Torque = {wear_torque:,.0f} > 11,000 threshold', '#dc4f4f'))
    if temp_diff < 8.6 and rpm < 1380:
        flags.append(('HDF', 'Heat Dissipation Failure', f'Temp diff = {temp_diff:.1f}K at {rpm} RPM', '#dc4f4f'))
    if power < 3500 or power > 9000:
        flags.append(('PWF', 'Power Failure', f'Power = {power:,.0f}W (safe: 3,500–9,000W)', '#dc4f4f'))
    if tool_wear >= 200:
        flags.append(('TWF', 'Tool Wear Failure', f'Tool wear = {tool_wear} min (threshold: 200–240 min)', '#dc4f4f'))

    engineered = {
        'Temp Difference': f'{temp_diff:.2f} K',
        'Power'          : f'{power:,.0f} W',
        'Wear × Torque'  : f'{wear_torque:,.0f} Nm·min',
    }

    return prob, int(pred), flags, engineered, input_df


# ── Gauge chart ────────────────────────────────────────────────
def make_gauge(prob):
    color = '#3dbc74' if prob < 0.3 else ('#e8a23a' if prob < 0.6 else '#dc4f4f')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={
            'suffix': '%',
            'font': {'family': 'IBM Plex Mono', 'size': 36, 'color': color}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#2a3040',
                'tickfont': {'family': 'IBM Plex Mono', 'size': 10, 'color': '#5a6478'},
            },
            'bar': {'color': color, 'thickness': 0.22},
            'bgcolor': '#111418',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  30],  'color': '#0d1a14'},
                {'range': [30, 60],  'color': '#1a1608'},
                {'range': [60, 100], 'color': '#1a0a0a'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='#111418',
        plot_bgcolor='#111418',
        margin=dict(t=20, b=10, l=20, r=20),
        height=220,
        font={'family': 'IBM Plex Mono'}
    )
    return fig


# ── Feature bar chart ──────────────────────────────────────────
def make_feature_bar(input_df):
    feature_importance = {
        'Wear_Torque'           : 0.31,
        'Torque__Nm'            : 0.18,
        'Rotational_speed__rpm' : 0.15,
        'Power'                 : 0.12,
        'Tool_wear__min'        : 0.10,
        'Temp_diff'             : 0.07,
        'Type_encoded'          : 0.04,
        'Process_temperature__K': 0.02,
        'Air_temperature__K'    : 0.01,
    }

    labels_map = {
        'Wear_Torque'           : 'Wear × Torque',
        'Torque__Nm'            : 'Torque',
        'Rotational_speed__rpm' : 'Rotational Speed',
        'Power'                 : 'Power',
        'Tool_wear__min'        : 'Tool Wear',
        'Temp_diff'             : 'Temp Difference',
        'Type_encoded'          : 'Machine Type',
        'Process_temperature__K': 'Process Temp',
        'Air_temperature__K'    : 'Air Temp',
    }

    feats  = list(feature_importance.keys())
    vals   = [feature_importance[f] for f in feats]
    labels = [labels_map[f] for f in feats]
    colors = ['#e8a23a' if f in ['Wear_Torque','Power','Temp_diff'] else '#2a3a5c' for f in feats]

    fig = go.Figure(go.Bar(
        x=vals, y=labels,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='#111418',
        plot_bgcolor='#111418',
        margin=dict(t=10, b=10, l=10, r=20),
        height=300,
        xaxis=dict(
            showgrid=True, gridcolor='#1e2329',
            tickfont=dict(family='IBM Plex Mono', size=10, color='#5a6478'),
            zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(family='IBM Plex Mono', size=11, color='#c8cdd6'),
            autorange='reversed'
        ),
        hoverlabel=dict(font_family='IBM Plex Mono'),
    )
    return fig


# ── Sensor range chart ─────────────────────────────────────────
def make_sensor_ranges(air_temp, process_temp, rpm, torque, tool_wear):
    sensors = ['Air Temp (K)', 'Process Temp (K)', 'RPM', 'Torque (Nm)', 'Tool Wear (min)']
    values  = [air_temp,        process_temp,        rpm,   torque,        tool_wear]
    mins    = [295,             305,                 1168,  3.8,           0]
    maxs    = [304,             313.8,               2886,  76.6,          253]

    normalised = [(v - mn) / (mx - mn) * 100 for v, mn, mx in zip(values, mins, maxs)]
    colors = ['#3dbc74' if 20 <= n <= 80 else '#e8a23a' if 10 <= n <= 90 else '#dc4f4f' for n in normalised]

    fig = go.Figure()
    for i, (s, n, c, v) in enumerate(zip(sensors, normalised, colors, values)):
        fig.add_trace(go.Bar(
            name=s, x=[n], y=[s],
            orientation='h',
            marker_color=c,
            marker_line_width=0,
            hovertemplate=f'{s}: {v}<extra></extra>',
            showlegend=False
        ))
        fig.add_shape(type='line', x0=20, x1=20, y0=i-0.4, y1=i+0.4,
                      line=dict(color='#2a3a5c', width=1, dash='dot'))
        fig.add_shape(type='line', x0=80, x1=80, y0=i-0.4, y1=i+0.4,
                      line=dict(color='#2a3a5c', width=1, dash='dot'))

    fig.update_layout(
        paper_bgcolor='#111418',
        plot_bgcolor='#111418',
        margin=dict(t=10, b=10, l=10, r=20),
        height=220,
        barmode='overlay',
        xaxis=dict(
            range=[0, 100],
            title=dict(text='Percentile within dataset range', font=dict(family='IBM Plex Mono', size=10, color='#5a6478')),
            tickfont=dict(family='IBM Plex Mono', size=10, color='#5a6478'),
            showgrid=True, gridcolor='#1a1f28', zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(family='IBM Plex Mono', size=11, color='#c8cdd6'),
        ),
    )
    return fig


# ── History tracker ────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'trigger_run' not in st.session_state:
    st.session_state['trigger_run'] = False


# ══════════════════════════════════════════════════════════════
#   SIDEBAR — INPUT CONTROLS
# ══════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("""
    <div style="padding:24px 0 16px 0;">
        <p style="font-family:IBM Plex Mono,monospace;font-size:10px;
                  letter-spacing:0.2em;text-transform:uppercase;
                  color:#5a6478;margin:0 0 4px 0;">System</p>
        <p style="font-family:IBM Plex Mono,monospace;font-size:18px;
                  font-weight:600;color:#e8a23a;margin:0;letter-spacing:0.04em;">
                  PredictMaint</p>
        <p style="font-family:IBM Plex Sans,sans-serif;font-size:12px;
                  color:#3d4860;margin:4px 0 0 0;">Industrial Sensor Analysis v1.0</p>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-family:IBM Plex Mono,monospace;font-size:10px;
              letter-spacing:0.15em;text-transform:uppercase;
              color:#5a6478;margin:16px 0 12px 0;">Machine Configuration</p>
    """, unsafe_allow_html=True)

    machine_type = st.selectbox(
        "Machine Type",
        options=['L', 'M', 'H'],
        format_func=lambda x: {'L': 'L — Low Quality', 'M': 'M — Medium Quality', 'H': 'H — High Quality'}[x],
        index=1
    )

    st.markdown("""
    <p style="font-family:IBM Plex Mono,monospace;font-size:10px;
              letter-spacing:0.15em;text-transform:uppercase;
              color:#5a6478;margin:20px 0 12px 0;">Sensor Readings</p>
    """, unsafe_allow_html=True)

    air_temp = st.slider(
        "Air Temperature [K]",
        min_value=295.0, max_value=304.0,
        value=298.1, step=0.1,
        format="%.1f"
    )

    process_temp = st.slider(
        "Process Temperature [K]",
        min_value=305.0, max_value=313.8,
        value=308.6, step=0.1,
        format="%.1f"
    )

    rpm = st.slider(
        "Rotational Speed [RPM]",
        min_value=1168, max_value=2886,
        value=1500, step=1
    )

    torque = st.slider(
        "Torque [Nm]",
        min_value=3.8, max_value=76.6,
        value=40.0, step=0.1,
        format="%.1f"
    )

    tool_wear = st.slider(
        "Tool Wear [min]",
        min_value=0, max_value=253,
        value=80, step=1
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    run_btn = st.button("RUN ANALYSIS", type="primary", key="sidebar_run_btn")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Quick scenario loader
    with st.expander("LOAD SCENARIO"):
        scenarios = {
            "Healthy Machine"    : (298.1, 308.6, 1551, 42.8,  0,   'M'),
            "Overstrain Risk"    : (298.2, 308.7, 1400, 52.0,  230, 'L'),
            "Heat Dissipation"   : (300.0, 307.0, 1300, 38.0,  80,  'M'),
            "Power Failure"      : (298.5, 309.0, 800,  20.0,  50,  'H'),
            "Borderline Case"    : (299.0, 309.5, 1420, 46.0,  180, 'L'),
        }
        scenario_name = st.selectbox("Preset", list(scenarios.keys()), label_visibility="collapsed")
        if st.button("LOAD"):
            s = scenarios[scenario_name]
            st.session_state['air_temp']     = s[0]
            st.session_state['process_temp'] = s[1]
            st.session_state['rpm']          = s[2]
            st.session_state['torque']       = s[3]
            st.session_state['tool_wear']    = s[4]
            st.rerun()

    st.markdown("""
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid #1e2329;">
        <p style="font-family:IBM Plex Mono,monospace;font-size:9px;
                  color:#2a3040;letter-spacing:0.08em;line-height:1.8;">
                  MODEL: XGBoost<br/>
                  F1: 0.703 &nbsp;|&nbsp; AUC: 0.978<br/>
                  DATASET: UCI AI4I 2020
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#   MAIN AREA
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="padding:32px 0 8px 0;display:flex;align-items:flex-end;gap:16px;">
    <div>
        <p style="font-family:IBM Plex Mono,monospace;font-size:10px;
                  letter-spacing:0.2em;text-transform:uppercase;
                  color:#5a6478;margin:0 0 4px 0;">Industrial AI Platform</p>
        <h1 style="font-family:IBM Plex Mono,monospace;font-size:28px;
                   font-weight:600;color:#c8cdd6;margin:0;letter-spacing:-0.01em;">
            Predictive Maintenance
            <span style="color:#e8a23a;">System</span>
        </h1>
    </div>
</div>
<p style="font-family:IBM Plex Sans,sans-serif;font-size:14px;color:#3d4860;
          margin:8px 0 28px 0;max-width:600px;">
    Real-time machine failure prediction from industrial sensor data.
    Trained on 10,000 readings using XGBoost with domain-informed feature engineering.
</p>
<hr/>
""", unsafe_allow_html=True)


# ── Initial state (no prediction yet) ─────────────────────────
if not run_btn and not st.session_state.history:
    tab1, tab2, tab3 = st.tabs(["PREDICTION", "MODEL PERFORMANCE", "ABOUT"])

    with tab1:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown(card("""
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                      Status</p>
            <p style='font-family:IBM Plex Mono,monospace;font-size:15px;
                      color:#2a3040;margin:0;'>
                      Configure sensor readings in the sidebar<br/>
                      and press RUN ANALYSIS to begin.</p>
            """, border_color="#1e2329"), unsafe_allow_html=True)
            run_btn = st.button("RUN ANALYSIS", type="primary", key="main_run_btn")
            if run_btn:
                st.session_state['trigger_run'] = True
                st.rerun()

        with col2:
            st.markdown(card("""
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 16px 0;'>
                      Quick Reference — Failure Thresholds</p>
            <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:11px;'>
                <tr>
                    <td style='color:#5a6478;padding:5px 0;'>OSF</td>
                    <td style='color:#c8cdd6;'>Wear × Torque &gt; 11,000 Nm·min</td>
                </tr>
                <tr>
                    <td style='color:#5a6478;padding:5px 0;'>HDF</td>
                    <td style='color:#c8cdd6;'>Temp diff &lt; 8.6K at RPM &lt; 1,380</td>
                </tr>
                <tr>
                    <td style='color:#5a6478;padding:5px 0;'>PWF</td>
                    <td style='color:#c8cdd6;'>Power outside 3,500–9,000W</td>
                </tr>
                <tr>
                    <td style='color:#5a6478;padding:5px 0;'>TWF</td>
                    <td style='color:#c8cdd6;'>Tool wear ≥ 200 min</td>
                </tr>
                <tr>
                    <td style='color:#5a6478;padding:5px 0;'>RNF</td>
                    <td style='color:#c8cdd6;'>Random — 0.1% probability</td>
                </tr>
            </table>
            """, border_color="#1e2329"), unsafe_allow_html=True)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("F1 Score",   "0.703", "Best model")
        with c2: st.metric("AUC-ROC",    "0.978", "+0.004 vs baseline")
        with c3: st.metric("Precision",  "0.650", "65% alert accuracy")
        with c4: st.metric("Recall",     "0.765", "76.5% failures caught")

        st.markdown("<br/>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(card("""
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                      Model Comparison</p>
            <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:11px;'>
                <tr style='border-bottom:1px solid #1e2329;'>
                    <td style='color:#5a6478;padding:6px 8px 6px 0;'>Model</td>
                    <td style='color:#5a6478;padding:6px 4px;'>Recall</td>
                    <td style='color:#5a6478;padding:6px 4px;'>Prec.</td>
                    <td style='color:#5a6478;padding:6px 0 6px 4px;'>F1</td>
                </tr>
                <tr>
                    <td style='color:#3d4860;padding:6px 8px 6px 0;'>Balanced RF</td>
                    <td style='color:#3dbc74;padding:6px 4px;'>0.941</td>
                    <td style='color:#dc4f4f;padding:6px 4px;'>0.234</td>
                    <td style='color:#3d4860;padding:6px 0 6px 4px;'>0.375</td>
                </tr>
                <tr style='background:#161b24;'>
                    <td style='color:#e8a23a;padding:6px 8px;'>XGBoost ✓</td>
                    <td style='color:#c8cdd6;padding:6px 4px;'>0.765</td>
                    <td style='color:#c8cdd6;padding:6px 4px;'>0.650</td>
                    <td style='color:#e8a23a;padding:6px 0 6px 4px;'>0.703</td>
                </tr>
                <tr>
                    <td style='color:#3d4860;padding:6px 8px 6px 0;'>RF+Threshold</td>
                    <td style='color:#3dbc74;padding:6px 4px;'>1.000</td>
                    <td style='color:#dc4f4f;padding:6px 4px;'>0.034</td>
                    <td style='color:#3d4860;padding:6px 0 6px 4px;'>0.066</td>
                </tr>
                <tr>
                    <td style='color:#3d4860;padding:6px 8px 6px 0;'>Ensemble</td>
                    <td style='color:#3dbc74;padding:6px 4px;'>0.794</td>
                    <td style='color:#c8cdd6;padding:6px 4px;'>0.587</td>
                    <td style='color:#3d4860;padding:6px 0 6px 4px;'>0.675</td>
                </tr>
            </table>
            """, border_color="#1e2329"), unsafe_allow_html=True)

        with col_b:
            st.markdown(card("""
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                      Business Impact — 10,000 Machines/Year</p>
            <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:11px;'>
                <tr style='border-bottom:1px solid #1e2329;'>
                    <td style='color:#5a6478;padding:6px 8px 6px 0;'>Model</td>
                    <td style='color:#5a6478;padding:6px 4px;'>Caught</td>
                    <td style='color:#5a6478;padding:6px 4px;'>FP/yr</td>
                    <td style='color:#5a6478;padding:6px 0 6px 4px;'>Cost</td>
                </tr>
                <tr style='background:#161b24;'>
                    <td style='color:#e8a23a;padding:6px 8px;'>XGBoost ✓</td>
                    <td style='color:#c8cdd6;padding:6px 4px;'>259/339</td>
                    <td style='color:#c8cdd6;padding:6px 4px;'>140</td>
                    <td style='color:#c8cdd6;padding:6px 0 6px 4px;'>$8.1M</td>
                </tr>
                <tr>
                    <td style='color:#3d4860;padding:6px 8px 6px 0;'>Ensemble</td>
                    <td style='color:#c8cdd6;padding:6px 4px;'>269/339</td>
                    <td style='color:#3d4860;padding:6px 4px;'>189</td>
                    <td style='color:#3d4860;padding:6px 0 6px 4px;'>$7.2M</td>
                </tr>
                <tr>
                    <td style='color:#3d4860;padding:6px 8px 6px 0;'>Balanced RF</td>
                    <td style='color:#3dbc74;padding:6px 4px;'>319/339</td>
                    <td style='color:#dc4f4f;padding:6px 4px;'>1,050</td>
                    <td style='color:#3d4860;padding:6px 0 6px 4px;'>$3.1M</td>
                </tr>
                <tr style='border-top:1px solid #1e2329;'>
                    <td colspan='4' style='color:#3d4860;padding:8px 0 0 0;font-size:10px;'>
                    No system baseline: $33.9M/year<br/>
                    XGBoost saves ~$25.8M vs no system</td>
                </tr>
            </table>
            """, border_color="#1e2329"), unsafe_allow_html=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(card("""
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                      Project</p>
            <p style='font-family:IBM Plex Sans,sans-serif;font-size:13px;color:#8a95a8;
                      line-height:1.7;margin:0 0 12px 0;'>
                Predictive maintenance system built on the UCI AI4I 2020 dataset.
                The model uses 5 raw sensor features and 3 domain-engineered features
                to predict machine failure before it occurs.
            </p>
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;
                      margin:16px 0 8px 0;'>Engineered Features</p>
            <table style='font-family:IBM Plex Mono,monospace;font-size:11px;width:100%;'>
                <tr><td style='color:#e8a23a;padding:4px 12px 4px 0;'>Temp_diff</td>
                    <td style='color:#8a95a8;'>Process − Air temp → HDF risk</td></tr>
                <tr><td style='color:#e8a23a;padding:4px 12px 4px 0;'>Power</td>
                    <td style='color:#8a95a8;'>Torque × ω → PWF boundary</td></tr>
                <tr><td style='color:#e8a23a;padding:4px 12px 4px 0;'>Wear_Torque</td>
                    <td style='color:#8a95a8;'>Wear × Torque → OSF risk</td></tr>
            </table>
            """, border_color="#1e2329"), unsafe_allow_html=True)

        with col_b:
            st.markdown(card("""
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                      Built By</p>
            <p style='font-family:IBM Plex Mono,monospace;font-size:16px;
                      color:#c8cdd6;font-weight:600;margin:0 0 4px 0;'>Muhaddisa</p>
            <p style='font-family:IBM Plex Sans,sans-serif;font-size:13px;
                      color:#5a6478;margin:0 0 16px 0;'>
                BS Artificial Intelligence — 6th Semester<br/>
                Shifa Tameer-e-Millat University, Pakistan
            </p>
            <hr/>
            <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                      letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;
                      margin:12px 0 8px 0;'>Stack</p>
            <p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#3d4860;
                      line-height:2;'>
                Python &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Scikit-learn<br/>
                Streamlit &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Plotly<br/>
                imbalanced-learn &nbsp;·&nbsp; Joblib
            </p>
            """, border_color="#1e2329"), unsafe_allow_html=True)


# ── Run prediction ─────────────────────────────────────────────
if run_btn or st.session_state.get('trigger_run', False):
    st.session_state['trigger_run'] = False
    if not model_loaded:
        st.error("Model file not found. Run the notebook first to generate best_model_xgboost.pkl")
    else:
        with st.spinner("Running analysis..."):
            prob, pred, flags, engineered, input_df = run_prediction(
                air_temp, process_temp, rpm, torque, tool_wear, machine_type
            )

            # Save to history
            st.session_state.history.append({
                'Air Temp': air_temp, 'Process Temp': process_temp,
                'RPM': rpm, 'Torque': torque, 'Tool Wear': tool_wear,
                'Type': machine_type, 'Probability': round(prob*100, 1),
                'Prediction': 'FAILURE' if pred == 1 else 'NORMAL'
            })

        tab1, tab2, tab3 = st.tabs(["PREDICTION", "MODEL PERFORMANCE", "ABOUT"])

        with tab1:

            # ── Result banner ──
            if pred == 1:
                banner_bg    = '#1a0a0a'
                banner_border = '#dc4f4f'
                status_color = '#dc4f4f'
                status_text  = 'FAILURE PREDICTED'
                status_icon  = '⚠'
            else:
                banner_bg    = '#0a1a12'
                banner_border = '#3dbc74'
                status_color = '#3dbc74'
                status_text  = 'NORMAL OPERATION'
                status_icon  = '✓'

            st.markdown(card(f"""
            <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>
                <div>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                              letter-spacing:0.2em;text-transform:uppercase;
                              color:#5a6478;margin:0 0 6px 0;'>Prediction Result</p>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:26px;
                              font-weight:600;color:{status_color};margin:0;letter-spacing:0.02em;'>
                        {status_icon}&nbsp; {status_text}
                    </p>
                </div>
                <div style='text-align:right;'>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                              letter-spacing:0.12em;text-transform:uppercase;
                              color:#5a6478;margin:0 0 4px 0;'>Failure Probability</p>
                    <p style='font-family:IBM Plex Mono,monospace;font-size:36px;
                              font-weight:600;color:{status_color};margin:0;'>
                        {prob*100:.1f}%
                    </p>
                </div>
            </div>
            """, border_color=banner_border, bg=banner_bg), unsafe_allow_html=True)

            # ── Main content grid ──
            col1, col2 = st.columns([1, 1.3])

            with col1:
                # Gauge
                st.markdown(label("Risk Gauge"), unsafe_allow_html=True)
                st.plotly_chart(make_gauge(prob), use_container_width=True, config={'displayModeBar': False})

                # Engineered values
                st.markdown(card(f"""
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;
                          color:#5a6478;margin:0 0 12px 0;'>Computed Features</p>
                <table style='width:100%;border-collapse:collapse;'>
                    <tr>
                        <td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#5a6478;padding:5px 0;'>Temp Difference</td>
                        <td style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#e8a23a;text-align:right;'>
                            {engineered["Temp Difference"]}</td>
                    </tr>
                    <tr>
                        <td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#5a6478;padding:5px 0;'>Power</td>
                        <td style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#e8a23a;text-align:right;'>
                            {engineered["Power"]}</td>
                    </tr>
                    <tr>
                        <td style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#5a6478;padding:5px 0;'>Wear × Torque</td>
                        <td style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#e8a23a;text-align:right;'>
                            {engineered["Wear × Torque"]}</td>
                    </tr>
                </table>
                """), unsafe_allow_html=True)

            with col2:
                # Sensor range chart
                st.markdown(label("Sensor Readings — Percentile Range"), unsafe_allow_html=True)
                st.plotly_chart(
                    make_sensor_ranges(air_temp, process_temp, rpm, torque, tool_wear),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )

                # Feature importance
                st.markdown(label("Feature Importance (XGBoost)"), unsafe_allow_html=True)
                st.plotly_chart(
                    make_feature_bar(input_df),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )

            # ── Failure flags ──
            if flags:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("""
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;
                          color:#5a6478;margin:0 0 8px 0;'>Active Failure Indicators</p>
                """, unsafe_allow_html=True)

                cols = st.columns(len(flags))
                for col, (code, name, reason, color) in zip(cols, flags):
                    with col:
                        st.markdown(card(f"""
                        <p style='font-family:IBM Plex Mono,monospace;font-size:13px;
                                  font-weight:600;color:{color};margin:0 0 4px 0;'>
                            {code}</p>
                        <p style='font-family:IBM Plex Sans,sans-serif;font-size:12px;
                                  color:#c8cdd6;margin:0 0 8px 0;'>{name}</p>
                        <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                                  color:#5a6478;margin:0;'>{reason}</p>
                        """, border_color=color, bg='#1a0a0a'), unsafe_allow_html=True)

            # ── Prediction history ──
            if len(st.session_state.history) > 1:
                st.markdown("<br/>", unsafe_allow_html=True)
                with st.expander("PREDICTION HISTORY"):
                    hist_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(
                        hist_df.style.applymap(
                            lambda v: 'color: #dc4f4f' if v == 'FAILURE' else 'color: #3dbc74',
                            subset=['Prediction']
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    if st.button("CLEAR HISTORY"):
                        st.session_state.history = []
                        st.rerun()

        with tab2:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("F1 Score",   "0.703")
            with c2: st.metric("AUC-ROC",    "0.978")
            with c3: st.metric("Precision",  "0.650")
            with c4: st.metric("Recall",     "0.765")

            st.markdown("<br/>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(card("""
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                          Model Comparison</p>
                <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:11px;'>
                    <tr style='border-bottom:1px solid #1e2329;'>
                        <td style='color:#5a6478;padding:6px 8px 6px 0;'>Model</td>
                        <td style='color:#5a6478;padding:6px 4px;'>Recall</td>
                        <td style='color:#5a6478;padding:6px 4px;'>Prec.</td>
                        <td style='color:#5a6478;padding:6px 0 6px 4px;'>F1</td>
                    </tr>
                    <tr><td style='color:#3d4860;padding:6px 8px 6px 0;'>Balanced RF</td>
                        <td style='color:#3dbc74;padding:6px 4px;'>0.941</td>
                        <td style='color:#dc4f4f;padding:6px 4px;'>0.234</td>
                        <td style='color:#3d4860;padding:6px 0 6px 4px;'>0.375</td></tr>
                    <tr style='background:#161b24;'>
                        <td style='color:#e8a23a;padding:6px 8px;'>XGBoost ✓</td>
                        <td style='color:#c8cdd6;padding:6px 4px;'>0.765</td>
                        <td style='color:#c8cdd6;padding:6px 4px;'>0.650</td>
                        <td style='color:#e8a23a;padding:6px 0 6px 4px;'>0.703</td></tr>
                    <tr><td style='color:#3d4860;padding:6px 8px 6px 0;'>RF+Threshold</td>
                        <td style='color:#3dbc74;padding:6px 4px;'>1.000</td>
                        <td style='color:#dc4f4f;padding:6px 4px;'>0.034</td>
                        <td style='color:#3d4860;padding:6px 0 6px 4px;'>0.066</td></tr>
                    <tr><td style='color:#3d4860;padding:6px 8px 6px 0;'>Ensemble</td>
                        <td style='color:#3dbc74;padding:6px 4px;'>0.794</td>
                        <td style='color:#c8cdd6;padding:6px 4px;'>0.587</td>
                        <td style='color:#3d4860;padding:6px 0 6px 4px;'>0.675</td></tr>
                </table>
                """, border_color="#1e2329"), unsafe_allow_html=True)

            with col_b:
                st.markdown(card("""
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                          Operational Impact — 10,000 Machines/Year</p>
                <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:11px;'>
                    <tr style='border-bottom:1px solid #1e2329;'>
                        <td style='color:#5a6478;padding:6px 8px 6px 0;'>Model</td>
                        <td style='color:#5a6478;padding:6px 4px;'>Caught</td>
                        <td style='color:#5a6478;padding:6px 4px;'>FP</td>
                        <td style='color:#5a6478;padding:6px 0 6px 4px;'>Cost</td></tr>
                    <tr style='background:#161b24;'>
                        <td style='color:#e8a23a;padding:6px 8px;'>XGBoost ✓</td>
                        <td style='color:#c8cdd6;padding:6px 4px;'>259/339</td>
                        <td style='color:#c8cdd6;padding:6px 4px;'>140</td>
                        <td style='color:#c8cdd6;padding:6px 0 6px 4px;'>$8.1M</td></tr>
                    <tr><td style='color:#3d4860;padding:6px 8px 6px 0;'>Ensemble</td>
                        <td style='color:#c8cdd6;padding:6px 4px;'>269/339</td>
                        <td style='color:#3d4860;padding:6px 4px;'>189</td>
                        <td style='color:#3d4860;padding:6px 0 6px 4px;'>$7.2M</td></tr>
                    <tr><td style='color:#3d4860;padding:6px 8px 6px 0;'>Balanced RF</td>
                        <td style='color:#3dbc74;padding:6px 4px;'>319/339</td>
                        <td style='color:#dc4f4f;padding:6px 4px;'>1,050</td>
                        <td style='color:#3d4860;padding:6px 0 6px 4px;'>$3.1M</td></tr>
                </table>
                """, border_color="#1e2329"), unsafe_allow_html=True)

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(card("""
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                          Project</p>
                <p style='font-family:IBM Plex Sans,sans-serif;font-size:13px;color:#8a95a8;
                          line-height:1.7;margin:0 0 12px 0;'>
                    Predictive maintenance system built on the UCI AI4I 2020 dataset.
                    Uses XGBoost with domain-informed feature engineering to predict
                    industrial machine failure from sensor data.
                </p>
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;
                          margin:16px 0 8px 0;'>Engineered Features</p>
                <table style='font-family:IBM Plex Mono,monospace;font-size:11px;width:100%;'>
                    <tr><td style='color:#e8a23a;padding:4px 12px 4px 0;'>Temp_diff</td>
                        <td style='color:#8a95a8;'>Process − Air → HDF risk</td></tr>
                    <tr><td style='color:#e8a23a;padding:4px 12px 4px 0;'>Power</td>
                        <td style='color:#8a95a8;'>Torque × ω → PWF boundary</td></tr>
                    <tr><td style='color:#e8a23a;padding:4px 12px 4px 0;'>Wear_Torque</td>
                        <td style='color:#8a95a8;'>Wear × Torque → OSF risk</td></tr>
                </table>
                """, border_color="#1e2329"), unsafe_allow_html=True)

            with col_b:
                st.markdown(card("""
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;margin:0 0 12px 0;'>
                          Built By</p>
                <p style='font-family:IBM Plex Mono,monospace;font-size:16px;
                          color:#c8cdd6;font-weight:600;margin:0 0 4px 0;'>Muhaddisa</p>
                <p style='font-family:IBM Plex Sans,sans-serif;font-size:13px;
                          color:#5a6478;margin:0 0 16px 0;'>
                    BS Artificial Intelligence — 6th Semester<br/>
                    Shifa Tameer-e-Millat University, Pakistan
                </p>
                <hr/>
                <p style='font-family:IBM Plex Mono,monospace;font-size:10px;
                          letter-spacing:0.12em;text-transform:uppercase;color:#5a6478;
                          margin:12px 0 8px 0;'>Stack</p>
                <p style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#3d4860;line-height:2;'>
                    Python &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Scikit-learn<br/>
                    Streamlit &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Plotly<br/>
                    imbalanced-learn &nbsp;·&nbsp; Joblib
                </p>
                """, border_color="#1e2329"), unsafe_allow_html=True)
