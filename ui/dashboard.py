from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.preprocessing import prepare_data, FEATURE_COLS
from utils.random_forest import RandomForest
from utils.feature_importance import (
    patient_explanation, permutation_importance,
    normalize_importances, get_top_features, FEATURE_NAMES
)
from utils.llm_explanation import (
    get_clinical_explanation, get_patient_explanation,
    denormalize_value, FEATURE_RANGES
)

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="TraceMed AI",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .risk-high   { color: #ff4b4b; font-size: 3rem; font-weight: bold; }
    .risk-medium { color: #ffa500; font-size: 3rem; font-weight: bold; }
    .risk-low    { color: #00cc44; font-size: 3rem; font-weight: bold; }
    .card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #2e3250;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .explanation-box {
        background: #1a1f35;
        border-left: 4px solid #4f8ef7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #e2e8f0;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .patient-box {
        background: #1a2e1a;
        border-left: 4px solid #00cc44;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #e2e8f0;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2e3250;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  LOAD & TRAIN  (cached so it runs once)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Training Random Forest model...")
def load_model():
    X_train, y_train, X_test, y_test, stats = prepare_data('data/heart.csv')
    forest = RandomForest(n_trees=50, max_depth=8, min_samples=3)
    forest.fit(X_train, y_train)

    # global feature importance
    importances, _ = permutation_importance(forest, X_test, y_test, n_repeats=5)
    norm            = normalize_importances(importances)
    top_features    = get_top_features(norm, top_n=13)

    return forest, X_train, X_test, y_test, stats, top_features

forest, X_train, X_test, y_test, stats, top_features = load_model()


# ─────────────────────────────────────────
#  NORMALIZE PATIENT INPUT
# ─────────────────────────────────────────
def normalize_input(raw_values, stats):
    row = []
    for key in FEATURE_COLS:
        mn = stats[key]['min']
        mx = stats[key]['max']
        val = raw_values[key]
        if mx - mn == 0:
            row.append(0.0)
        else:
            row.append((val - mn) / (mx - mn))
    return row


# ─────────────────────────────────────────
#  RISK GAUGE CHART
# ─────────────────────────────────────────
def risk_gauge(risk_score):
    color = "#ff4b4b" if risk_score >= 70 else "#ffa500" if risk_score >= 40 else "#00cc44"
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = risk_score,
        number = {'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge  = {
            'axis': {'range': [0, 100], 'tickcolor': "#a0aec0",
                     'tickfont': {'color': '#a0aec0'}},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': "#1e2130",
            'steps': [
                {'range': [0,  40], 'color': '#0d2b0d'},
                {'range': [40, 70], 'color': '#2b2000'},
                {'range': [70, 100],'color': '#2b0000'},
            ],
            'threshold': {
                'line':  {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="#0f1117",
        font={'color': '#e2e8f0'}
    )
    return fig


# ─────────────────────────────────────────
#  CONTRIBUTION BAR CHART
# ─────────────────────────────────────────
def contribution_chart(contributions):
    names  = [c['feature_name']  for c in contributions]
    values = [c['contribution']  for c in contributions]
    colors = ["#ff4b4b" if v > 0 else "#00cc44" for v in values]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = names,
        orientation = 'h',
        marker_color= colors,
        text        = [f"{v:+.1f}%" for v in values],
        textposition= 'outside',
        textfont    = {'color': '#e2e8f0', 'size': 12}
    ))
    fig.update_layout(
        height      = 320,
        margin      = dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="#0f1117",
        plot_bgcolor= "#1e2130",
        xaxis       = dict(color='#a0aec0', zeroline=True,
                           zerolinecolor='#4a5568', zerolinewidth=2),
        yaxis       = dict(color='#e2e8f0', autorange='reversed'),
        font        = {'color': '#e2e8f0'}
    )
    return fig


# ─────────────────────────────────────────
#  GLOBAL IMPORTANCE CHART
# ─────────────────────────────────────────
def global_importance_chart(top_features):
    names  = [f['feature_name'] for f in top_features[:8]]
    values = [f['importance']   for f in top_features[:8]]

    fig = px.bar(
        x=values, y=names, orientation='h',
        color=values,
        color_continuous_scale=[[0,'#1a3a5c'],[0.5,'#4f8ef7'],[1,'#ff4b4b']]
    )
    fig.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=20),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e2130",
        xaxis=dict(color='#a0aec0', title="Importance %"),
        yaxis=dict(color='#e2e8f0', autorange='reversed'),
        coloraxis_showscale=False,
        font={'color': '#e2e8f0'}
    )
    return fig


# ═════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════

# ── Header ───────────────────────────────
st.markdown("# 🫀 TraceMed AI")
st.markdown("##### Explainable Medical Risk Prediction & Clinical Decision Support")
st.markdown("---")

# ── Sidebar — Patient Input ───────────────
with st.sidebar:
    st.markdown("## 👤 Patient Data")
    st.markdown("Enter patient values below:")

    age      = st.slider("Age (years)",             29, 77,  55)
    sex      = st.selectbox("Sex",                  [("Male", 1), ("Female", 0)],
                             format_func=lambda x: x[0])
    cp       = st.selectbox("Chest Pain Type",
                             [(1,"Typical Angina"),(2,"Atypical Angina"),
                              (3,"Non-Anginal"),(4,"Asymptomatic")],
                             format_func=lambda x: x[1])
    trestbps = st.slider("Blood Pressure (mmHg)",   94, 200, 130)
    chol     = st.slider("Cholesterol (mg/dL)",     126, 564, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120",
                             [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])
    restecg  = st.selectbox("Resting ECG",
                             [(0,"Normal"),(1,"ST Abnormality"),(2,"LV Hypertrophy")],
                             format_func=lambda x: x[1])
    thalach  = st.slider("Max Heart Rate (bpm)",    71, 202, 150)
    exang    = st.selectbox("Exercise Induced Angina",
                             [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])
    oldpeak  = st.slider("ST Depression",           0.0, 6.2, 1.0, step=0.1)
    slope    = st.selectbox("Slope of ST Segment",
                             [(1,"Upsloping"),(2,"Flat"),(3,"Downsloping")],
                             format_func=lambda x: x[1])
    ca       = st.slider("Major Vessels (0–3)",     0, 3, 0)
    thal     = st.selectbox("Thalassemia",
                             [(3,"Normal"),(6,"Fixed Defect"),(7,"Reversible Defect")],
                             format_func=lambda x: x[1])

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ── Tabs ─────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Risk Analysis", "🔬 What-If Simulator", "📈 Model Insights"])


# ════════════════════════════════════════
#  TAB 1 — RISK ANALYSIS
# ════════════════════════════════════════
with tab1:
    if predict_btn:
        # build raw input
        raw_input = {
            'age': age, 'sex': sex[1], 'cp': cp[0],
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs[0],
            'restecg': restecg[0], 'thalach': thalach, 'exang': exang[0],
            'oldpeak': oldpeak, 'slope': slope[0], 'ca': ca, 'thal': thal[0]
        }

        # normalize and predict
        patient_row    = normalize_input(raw_input, stats)
        label, proba   = forest.predict_one_row(patient_row)
        risk_score     = round(proba * 100, 1)
        contributions, _ = patient_explanation(patient_row, X_train, forest, top_n=5)

        # store in session
        st.session_state['risk_score']     = risk_score
        st.session_state['label']          = label
        st.session_state['contributions']  = contributions
        st.session_state['patient_row']    = patient_row
        st.session_state['raw_input']      = raw_input

    if 'risk_score' in st.session_state:
        risk_score    = st.session_state['risk_score']
        label         = st.session_state['label']
        contributions = st.session_state['contributions']
        patient_row   = st.session_state['patient_row']

        risk_level = "🔴 HIGH RISK"    if risk_score >= 70 \
                else "🟡 MODERATE RISK" if risk_score >= 40 \
                else "🟢 LOW RISK"

        # ── Row 1: Gauge + Risk Label ─────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="section-title">Heart Disease Risk Score</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(risk_score), use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Risk Summary</div>',
                        unsafe_allow_html=True)
            st.markdown(f"### {risk_level}")
            st.markdown(f"**Prediction:** {'❤️‍🔥 Heart Disease Detected' if label == 1 else '✅ No Heart Disease'}")
            st.markdown("---")
            st.markdown("**Top Risk Factors:**")
            for c in contributions[:3]:
                arrow = "🔺" if c['contribution'] > 0 else "🔻"
                st.markdown(f"{arrow} **{c['feature_name']}** — {c['contribution']:+.1f}%")

        # ── Row 2: Contribution Chart ─────
        st.markdown("---")
        st.markdown("### 📊 Feature Contribution to Risk")
        st.plotly_chart(contribution_chart(contributions), use_container_width=True)
        st.caption("🔴 Red = increases risk &nbsp;&nbsp; 🟢 Green = decreases risk")

        # ── Row 3: AI Explanations ────────
        st.markdown("---")
        st.markdown("### 🤖 AI Clinical Explanation")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("**👨‍⚕️ Doctor View**")
            with st.spinner("Generating clinical explanation..."):
                doc_text = get_clinical_explanation(risk_score, contributions, patient_row)
            st.markdown(f'<div class="explanation-box">{doc_text}</div>',
                        unsafe_allow_html=True)

        with exp_col2:
            st.markdown("**🧑 Patient View**")
            with st.spinner("Generating patient explanation..."):
                pat_text = get_patient_explanation(risk_score, contributions, patient_row)
            st.markdown(f'<div class="patient-box">{pat_text}</div>',
                        unsafe_allow_html=True)

    else:
        st.info("👈 Enter patient data in the sidebar and click **Predict Risk** to begin.")


# ════════════════════════════════════════
#  TAB 2 — WHAT-IF SIMULATOR
# ════════════════════════════════════════
with tab2:
    st.markdown("### 🔬 What-If Risk Simulator")
    st.markdown("Adjust individual values to see how lifestyle changes affect risk.")

    if 'raw_input' not in st.session_state:
        st.info("👈 First run a prediction in the **Risk Analysis** tab.")
    else:
        raw = st.session_state['raw_input']
        original_risk = st.session_state['risk_score']

        st.markdown(f"**Original Risk: `{original_risk}%`**")
        st.markdown("---")

        w1, w2 = st.columns(2)

        with w1:
            new_bp    = st.slider("Blood Pressure (mmHg)", 94,  200, int(raw['trestbps']), key="wi_bp")
            new_chol  = st.slider("Cholesterol (mg/dL)",   126, 564, int(raw['chol']),     key="wi_chol")
            new_thal  = st.slider("Max Heart Rate (bpm)",  71,  202, int(raw['thalach']),  key="wi_hr")

        with w2:
            new_ca    = st.slider("Major Vessels (0–3)",   0, 3, int(raw['ca']),           key="wi_ca")
            new_smoke = st.selectbox("Exercise Angina",
                                     [(0,"No"),(1,"Yes")],
                                     index=int(raw['exang']),
                                     format_func=lambda x: x[1],
                                     key="wi_exang")
            new_old   = st.slider("ST Depression",         0.0, 6.2, float(raw['oldpeak']),
                                  step=0.1, key="wi_old")

        if st.button("⚡ Recalculate Risk", type="primary"):
            modified = dict(raw)
            modified['trestbps'] = new_bp
            modified['chol']     = new_chol
            modified['thalach']  = new_thal
            modified['ca']       = new_ca
            modified['exang']    = new_smoke[0]
            modified['oldpeak']  = new_old

            mod_row         = normalize_input(modified, stats)
            _, new_proba    = forest.predict_one_row(mod_row)
            new_risk        = round(new_proba * 100, 1)
            delta           = new_risk - original_risk

            r1, r2, r3 = st.columns(3)
            r1.metric("Original Risk",  f"{original_risk}%")
            r2.metric("New Risk",       f"{new_risk}%")
            r3.metric("Change",         f"{delta:+.1f}%",
                      delta_color="inverse")

            if delta < 0:
                st.success(f"✅ Risk reduced by {abs(delta):.1f}% with these changes!")
            elif delta > 0:
                st.error(f"⚠️ Risk increased by {delta:.1f}% with these changes.")
            else:
                st.info("No change in risk.")

            # comparison gauge
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**Before**")
                st.plotly_chart(risk_gauge(original_risk), use_container_width=True)
            with comp_col2:
                st.markdown("**After**")
                st.plotly_chart(risk_gauge(new_risk), use_container_width=True)


# ════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Global Feature Importance")
    st.markdown("Which features matter most across all patients:")
    st.plotly_chart(global_importance_chart(top_features), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Model Performance")

    preds   = forest.predict(X_test)
    tp = sum(1 for p, a in zip(preds, y_test) if p == 1 and a == 1)
    tn = sum(1 for p, a in zip(preds, y_test) if p == 0 and a == 0)
    fp = sum(1 for p, a in zip(preds, y_test) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds, y_test) if p == 0 and a == 1)

    accuracy  = (tp + tn) / len(y_test) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{accuracy:.1f}%")
    m2.metric("Precision", f"{precision:.1f}%")
    m3.metric("Recall",    f"{recall:.1f}%")
    m4.metric("F1 Score",  f"{f1:.1f}%")

    st.markdown("---")
    st.markdown("### 🧠 About TraceMed AI")
    st.markdown("""
| Component | Technology |
|---|---|
| Machine Learning | Random Forest (built from scratch) |
| Explainability | Permutation Importance (built from scratch) |
| LLM Explanation | Llama 3 via Groq API |
| Dataset | UCI Cleveland Heart Disease (207 patients) |
| Frontend | Streamlit + Plotly |
""")