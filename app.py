import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetIQ — Clinical Risk Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
:root {
    --bg: #0B1120; --surface: #111827; --card: #1A2535; --border: #243048;
    --teal: #0FCFB0; --teal-dim: #0A9E88; --amber: #F5A623; --red: #F05252;
    --blue: #4F8EF7; --text: #E8EDF7; --muted: #7A8BAD;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1626 0%, #111827 100%); border-right: 1px solid var(--border); }
.stButton > button {
    background: linear-gradient(135deg, #0FCFB0 0%, #0A9E88 100%) !important;
    color: #0B1120 !important; font-weight: 700 !important; border-radius: 10px !important; width: 100%;
}
.metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 20px 24px; position: relative; }
.risk-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700; }
.badge-low { background: rgba(15,207,176,0.15); color: #0FCFB0; border: 1px solid #0FCFB0; }
.badge-mod { background: rgba(245,166,35,0.15); color: #F5A623; border: 1px solid #F5A623; }
.badge-high { background: rgba(240,82,82,0.15); color: #F05252; border: 1px solid #F05252; }
.rec-item { display: flex; gap: 12px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except:
        return None, None, False

model, scaler, model_loaded = load_model_files()

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL REFERENCE RANGES
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_REF = {
    "Glucose": {"low": 70, "normal": 99, "elevated": 125, "unit": "mg/dL", "label": "Blood Glucose"},
    "BloodPressure": {"low": 60, "normal": 80, "elevated": 90, "unit": "mmHg", "label": "Blood Pressure"},
    "BMI": {"low": 18.5, "normal": 24.9, "elevated": 29.9, "unit": "kg/m²", "label": "BMI"},
    "SkinThickness": {"low": 10, "normal": 28, "elevated": 40, "unit": "mm", "label": "Skin Thickness"},
    "Insulin": {"low": 16, "normal": 166, "elevated": 300, "unit": "μU/mL", "label": "Insulin Level"},
    "DiabetesPedigreeFunction": {"low": 0.0, "normal": 0.5, "elevated": 1.0, "unit": "score", "label": "Diabetes Pedigree"},
    "Pregnancies": {"low": 0, "normal": 4, "elevated": 7, "unit": "count", "label": "Pregnancies"},
    "Age": {"low": 21, "normal": 40, "elevated": 55, "unit": "years", "label": "Age"},
}

FEATURE_DESCRIPTIONS = {
    "Glucose": "Plasma glucose concentration 2 hours after oral glucose tolerance test.",
    "BloodPressure": "Diastolic blood pressure.",
    "BMI": "Body Mass Index.",
    "SkinThickness": "Triceps skin fold thickness.",
    "Insulin": "2-hour serum insulin.",
    "DiabetesPedigreeFunction": "Genetic risk score encoding family history.",
    "Pregnancies": "Number of gestational cycles.",
    "Age": "Biological age.",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 DiabetIQ\n**Clinical Risk Intelligence**")
    st.divider()
    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 117)
    bp = st.slider("Blood Pressure (mmHg)", 0, 122, 72)
    skin = st.slider("Skin Thickness (mm)", 0, 99, 23)
    insulin = st.slider("Insulin (μU/mL)", 0, 846, 30)
    bmi = st.slider("BMI (kg/m²)", 0.0, 67.1, 32.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.4, 0.37)
    age = st.slider("Age (years)", 21, 81, 29)
    analyze = st.button("🔬 Analyze Risk Profile")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (CRITICAL FIX FOR SCALER ORDER)
# ─────────────────────────────────────────────────────────────────────────────
def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # 1. Insulin-to-Glucose ratio
    d["Insulin_Glucose_Ratio"] = d["Insulin"] / d["Glucose"].replace(0, np.nan).fillna(1)
    
    # 2. BMI Class
    d["BMI_Class"] = pd.cut(
        d["BMI"],
        bins=[0, 18.5, 25.0, 30.0, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)
    
    # 3. Age_Glucose interaction
    d["Age_Glucose"] = d["Age"] * d["Glucose"]

    # ── FIXED ORDER: EXACTLY matches your scaler.pkl ──
    FIT_COLUMNS = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Insulin_Glucose_Ratio",
        "BMI_Class",
        "Age_Glucose"
    ]
    return d[FIT_COLUMNS]

input_df = pd.DataFrame({
    "Pregnancies": [pregnancies], "Glucose": [glucose], "BloodPressure": [bp],
    "SkinThickness": [skin], "Insulin": [insulin], "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf], "Age": [age]
})

# ─────────────────────────────────────────────────────────────────────────────
# RISK CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
if model_loaded:
    input_full = build_model_input(input_df)
    input_scaled = scaler.transform(input_full)
    risk_score = model.predict_proba(input_scaled)[0][1]
    risk_pct = risk_score * 100
else:
    st.error("Model files not found. Check if scaler.pkl and diabetes_model.pkl are in the folder.")
    st.stop()

# Tier Logic
if risk_pct < 30:
    tier, badge_cls, tier_color = "Low Risk", "badge-low", "#0FCFB0"
elif risk_pct < 65:
    tier, badge_cls, tier_color = "Moderate Risk", "badge-mod", "#F5A623"
else:
    tier, badge_cls, tier_color = "High Risk", "badge-high", "#F05252"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-family:Playfair Display;'>Clinical Diabetes <span>Risk Intelligence</span></h1>", unsafe_allow_html=True)

kpi_cols = st.columns(4)
for col, key, val, unit in zip(kpi_cols, ["Glucose", "BloodPressure", "BMI", "Age"], [glucose, bp, bmi, age], ["mg/dL", "mmHg", "kg/m²", "years"]):
    col.markdown(f"""
    <div class='metric-card'>
        <div style='color:#7A8BAD; font-size:10px;'>{CLINICAL_REF[key]['label']}</div>
        <div style='font-size:28px; font-weight:700;'>{val} <span style='font-size:12px;'>{unit}</span></div>
    </div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Risk Overview", "📋 Clinical Report"])

with tab1:
    l, r = st.columns([1, 1])
    with l:
        st.markdown(f"""
        <div style='background:#1A2535; padding:30px; border-radius:15px; border:1px solid #243048;'>
            <h2 style='color:#7A8BAD; font-size:12px;'>PROBABILITY OF DIABETES</h2>
            <div style='font-size:60px; font-weight:700; color:{tier_color};'>{risk_pct:.1f}%</div>
            <span class='risk-badge {badge_cls}'>{tier}</span>
        </div>""", unsafe_allow_html=True)
    with r:
        # Simple Matplotlib Horizontal Bar for Feature Importance
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#1A2535")
        ax.set_facecolor("#1A2535")
        
        # Determine relative importance for display
        impacts = input_full.iloc[0].values
        names = input_full.columns
        ax.barh(names, impacts, color="#0FCFB0")
        ax.tick_params(colors="white")
        st.pyplot(fig)

with tab2:
    st.markdown("### Assessment Report")
    st.write(f"Patient exhibits a {tier} profile based on provided biomarkers.")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))
