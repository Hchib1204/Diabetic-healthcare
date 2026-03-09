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
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetIQ — Clinical Risk Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark clinical theme with teal & amber accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

/* ── Root variables ── */
:root {
    --bg:         #0B1120;
    --surface:    #111827;
    --card:       #1A2535;
    --border:     #243048;
    --teal:       #0FCFB0;
    --teal-dim:   #0A9E88;
    --amber:      #F5A623;
    --red:        #F05252;
    --blue:       #4F8EF7;
    --text:       #E8EDF7;
    --muted:      #7A8BAD;
    --font:       'DM Sans', sans-serif;
    --mono:       'DM Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"]  { font-family: var(--font); }
.stApp { background: var(--bg); color: var(--text); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1626 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSlider > div { padding: 0 4px; }
[data-testid="stSidebar"] label {
    color: var(--muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Slider accent ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background: var(--teal) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0FCFB0 0%, #0A9E88 100%) !important;
    color: #0B1120 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.04em;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    width: 100%;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(15,207,176,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(15,207,176,0.4) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--blue));
}
.metric-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-unit {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-low    { background: rgba(15,207,176,0.15); color: #0FCFB0; border: 1px solid #0FCFB0; }
.badge-mod    { background: rgba(245,166,35,0.15);  color: #F5A623; border: 1px solid #F5A623; }
.badge-high   { background: rgba(240,82,82,0.15);   color: #F05252; border: 1px solid #F05252; }

/* ── Section headers ── */
.section-title {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Recommendation pills ── */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.rec-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.rec-text { font-size: 13.5px; color: var(--text); line-height: 1.55; }
.rec-title { font-weight: 700; font-size: 13px; margin-bottom: 3px; }

/* ── Header hero ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 40px;
    font-weight: 700;
    line-height: 1.15;
    color: var(--text);
}
.hero-title span { color: var(--teal); }
.hero-sub {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.7;
    max-width: 560px;
}

/* ── Gauge container ── */
.gauge-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px 16px 12px; }

/* ── Risk meter segments ── */
.risk-bar-wrap { background: var(--card); border-radius: 12px; padding: 18px 20px; border: 1px solid var(--border); }
.risk-bar-track {
    height: 14px; border-radius: 7px;
    background: linear-gradient(90deg, #0FCFB0 0%, #F5A623 50%, #F05252 100%);
    position: relative;
    margin: 8px 0;
}
.risk-needle {
    position: absolute;
    top: -6px;
    width: 3px; height: 26px;
    background: white;
    border-radius: 2px;
    box-shadow: 0 0 6px rgba(255,255,255,0.6);
    transform: translateX(-50%);
}
.risk-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-top: 5px; }

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tab"] {
    font-size: 12px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

model, scaler, model_loaded = load_model()

if not model_loaded:
    st.error("⚠️ Model files not found — running in **Demo Mode** with simulated predictions.")
    st.info("Place `diabetes_model.pkl` and `scaler.pkl` in the app directory to enable live inference.")

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL REFERENCE RANGES
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_REF = {
    "Glucose":                  {"low": 70,  "normal": 99,   "elevated": 125, "unit": "mg/dL",   "label": "Blood Glucose"},
    "BloodPressure":            {"low": 60,  "normal": 80,   "elevated": 90,  "unit": "mmHg",    "label": "Blood Pressure"},
    "BMI":                      {"low": 18.5,"normal": 24.9, "elevated": 29.9,"unit": "kg/m²",   "label": "BMI"},
    "SkinThickness":            {"low": 10,  "normal": 28,   "elevated": 40,  "unit": "mm",      "label": "Skin Thickness"},
    "Insulin":                  {"low": 16,  "normal": 166,  "elevated": 300, "unit": "μU/mL",   "label": "Insulin Level"},
    "DiabetesPedigreeFunction": {"low": 0.0, "normal": 0.5,  "elevated": 1.0, "unit": "score",   "label": "Diabetes Pedigree"},
    "Pregnancies":              {"low": 0,   "normal": 4,    "elevated": 7,   "unit": "count",   "label": "Pregnancies"},
    "Age":                      {"low": 21,  "normal": 40,   "elevated": 55,  "unit": "years",   "label": "Age"},
}

FEATURE_DESCRIPTIONS = {
    "Glucose":                  "Plasma glucose concentration 2 hours after oral glucose tolerance test. Values >126 mg/dL indicate diabetes.",
    "BloodPressure":            "Diastolic blood pressure. Chronic hypertension worsens insulin resistance and accelerates nephropathy.",
    "BMI":                      "Body Mass Index. Central obesity (BMI>30) is the strongest modifiable risk factor for T2DM.",
    "SkinThickness":            "Triceps skin fold thickness — a proxy for subcutaneous fat distribution and insulin resistance.",
    "Insulin":                  "2-hour serum insulin. Elevated fasting insulin indicates early pancreatic compensation for resistance.",
    "DiabetesPedigreeFunction": "Genetic risk score encoding family history. Higher values indicate stronger hereditary predisposition.",
    "Pregnancies":              "Number of gestational cycles. Each pregnancy with GDM raises lifetime T2DM risk by ~50%.",
    "Age":                      "Biological age. Beta-cell function declines ~1% per year after 40, compounding metabolic risk.",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT VITALS INPUT
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:18px; font-weight:700; color:#E8EDF7; letter-spacing:0.02em;'>🧬 DiabetIQ</div>
        <div style='font-size:10px; color:#7A8BAD; letter-spacing:0.14em; text-transform:uppercase; margin-top:3px;'>Clinical Risk Intelligence</div>
    </div>
    <hr style='margin: 12px 0; border-color:#243048;'>
    <div style='font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:#7A8BAD; font-weight:700; margin-bottom:14px;'>Patient Vitals</div>
    """, unsafe_allow_html=True)

    pregnancies = st.slider("Pregnancies",                    0,    17,  3,    1)
    glucose     = st.slider("Glucose (mg/dL)",                0,   200, 117,   1)
    bp          = st.slider("Blood Pressure (mmHg)",          0,   122,  72,   1)
    skin        = st.slider("Skin Thickness (mm)",            0,    99,  23,   1)
    insulin     = st.slider("Insulin (μU/mL)",                0,   846,  30,   1)
    bmi         = st.slider("BMI (kg/m²)",                  0.0,  67.1, 32.0, 0.1)
    dpf         = st.slider("Diabetes Pedigree Function",   0.0,   2.4,  0.37, 0.01)
    age         = st.slider("Age (years)",                   21,    81,  29,   1)

    st.markdown("<hr style='margin:16px 0; border-color:#243048;'>", unsafe_allow_html=True)
    analyze = st.button("🔬 Analyze Risk Profile")
    st.markdown("""
    <div style='font-size:10px; color:#7A8BAD; line-height:1.6; margin-top:14px;'>
    ⚕️ This tool is a clinical decision-support aid and does not replace professional medical diagnosis.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
input_df = pd.DataFrame({
    "Pregnancies":              [pregnancies],
    "Glucose":                  [glucose],
    "BloodPressure":            [bp],
    "SkinThickness":            [skin],
    "Insulin":                  [insulin],
    "BMI":                      [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age":                      [age],
})

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — matches the pkl that was actually saved
# The scaler requires these 11 columns in this exact order:
#   8 raw features  +  Age_Glucose  +  BMI_Class  +  Insulin_Glucose_Ratio
# ─────────────────────────────────────────────────────────────────────────────
def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Match the engineering logic from the training script
    d["Age_Glucose"] = d["Age"] * d["Glucose"]
    d["BMI_Class"] = pd.cut(
        d["BMI"],
        bins=[0, 18.5, 25.0, 30.0, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float)
    d["Insulin_Glucose_Ratio"] = d["Insulin"] / d["Glucose"].replace(0, np.nan).fillna(1)

    # Use the exact same sequence as Step 1
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
        "Age_Glucose",
    ]
    return d[FIT_COLUMNS]

input_df_full = build_model_input(input_df)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if model_loaded:
    input_scaled = scaler.transform(input_df_full)
    risk_score   = model.predict_proba(input_scaled)[0][1]
    # Per-feature importance via perturbation on the 8 raw slider columns
    baseline = risk_score
    feature_impacts = {}
    for col in input_df.columns:
        perturbed_df     = input_df.copy()
        perturbed_df[col] = perturbed_df[col] + perturbed_df[col].abs().mean() * 0.1 + 1e-3
        perturbed_full   = build_model_input(perturbed_df)
        perturbed_scaled = scaler.transform(perturbed_full)
        perturbed_score  = model.predict_proba(perturbed_scaled)[0][1]
        feature_impacts[col] = round((perturbed_score - baseline) * 100, 2)
else:
    # Demo mode — deterministic sim from inputs
    score_raw  = (
        (glucose / 200) * 0.35 +
        (bmi / 67.1)    * 0.20 +
        (age / 81)      * 0.15 +
        (dpf / 2.4)     * 0.15 +
        (insulin / 846) * 0.05 +
        (pregnancies/17)* 0.05 +
        (bp / 122)      * 0.03 +
        (skin / 99)     * 0.02
    )
    risk_score = float(np.clip(score_raw, 0.02, 0.97))
    feature_impacts = {
        "Glucose":                  round((glucose/200)*18, 2),
        "BMI":                      round((bmi/67.1)*12, 2),
        "Age":                      round((age/81)*9, 2),
        "DiabetesPedigreeFunction": round((dpf/2.4)*9, 2),
        "Insulin":                  round((insulin/846)*4, 2),
        "Pregnancies":              round((pregnancies/17)*4, 2),
        "BloodPressure":            round((bp/122)*2, 2),
        "SkinThickness":            round((skin/99)*2, 2),
    }

risk_pct = risk_score * 100

# Risk tier
if risk_pct < 30:
    tier, badge_cls, tier_color, tier_icon = "Low Risk", "badge-low", "#0FCFB0", "✅"
elif risk_pct < 65:
    tier, badge_cls, tier_color, tier_icon = "Moderate Risk", "badge-mod", "#F5A623", "⚠️"
else:
    tier, badge_cls, tier_color, tier_icon = "High Risk", "badge-high", "#F05252", "🔴"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────────────────────────────────────
def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#1A2535")
    ax.set_facecolor("#1A2535")
    for spine in ax.spines.values():
        spine.set_color("#243048")
    ax.tick_params(colors="#7A8BAD", labelsize=9)
    ax.xaxis.label.set_color("#7A8BAD")
    ax.yaxis.label.set_color("#7A8BAD")
    return fig, ax

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# ── HERO HEADER ──
st.markdown(f"""
<div style='padding: 32px 0 20px 0;'>
    <div class='hero-title'>Clinical Diabetes<br><span>Risk Intelligence</span></div>
    <div class='hero-sub' style='margin-top:10px;'>
        AI-powered screening built on the Pima Indians Diabetes dataset.
        Adjust patient vitals in the sidebar, then run a full risk analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOP KPI ROW — live vitals with status colors ──
def status_color(key, val):
    ref = CLINICAL_REF[key]
    if val <= ref["normal"]:  return "#0FCFB0"
    elif val <= ref["elevated"]: return "#F5A623"
    else: return "#F05252"

kpi_cols = st.columns(4)
kpi_items = [
    ("Glucose",       glucose, "mg/dL"),
    ("BloodPressure", bp,      "mmHg"),
    ("BMI",           bmi,     "kg/m²"),
    ("Age",           age,     "years"),
]
for col, (key, val, unit) in zip(kpi_cols, kpi_items):
    sc = status_color(key, val)
    label = CLINICAL_REF[key]["label"]
    col.markdown(f"""
    <div class='metric-card' style='border-top: 3px solid {sc};'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{sc};'>{val}</div>
        <div class='metric-unit'>{unit}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ── MAIN CONTENT TABS ──
tab_overview, tab_vitals, tab_biomarkers, tab_factors, tab_clinical = st.tabs([
    "📊 Risk Overview",
    "🩺 Vital Signs",
    "🔬 Biomarker Profile",
    "🧬 Risk Factors",
    "📋 Clinical Report",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — RISK OVERVIEW
# ════════════════════════════════════════════════════════════
with tab_overview:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        # ── Risk gauge (matplotlib half-donut) ──
        st.markdown("<div class='section-title'>Composite Risk Score</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5.5, 3.0), subplot_kw=dict(polar=False))
        fig.patch.set_facecolor("#1A2535")
        ax.set_facecolor("#1A2535")
        ax.axis("off")

        # Semi-circle track segments
        theta = np.linspace(np.pi, 0, 200)
        r_outer, r_inner = 1.0, 0.65
        zones = [(0, 0.30, "#0FCFB0"), (0.30, 0.65, "#F5A623"), (0.65, 1.0, "#F05252")]
        for z_start, z_end, z_color in zones:
            t_start = np.pi - z_start * np.pi
            t_end   = np.pi - z_end   * np.pi
            t = np.linspace(t_start, t_end, 80)
            x_outer = r_outer * np.cos(t)
            y_outer = r_outer * np.sin(t)
            x_inner = r_inner * np.cos(t[::-1])
            y_inner = r_inner * np.sin(t[::-1])
            ax.fill(
                np.concatenate([x_outer, x_inner]),
                np.concatenate([y_outer, y_inner]),
                color=z_color, alpha=0.85
            )

        # Needle
        needle_angle = np.pi - risk_score * np.pi
        nx = 0.82 * np.cos(needle_angle)
        ny = 0.82 * np.sin(needle_angle)
        ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                   lw=2.5, mutation_scale=14))
        ax.add_patch(plt.Circle((0, 0), 0.06, color="white", zorder=5))

        # Center text
        ax.text(0, 0.12, f"{risk_pct:.1f}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color=tier_color,
                fontfamily="DM Sans")
        ax.text(0, -0.12, tier, ha="center", va="center",
                fontsize=10, color="#7A8BAD", fontfamily="DM Sans")

        # Zone labels
        ax.text(-1.0,  0.05, "LOW",  fontsize=7.5, color="#0FCFB0", ha="center", fontfamily="DM Sans")
        ax.text( 0.0,  1.08, "MOD",  fontsize=7.5, color="#F5A623", ha="center", fontfamily="DM Sans")
        ax.text( 1.0,  0.05, "HIGH", fontsize=7.5, color="#F05252", ha="center", fontfamily="DM Sans")

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.35, 1.25)
        for sp in ax.spines.values(): sp.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Probability breakdown bars ──
        st.markdown("<div class='section-title' style='margin-top:12px;'>Probability Distribution</div>", unsafe_allow_html=True)
        fig2, ax2 = dark_fig(5.5, 1.5)
        categories = ["Non-Diabetic", "Diabetic"]
        values     = [1 - risk_score, risk_score]
        colors     = ["#0FCFB0", tier_color]
        bars = ax2.barh(categories, values, color=colors, height=0.45, edgecolor="none")
        for bar, v in zip(bars, values):
            ax2.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                     f"{v:.1%}", va="center", ha="left", color="white", fontsize=10,
                     fontweight="bold", fontfamily="DM Sans")
        ax2.set_xlim(0, 1.15)
        ax2.set_xlabel("")
        ax2.tick_params(left=False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with right:
        # ── Feature impact bars ──
        st.markdown("<div class='section-title'>Feature Impact on Risk Score</div>", unsafe_allow_html=True)

        sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        labels  = [CLINICAL_REF[k]["label"] for k, _ in sorted_impacts]
        impacts = [v for _, v in sorted_impacts]
        colors_imp = ["#F05252" if v > 0 else "#0FCFB0" for v in impacts]

        fig3, ax3 = dark_fig(5, 3.8)
        bars3 = ax3.barh(labels[::-1], impacts[::-1], color=colors_imp[::-1],
                         height=0.55, edgecolor="none")
        ax3.axvline(0, color="#243048", linewidth=1)
        for bar, v in zip(bars3, impacts[::-1]):
            xpos = v + 0.3 if v >= 0 else v - 0.3
            ha   = "left" if v >= 0 else "right"
            ax3.text(xpos, bar.get_y() + bar.get_height()/2,
                     f"{v:+.1f}%", va="center", ha=ha, fontsize=8.5,
                     color="white", fontfamily="DM Sans")
        ax3.set_xlabel("Risk contribution (%)", color="#7A8BAD", fontsize=9)
        ax3.tick_params(labelsize=8.5)
        for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        # ── Risk summary card ──
        st.markdown(f"""
        <div style='background:#1A2535; border:1px solid #243048; border-radius:14px; padding:20px; margin-top:8px;'>
            <div class='metric-label'>Assessment Summary</div>
            <div style='font-size:32px; font-weight:700; color:{tier_color}; margin:8px 0;'>{risk_pct:.1f}%</div>
            <span class='risk-badge {badge_cls}'>{tier}</span>
            <div style='margin-top:14px; font-size:12.5px; color:#7A8BAD; line-height:1.7;'>
                {"Metabolic indicators are within acceptable ranges. Maintain lifestyle habits and schedule annual screening." if risk_pct < 30 else
                 "Multiple risk factors detected. Medical consultation and targeted lifestyle intervention recommended." if risk_pct < 65 else
                 "Critical risk profile. Immediate clinical evaluation, HbA1c testing, and specialist referral strongly advised."}
            </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — VITAL SIGNS
# ════════════════════════════════════════════════════════════
with tab_vitals:
    st.markdown("<div class='section-title'>Clinical Ranges vs. Patient Values</div>", unsafe_allow_html=True)

    vitals_to_show = ["Glucose", "BloodPressure", "BMI", "Insulin"]
    v_cols = st.columns(2)

    for idx, key in enumerate(vitals_to_show):
        ref  = CLINICAL_REF[key]
        val  = input_df[key].values[0]
        col  = v_cols[idx % 2]

        # Normalise to 0-1 for bar plotting
        v_max   = ref["elevated"] * 1.5
        val_pct = min(val / v_max, 1.0)
        nor_pct = ref["normal"]   / v_max
        ele_pct = ref["elevated"] / v_max

        sc = status_color(key, val)

        with col:
            fig_v, ax_v = dark_fig(5, 1.8)
            # Background track
            ax_v.barh(0, 1.0, color="#243048", height=0.45, edgecolor="none", left=0)
            # Normal zone
            ax_v.barh(0, nor_pct, color="#0FCFB0", alpha=0.25, height=0.45, edgecolor="none", left=0)
            # Elevated zone
            ax_v.barh(0, ele_pct - nor_pct, color="#F5A623", alpha=0.25,
                      height=0.45, edgecolor="none", left=nor_pct)
            # High zone
            ax_v.barh(0, 1.0 - ele_pct, color="#F05252", alpha=0.25,
                      height=0.45, edgecolor="none", left=ele_pct)
            # Patient value
            ax_v.barh(0, val_pct, color=sc, height=0.28, edgecolor="none", left=0, alpha=0.9)
            # Marker
            ax_v.axvline(val_pct, color="white", linewidth=2, ymin=0.18, ymax=0.82)
            ax_v.set_xlim(0, 1.0)
            ax_v.axis("off")
            ax_v.set_title(
                f"{ref['label']}:  {val} {ref['unit']}",
                color=sc, fontsize=11, fontweight="bold",
                pad=4, loc="left", fontfamily="DM Sans"
            )
            # Zone legends
            for x_pct, label, lc in [
                (nor_pct/2, "Normal", "#0FCFB0"),
                ((nor_pct+ele_pct)/2, "Elevated", "#F5A623"),
                ((ele_pct+1)/2, "High", "#F05252"),
            ]:
                ax_v.text(x_pct, -0.42, label, ha="center", va="top",
                          fontsize=7.5, color=lc, fontfamily="DM Sans")
            fig_v.tight_layout(pad=0.5)
            col.pyplot(fig_v, use_container_width=True)
            plt.close(fig_v)

    # ── Radar chart ──
    st.markdown("<div class='section-title' style='margin-top:16px;'>Multi-Dimensional Profile vs. Healthy Baseline</div>", unsafe_allow_html=True)

    radar_keys   = ["Glucose", "BloodPressure", "BMI", "SkinThickness", "Insulin", "Age"]
    radar_labels = [CLINICAL_REF[k]["label"] for k in radar_keys]
    patient_norm = [
        min(input_df[k].values[0] / (CLINICAL_REF[k]["elevated"] * 1.4), 1.0)
        for k in radar_keys
    ]
    healthy_norm = [CLINICAL_REF[k]["normal"] / (CLINICAL_REF[k]["elevated"] * 1.4) for k in radar_keys]

    N    = len(radar_keys)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    patient_norm += patient_norm[:1]
    healthy_norm += healthy_norm[:1]

    fig_r, ax_r = plt.subplots(figsize=(5.5, 4), subplot_kw=dict(polar=True))
    fig_r.patch.set_facecolor("#1A2535")
    ax_r.set_facecolor("#1A2535")
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(radar_labels, color="#7A8BAD", fontsize=8.5, fontfamily="DM Sans")
    ax_r.set_ylim(0, 1)
    ax_r.yaxis.set_tick_params(labelleft=False)
    ax_r.grid(color="#243048", linewidth=0.8)
    ax_r.spines["polar"].set_color("#243048")

    ax_r.fill(angles, healthy_norm, alpha=0.18, color="#0FCFB0")
    ax_r.plot(angles, healthy_norm, color="#0FCFB0", linewidth=1.5, linestyle="--", label="Healthy Baseline")

    ax_r.fill(angles, patient_norm, alpha=0.25, color=tier_color)
    ax_r.plot(angles, patient_norm, color=tier_color, linewidth=2.2, label="Patient Profile")

    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                labelcolor="#E8EDF7", fontsize=9,
                facecolor="#1A2535", edgecolor="#243048")
    fig_r.tight_layout()

    rc1, rc2, rc3 = st.columns([1, 1.2, 1])
    with rc2:
        st.pyplot(fig_r, use_container_width=True)
    plt.close(fig_r)

# ════════════════════════════════════════════════════════════
# TAB 3 — BIOMARKER PROFILE
# ════════════════════════════════════════════════════════════
with tab_biomarkers:
    st.markdown("<div class='section-title'>All Biomarkers — Patient vs. Clinical Thresholds</div>", unsafe_allow_html=True)

    all_keys = list(CLINICAL_REF.keys())
    all_labels= [CLINICAL_REF[k]["label"] for k in all_keys]
    patient_vals = [float(input_df[k].values[0]) for k in all_keys]
    normal_vals  = [CLINICAL_REF[k]["normal"] for k in all_keys]
    elev_vals    = [CLINICAL_REF[k]["elevated"] for k in all_keys]

    # Normalize each to its own elevated threshold for visual comparison
    p_norm = [min(pv / (ev * 1.5), 1.0) for pv, ev in zip(patient_vals, elev_vals)]
    n_norm = [nv / (ev * 1.5) for nv, ev in zip(normal_vals, elev_vals)]

    fig_b, ax_b = dark_fig(10, 3.5)
    x = np.arange(len(all_keys))
    w = 0.35
    bar_colors = [status_color(k, float(input_df[k].values[0])) for k in all_keys]
    ax_b.bar(x - w/2, n_norm, w, label="Normal Threshold", color="#243048", edgecolor="#3A4A66")
    bars_b = ax_b.bar(x + w/2, p_norm, w, label="Patient Value", color=bar_colors, edgecolor="none", alpha=0.9)

    # Value labels on patient bars
    for bar, pv, key in zip(bars_b, patient_vals, all_keys):
        unit = CLINICAL_REF[key]["unit"]
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f"{pv:.1f}", ha="center", va="bottom", fontsize=7.5,
                  color="white", fontweight="bold", fontfamily="DM Sans")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(all_labels, rotation=15, ha="right", fontsize=8.5)
    ax_b.set_ylabel("Normalised value", fontsize=9, color="#7A8BAD")
    ax_b.legend(fontsize=9, facecolor="#1A2535", edgecolor="#243048", labelcolor="#E8EDF7")
    for sp in ["top", "right"]: ax_b.spines[sp].set_visible(False)
    fig_b.tight_layout()
    st.pyplot(fig_b, use_container_width=True)
    plt.close(fig_b)

    # ── Per-biomarker detail cards ──
    st.markdown("<div class='section-title' style='margin-top:16px;'>Biomarker Encyclopedia</div>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    for i, key in enumerate(all_keys):
        ref = CLINICAL_REF[key]
        val = float(input_df[key].values[0])
        sc  = status_color(key, val)
        status_word = "Normal" if val <= ref["normal"] else ("Elevated" if val <= ref["elevated"] else "High")
        col = b1 if i % 2 == 0 else b2
        with col:
            with st.expander(f"{ref['label']} — {val} {ref['unit']}  [{status_word}]"):
                st.markdown(f"""
                <div style='font-size:12.5px; color:#7A8BAD; line-height:1.7; margin-bottom:10px;'>
                {FEATURE_DESCRIPTIONS[key]}
                </div>
                <table style='width:100%; font-size:11.5px; border-collapse:collapse;'>
                  <tr><td style='color:#7A8BAD; padding:3px 0;'>Normal range</td>
                      <td style='color:#0FCFB0; text-align:right;'>≤ {ref['normal']} {ref['unit']}</td></tr>
                  <tr><td style='color:#7A8BAD; padding:3px 0;'>Elevated threshold</td>
                      <td style='color:#F5A623; text-align:right;'>≤ {ref['elevated']} {ref['unit']}</td></tr>
                  <tr><td style='color:#7A8BAD; padding:3px 0;'>Patient value</td>
                      <td style='color:{sc}; text-align:right; font-weight:700;'>{val} {ref['unit']}</td></tr>
                  <tr><td style='color:#7A8BAD; padding:3px 0;'>Status</td>
                      <td style='color:{sc}; text-align:right; font-weight:700;'>{status_word}</td></tr>
                </table>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — RISK FACTORS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
with tab_factors:
    fc1, fc2 = st.columns([1, 1], gap="large")

    with fc1:
        st.markdown("<div class='section-title'>Identified Risk Factors</div>", unsafe_allow_html=True)

        risk_flags = []
        if glucose > 125:
            risk_flags.append(("🔴", "Diabetic Glucose Range", f"Glucose {glucose} mg/dL exceeds diagnostic threshold of 126 mg/dL. Fasting plasma glucose or HbA1c confirmatory testing is essential."))
        elif glucose > 99:
            risk_flags.append(("🟡", "Pre-Diabetic Glucose", f"Glucose {glucose} mg/dL falls in pre-diabetic range (100–125 mg/dL). Lifestyle intervention can prevent progression."))

        if bmi >= 30:
            risk_flags.append(("🔴", "Obese BMI", f"BMI {bmi:.1f} kg/m² classifies as obese. Each 1-unit BMI reduction cuts diabetes risk by ~16%."))
        elif bmi >= 25:
            risk_flags.append(("🟡", "Overweight BMI", f"BMI {bmi:.1f} kg/m² is overweight. Weight reduction of 5–7% body weight halves diabetes progression."))

        if bp >= 90:
            risk_flags.append(("🔴", "Hypertension", f"Blood pressure {bp} mmHg indicates Stage 2 hypertension, compounding cardiometabolic risk."))
        elif bp >= 80:
            risk_flags.append(("🟡", "Elevated Blood Pressure", f"Blood pressure {bp} mmHg (Stage 1 hypertension). Dietary sodium reduction and exercise recommended."))

        if dpf >= 1.0:
            risk_flags.append(("🔴", "Strong Family History", f"Pedigree score {dpf:.2f} indicates high hereditary predisposition. First-degree relatives with T2DM triple personal risk."))
        elif dpf >= 0.5:
            risk_flags.append(("🟡", "Moderate Family Risk", f"Pedigree score {dpf:.2f} reflects moderate genetic burden. Genetic counselling may be beneficial."))

        if age >= 55:
            risk_flags.append(("🔴", "Advanced Age", f"Age {age} years — beta-cell decline and insulin resistance accumulate progressively after 45."))
        elif age >= 40:
            risk_flags.append(("🟡", "Age-Related Risk", f"Age {age} years places the patient in the higher-risk screening cohort (40–54)."))

        if insulin > 300:
            risk_flags.append(("🔴", "Hyperinsulinaemia", f"Insulin {insulin} μU/mL — severely elevated, indicating significant insulin resistance."))
        elif insulin > 166:
            risk_flags.append(("🟡", "Elevated Insulin", f"Insulin {insulin} μU/mL above normal, suggesting early compensatory insulin resistance."))

        if pregnancies >= 4:
            risk_flags.append(("🟡", "Gestational History", f"{pregnancies} pregnancies — each prior GDM episode carries ~50% lifetime T2DM conversion risk."))

        if not risk_flags:
            risk_flags.append(("✅", "No Major Risk Flags", "All primary indicators are within normal clinical ranges. Maintain current lifestyle and schedule annual re-screening."))

        for icon_e, title, desc in risk_flags:
            st.markdown(f"""
            <div class='rec-item'>
                <div class='rec-icon'>{icon_e}</div>
                <div class='rec-text'>
                    <div class='rec-title'>{title}</div>
                    {desc}
                </div>
            </div>""", unsafe_allow_html=True)

    with fc2:
        st.markdown("<div class='section-title'>Clinical Recommendations</div>", unsafe_allow_html=True)

        # Dynamic recommendations
        recs = []
        if glucose > 125:
            recs.append(("🧪", "Urgent: HbA1c Testing", "Order glycated haemoglobin (HbA1c) test immediately. Result ≥6.5% confirms diagnosis of Type 2 Diabetes Mellitus."))
        if glucose > 99:
            recs.append(("📋", "Oral Glucose Tolerance Test", "Schedule 75g OGTT to characterise glucose disposal and confirm pre-diabetic status."))
        if bmi >= 30:
            recs.append(("🏃", "Structured Weight Loss Programme", "Enrol in medically-supervised programme targeting ≥7% body weight reduction over 12 months via caloric restriction and 150 min/week moderate aerobic activity."))
        if bp >= 80:
            recs.append(("💊", "Hypertension Management", "Evaluate need for ACE inhibitor or ARB therapy. DASH diet (sodium <2.3g/day) and BP monitoring twice weekly."))
        if dpf >= 0.5 or age >= 40:
            recs.append(("📅", "Bi-Annual Screening", "Given family history and/or age, schedule fasting glucose and HbA1c every 6 months rather than annually."))
        if insulin > 166:
            recs.append(("🥗", "Low Glycaemic Index Diet", "Prescribe Mediterranean-style diet rich in fibre, lean protein, and healthy fats to reduce postprandial insulin demand."))
        if risk_pct >= 65:
            recs.append(("⚕️", "Endocrinologist Referral", "Risk profile warrants specialist evaluation. Consider metformin prophylaxis discussion as per ADA guidelines for high-risk pre-diabetics."))
        if not recs:
            recs.append(("✅", "Maintain Preventive Lifestyle", "Continue balanced diet, ≥150 min/week physical activity, and annual metabolic panel screening."))
            recs.append(("💧", "Hydration & Sleep Hygiene", "Adequate hydration (>2L/day) and 7–9 hours of sleep demonstrably reduce insulin resistance over time."))

        for icon_e, title, desc in recs:
            st.markdown(f"""
            <div class='rec-item'>
                <div class='rec-icon'>{icon_e}</div>
                <div class='rec-text'>
                    <div class='rec-title'>{title}</div>
                    {desc}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Risk trajectory line chart ──
        st.markdown("<div class='section-title' style='margin-top:18px;'>10-Year Risk Trajectory (Modelled)</div>", unsafe_allow_html=True)
        years = list(range(0, 11))
        base  = risk_score
        # Simulate: baseline trajectory vs. with intervention
        traj_base = [min(base + year * 0.025 * (1 + dpf), 0.98) for year in years]
        traj_int  = [max(base - year * 0.018 * (1 + (bmi - 25) / 50), 0.02) for year in years]

        fig_t, ax_t = dark_fig(5.2, 2.8)
        ax_t.plot(years, [v * 100 for v in traj_base], color="#F05252", linewidth=2,
                  label="No Intervention", marker="o", markersize=4, markerfacecolor="#F05252")
        ax_t.plot(years, [v * 100 for v in traj_int],  color="#0FCFB0", linewidth=2,
                  label="With Intervention", marker="o", markersize=4, markerfacecolor="#0FCFB0")
        ax_t.fill_between(years, [v*100 for v in traj_int],
                           [v*100 for v in traj_base], alpha=0.08, color="#F5A623")
        ax_t.axhline(65, color="#F05252", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_t.axhline(30, color="#0FCFB0", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_t.set_xlabel("Years", color="#7A8BAD", fontsize=9)
        ax_t.set_ylabel("Risk (%)", color="#7A8BAD", fontsize=9)
        ax_t.set_ylim(0, 100)
        ax_t.legend(fontsize=8.5, facecolor="#1A2535", edgecolor="#243048", labelcolor="#E8EDF7")
        for sp in ["top", "right"]: ax_t.spines[sp].set_visible(False)
        fig_t.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

# ════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL REPORT
# ════════════════════════════════════════════════════════════
with tab_clinical:
    st.markdown("<div class='section-title'>Structured Clinical Assessment Report</div>", unsafe_allow_html=True)

    icd_code = "E11.9" if risk_pct >= 65 else ("R73.09" if risk_pct >= 30 else "Z13.1")
    icd_desc = {
        "E11.9":  "Type 2 Diabetes Mellitus, without complications",
        "R73.09": "Pre-Diabetes / Impaired Glucose Tolerance",
        "Z13.1":  "Screening for Diabetes Mellitus — Low Risk",
    }

    st.markdown(f"""
    <div style='background:#1A2535; border:1px solid #243048; border-radius:14px; padding:28px 32px; font-size:13px; line-height:2;'>

    <div style='font-size:16px; font-weight:700; color:#E8EDF7; margin-bottom:4px; letter-spacing:0.02em;'>
        DIABETES RISK ASSESSMENT REPORT
    </div>
    <div style='font-size:10px; letter-spacing:0.12em; color:#7A8BAD; margin-bottom:22px; text-transform:uppercase;'>
        Generated by DiabetIQ Clinical Intelligence System
    </div>

    <table style='width:100%; border-collapse:collapse; font-size:12.5px;'>
        <tr>
            <td style='color:#7A8BAD; width:38%; padding:5px 0;'>ICD-10 Code</td>
            <td style='color:#E8EDF7; font-weight:600;'>{icd_code} — {icd_desc[icd_code]}</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Composite Risk Score</td>
            <td style='color:{tier_color}; font-weight:700; font-size:15px;'>{risk_pct:.1f}% — {tier}</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Plasma Glucose</td>
            <td style='color:{status_color("Glucose", glucose)};'>{glucose} mg/dL</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Body Mass Index</td>
            <td style='color:{status_color("BMI", bmi)};'>{bmi} kg/m²</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Diastolic BP</td>
            <td style='color:{status_color("BloodPressure", bp)};'>{bp} mmHg</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Serum Insulin</td>
            <td style='color:{status_color("Insulin", insulin)};'>{insulin} μU/mL</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Diabetes Pedigree Function</td>
            <td style='color:{status_color("DiabetesPedigreeFunction", dpf)};'>{dpf:.2f}</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Age</td>
            <td style='color:#E8EDF7;'>{age} years</td>
        </tr>
        <tr>
            <td style='color:#7A8BAD; padding:5px 0;'>Pregnancies</td>
            <td style='color:#E8EDF7;'>{pregnancies}</td>
        </tr>
    </table>

    <div style='border-top:1px solid #243048; margin:20px 0;'></div>

    <div style='color:#7A8BAD; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>Clinical Impression</div>
    <div style='color:#E8EDF7; font-size:13px; line-height:1.8;'>
    {"The patient presents with <strong style='color:#0FCFB0;'>favourable metabolic indicators</strong>. All primary biomarkers fall within normal reference ranges. Preventive lifestyle maintenance and annual metabolic screening are the primary recommendations." if risk_pct < 30 else
     f"The patient exhibits <strong style='color:#F5A623;'>multiple pre-diabetic risk factors</strong>, most notably {'hyperglycaemia' if glucose > 125 else 'impaired fasting glucose' if glucose > 99 else 'elevated BMI'} and a composite risk score of {risk_pct:.1f}%. Structured lifestyle intervention and follow-up testing within 3 months are indicated." if risk_pct < 65 else
     f"The patient presents with a <strong style='color:#F05252;'>high-risk metabolic profile</strong> (composite score {risk_pct:.1f}%). Findings are consistent with probable Type 2 Diabetes Mellitus or advanced pre-diabetic state. Immediate specialist referral, confirmatory HbA1c/OGTT testing, and initiation of pharmacological risk reduction should be considered."}
    </div>

    <div style='border-top:1px solid #243048; margin:20px 0;'></div>
    <div style='color:#7A8BAD; font-size:10px; line-height:1.6;'>
    ⚕️ DISCLAIMER: This report is generated by an AI clinical decision-support tool. It is intended to assist, not replace, the clinical judgement of a licensed healthcare professional. All findings must be confirmed by appropriate laboratory investigations and physical examination.
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Printable data table ──
    st.markdown("<div class='section-title' style='margin-top:20px;'>Raw Input Data</div>", unsafe_allow_html=True)
    display_df = input_df.T.rename(columns={0: "Patient Value"})
    display_df["Normal Threshold"] = [CLINICAL_REF[k]["normal"] for k in display_df.index]
    display_df["Status"] = [
        "✅ Normal" if float(input_df[k].values[0]) <= CLINICAL_REF[k]["normal"]
        else "⚠️ Elevated" if float(input_df[k].values[0]) <= CLINICAL_REF[k]["elevated"]
        else "🔴 High"
        for k in display_df.index
    ]
    st.dataframe(display_df, use_container_width=True)
