import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DiabetIQ", page_icon="🧬", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('diabetes_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s, True
    except:
        return None, None, False

model, scaler, loaded = load_assets()

if not loaded:
    st.error("⚠️ Files missing! Run final_train.py first to generate scaler.pkl and diabetes_model.pkl.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient Vitals")
    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 117)
    bp = st.slider("Blood Pressure (mmHg)", 0, 122, 72)
    skin = st.slider("Skin Thickness (mm)", 0, 99, 23)
    insulin = st.slider("Insulin (μU/mL)", 0, 846, 30)
    bmi = st.slider("BMI (kg/m²)", 0.0, 67.1, 32.0)
    dpf = st.slider("Diabetes Pedigree", 0.0, 2.4, 0.37)
    age = st.slider("Age (years)", 21, 81, 29)
    analyze = st.button("Analyze Risk")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (MUST MATCH TRAINING SCRIPT)
# ─────────────────────────────────────────────────────────────────────────────
def build_model_input(df):
    d = df.copy()
    # 1. Engineering logic
    d["Insulin_Glucose_Ratio"] = d["Insulin"] / d["Glucose"].replace(0, np.nan).fillna(1)
    d["BMI_Class"] = pd.cut(d["BMI"], bins=[0, 18.5, 25.0, 30.0, float("inf")], labels=[0, 1, 2, 3]).astype(float)
    d["Age_Glucose"] = d["Age"] * d["Glucose"]

    # 2. EXACT COLUMN ORDER FOR SCALER
    FIT_COLUMNS = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
        "BMI", "DiabetesPedigreeFunction", "Age", "Insulin_Glucose_Ratio", 
        "BMI_Class", "Age_Glucose"
    ]
    return d[FIT_COLUMNS]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("Clinical Diabetes Risk Intelligence")

input_df = pd.DataFrame({
    "Pregnancies": [pregnancies], "Glucose": [glucose], "BloodPressure": [bp],
    "SkinThickness": [skin], "Insulin": [insulin], "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf], "Age": [age]
})

# Execute Prediction
input_full = build_model_input(input_df)
input_scaled = scaler.transform(input_full) # This will no longer throw an error
risk_prob = model.predict_proba(input_scaled)[0][1]

# Display Result
st.metric("Probability of Diabetes", f"{risk_prob*100:.1f}%")
if risk_prob > 0.65:
    st.error("Status: High Risk")
elif risk_prob > 0.30:
    st.warning("Status: Moderate Risk")
else:
    st.success("Status: Low Risk")