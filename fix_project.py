# fix_project.py
import os

# This contains the CORRECT app code with the 11 necessary features
correct_app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD MODEL ---
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("CRITICAL ERROR: Model files not found.")
    st.stop()

# --- CONFIG ---
st.set_page_config(page_title="Diabetes Risk AI", page_icon="🩺", layout="wide")

st.markdown(\"\"\"
    <style>
    .big-font { font-size:30px !important; font-weight: bold; color: #2E86C1; }
    .stButton>button { background-color: #2E86C1; color: white; border-radius: 10px; }
    </style>
    \"\"\", unsafe_allow_html=True)

st.markdown('<p class="big-font">🩺 Intelligent Diabetes Intervention System</p>', unsafe_allow_html=True)
st.divider()

# --- SIDEBAR ---
st.sidebar.header("Patient Vitals")

def categorize_bmi(bmi):
    if bmi < 25: return 0
    elif bmi < 30: return 1
    else: return 2

def user_input_features():
    # 1. RAW INPUTS
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.37)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    # 2. CREATE DATAFRAME
    features = pd.DataFrame({
        'Pregnancies': [pregnancies], 
        'Glucose': [glucose], 
        'BloodPressure': [bp],
        'SkinThickness': [skin], 
        'Insulin': [insulin], 
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf], 
        'Age': [age]
    })
    
    # 3. *** FEATURE ENGINEERING (THE FIX) ***
    # This matches the training data exactly
    
    # Feature: Insulin_Glucose_Ratio
    if glucose == 0:
        features['Insulin_Glucose_Ratio'] = 0.0
    else:
        features['Insulin_Glucose_Ratio'] = insulin / glucose
        
    # Feature: BMI_Class
    features['BMI_Class'] = categorize_bmi(bmi)
    
    # Feature: Age_Glucose
    features['Age_Glucose'] = age * glucose
    
    return features

input_df = user_input_features()

# --- MAIN PAGE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Data")
    st.dataframe(input_df)

    if st.button('Analyze Risk Profile'):
        try:
            # TRANSFORM
            input_scaled = scaler.transform(input_df)
            
            # PREDICT
            prediction_prob = model.predict_proba(input_scaled)
            risk_score = prediction_prob[0][1]
            
            # OUTPUT
            st.subheader("Results")
            if risk_score < 0.3:
                st.success(f"Low Risk: {risk_score:.1%}")
            elif risk_score < 0.7:
                st.warning(f"Moderate Risk: {risk_score:.1%}")
            else:
                st.error(f"High Risk: {risk_score:.1%}")
                
            st.divider()
            if input_df['Glucose'].values[0] > 140:
                st.info("• Glucose is high. Consider HbA1c test.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with col2:
    st.subheader("Visuals")
    fig, ax = plt.subplots()
    ax.bar(['Glu', 'BP', 'BMI'], [100, 70, 22], color='green', alpha=0.5)
    ax.bar(['Glu', 'BP', 'BMI'], [input_df['Glucose'][0], input_df['BloodPressure'][0], input_df['BMI'][0]], color='blue', alpha=0.5)
    st.pyplot(fig)
"""

# Write the clean code to app.py
with open("app.py", "w") as f:
    f.write(correct_app_code)

print("✅ SUCCESS! app.py has been force-updated.")
print("You can now run 'streamlit run app.py'")