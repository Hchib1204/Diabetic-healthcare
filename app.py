import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- LOAD MODEL ---
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("⚠️ Error: Model files not found! Please upload 'diabetes_model.pkl' and 'scaler.pkl'.")
    st.stop()

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Diabetes Risk AI", page_icon="🩺", layout="wide")

# Custom CSS for the Report Download Button and styling
st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; color: #2E86C1; }
    .stButton>button { background-color: #2E86C1; color: white; border-radius: 10px; width: 100%; }
    .report-text { font-family: 'Courier New', Courier, monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: INFO & VITALS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
st.sidebar.header("🏥 Clinic Details")
doctor_name = st.sidebar.text_input("Doctor's Name", "Dr. AI Specialist")
patient_name = st.sidebar.text_input("Patient's Name", "John Doe")

st.sidebar.divider()
st.sidebar.header("Patient Vitals")

# --- HELPER FUNCTIONS ---
def categorize_bmi(bmi):
    if bmi < 25: return 0
    elif bmi < 30: return 1
    else: return 2

def user_input_features():
    # 1. Capture Raw Inputs
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 117)
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    skin = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree', 0.0, 2.4, 0.37)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    # 2. Initial DataFrame
    data = {
        'Pregnancies': [pregnancies], 'Glucose': [glucose], 'BloodPressure': [bp],
        'SkinThickness': [skin], 'Insulin': [insulin], 'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf], 'Age': [age]
    }
    features = pd.DataFrame(data)

    # 3. Feature Engineering (Must match training logic)
    if glucose == 0:
        features['Insulin_Glucose_Ratio'] = 0.0
    else:
        features['Insulin_Glucose_Ratio'] = insulin / glucose
        
    features['BMI_Class'] = categorize_bmi(bmi)
    features['Age_Glucose'] = age * glucose
    
    return features

input_df = user_input_features()

# --- MAIN PAGE STRUCTURE ---
st.markdown(f'<p class="big-font">🩺 Intelligent Diabetes Intervention System</p>', unsafe_allow_html=True)
st.markdown(f"**Welcome, {doctor_name}**. analyzing patient: **{patient_name}**")

# Create 3 Tabs
tab1, tab2, tab3 = st.tabs(["📋 Patient Details", "🔍 Risk Analysis", "📊 Visual Insights"])

# --- TAB 1: PATIENT DETAILS ---
with tab1:
    st.subheader("Patient Vitals Overview")
    st.dataframe(input_df.style.format("{:.2f}"))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("BMI", f"{input_df['BMI'][0]}", delta_color="inverse")
    col2.metric("Glucose", f"{input_df['Glucose'][0]} mg/dL")
    col3.metric("Blood Pressure", f"{input_df['BloodPressure'][0]} mm Hg")
    
    st.info("Ensure all vitals are entered correctly in the sidebar before proceeding to analysis.")

# --- RUN PREDICTION (Global for Tabs 2 & 3) ---
try:
    input_scaled = scaler.transform(input_df)
    prediction_prob = model.predict_proba(input_scaled)
    risk_score = prediction_prob[0][1]
except Exception as e:
    st.error(f"Error calculating risk: {e}")
    risk_score = 0

# --- TAB 2: RISK ANALYSIS & REPORT ---
with tab2:
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Risk Score")
        # Gauge Chart using Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        st.subheader("AI Assessment")
        if risk_score < 0.3:
            st.success(f"✅ Low Risk: {risk_score:.1%}")
            advice = "Patient is in a healthy range. Maintain current lifestyle."
        elif risk_score < 0.7:
            st.warning(f"⚠️ Moderate Risk: {risk_score:.1%}")
            advice = "Pre-diabetic indicators detected. Recommended: Diet adjustment & exercise."
        else:
            st.error(f"🚨 High Risk: {risk_score:.1%}")
            advice = "Strong diabetic indicators. Immediate clinical consultation required."
        
        st.write(f"**Recommendation:** {advice}")

        # Specific Flags
        if input_df['Glucose'].values[0] > 140:
             st.write("• 🩸 **High Glucose:** Consider HbA1c test.")
        if input_df['BMI'].values[0] > 30:
             st.write("• ⚖️ **Obesity:** Lowering BMI is the most effective intervention.")

    # --- REPORT GENERATION ---
    st.divider()
    st.subheader("📄 Generate Report")
    
    report_text = f"""
    MEDICAL ANALYSIS REPORT
    -----------------------------------
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    Doctor: {doctor_name}
    Patient: {patient_name}
    -----------------------------------
    VITALS:
    - Pregnancies: {input_df['Pregnancies'][0]}
    - Glucose: {input_df['Glucose'][0]}
    - Blood Pressure: {input_df['BloodPressure'][0]}
    - Skin Thickness: {input_df['SkinThickness'][0]}
    - Insulin: {input_df['Insulin'][0]}
    - BMI: {input_df['BMI'][0]}
    - Diabetes Pedigree: {input_df['DiabetesPedigreeFunction'][0]}
    - Age: {input_df['Age'][0]}
    -----------------------------------
    ANALYSIS RESULTS:
    Risk Probability: {risk_score:.1%}
    Risk Category: {'High' if risk_score > 0.7 else 'Moderate' if risk_score > 0.3 else 'Low'}
    
    AI RECOMMENDATION:
    {advice}
    -----------------------------------
    Generated by Diabetes Risk AI System
    """
    
    st.download_button(
        label="📥 Download Patient Report (TXT)",
        data=report_text,
        file_name=f"Report_{patient_name}_{datetime.now().date()}.txt",
        mime="text/plain"
    )

# --- TAB 3: VISUAL INSIGHTS ---
with tab3:
    st.subheader("Interactive Vitals Comparison")
    
    # 1. Bar Chart: Patient vs Healthy Average
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.caption("Patient vs. Healthy Benchmarks")
        metrics = ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness']
        # Healthy averages (approximate)
        healthy_vals = [100, 72, 22, 20]
        patient_vals = [input_df[m][0] for m in metrics]
        
        df_vis = pd.DataFrame({
            'Metric': metrics * 2,
            'Value': healthy_vals + patient_vals,
            'Type': ['Healthy Avg']*4 + ['Patient']*4
        })
        
        fig_bar = px.bar(df_vis, x='Metric', y='Value', color='Type', barmode='group',
                         color_discrete_map={'Healthy Avg': '#2ecc71', 'Patient': '#3498db'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # 2. Radar Chart: The "Health Footprint"
    with col_vis2:
        st.caption("Multivariate Health Profile (Radar Chart)")
        
        # Normalize data for radar chart (just for visualization)
        categories = ['Glucose', 'BP', 'BMI', 'Age', 'Insulin']
        # Max values for normalization
        max_vals = [200, 122, 67, 81, 300]
        
        values_patient = [input_df['Glucose'][0], input_df['BloodPressure'][0], input_df['BMI'][0], input_df['Age'][0], input_df['Insulin'][0]]
        values_norm = [v/m for v, m in zip(values_patient, max_vals)]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=values_norm,
          theta=categories,
          fill='toself',
          name=patient_name
        ))
        
        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
          showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Scatter Plot Context
    st.subheader("Risk Context: Glucose vs BMI")
    st.caption("See where the patient stands in the risk matrix.")
    
    # Create a synthetic background dataset for context
    # (In a real app, this would be your training data)
    synth_data = pd.DataFrame({
        'Glucose': np.random.randint(50, 200, 100),
        'BMI': np.random.uniform(15, 50, 100),
        'Risk': np.random.choice(['Low', 'High'], 100)
    })
    
    fig_scatter = px.scatter(synth_data, x='Glucose', y='BMI', color='Risk', 
                             color_discrete_map={'Low': 'green', 'High': 'red'}, opacity=0.3)
    
    # Add the current patient as a big star
    fig_scatter.add_trace(go.Scatter(
        x=[input_df['Glucose'][0]], 
        y=[input_df['BMI'][0]],
        mode='markers',
        marker=dict(size=20, symbol='star', color='blue'),
        name='Current Patient'
    ))
    
    st.plotly_chart(fig_scatter, use_container_width=True)
 
