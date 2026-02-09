import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
from src.infer import predict_proba_one

st.title("Telco Churn Predictor")

with st.form("inputs"):
    internet_service = st.selectbox("Internet Service", ["Yes", "No"])
    internet_type = st.selectbox("Internet Type", ["Fiber Optic", "Cable", "DSL", "None"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection Plan", ["Yes", "No"])
    premium_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])

    avg_gb = st.number_input("Avg Monthly GB Download", min_value=0, value=20, step=1)
    total_ld = st.number_input("Total Long Distance Charges", min_value=0.0, value=30.0, step=1.0)
    age = st.number_input("Age", min_value=18, value=35, step=1)

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    features = {
        "Internet Service": internet_service,
        "Internet Type": internet_type,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Online Security": online_security,
        "Device Protection Plan": device_protection,
        "Premium Tech Support": premium_support,
        "Contract": contract,
        "Avg Monthly GB Download": int(avg_gb),
        "Total Long Distance Charges": float(total_ld),
        "Age": int(age),
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
    }

    out = predict_proba_one(features)
    st.metric("Churn probability", f"{out['churn_probability']:.2%}")
    st.caption(f"Loaded MLflow run: {out['run_id']}")
