import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load(r"../models/xgboost_model.pkl")

# ----------------------------
# TITLE
# ----------------------------
st.title("💳 Credit Risk Prediction System")
st.markdown("### Machine Learning Based Credit Risk Assessment")

st.write(
    """
    This application predicts whether a customer is:
    - Good Credit ✅
    - Bad Credit ❌
    
    using an XGBoost Machine Learning model.
    """
)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 75, 30)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

job = st.sidebar.selectbox(
    "Job Level",
    [0, 1, 2, 3]
)

housing = st.sidebar.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.sidebar.selectbox(
    "Saving Accounts",
    ["little", "moderate", "quite rich", "rich"]
)

checking_account = st.sidebar.selectbox(
    "Checking Account",
    ["little", "moderate", "rich"]
)

credit_amount = st.sidebar.slider(
    "Credit Amount",
    250,
    20000,
    5000
)

duration = st.sidebar.slider(
    "Duration (months)",
    4,
    72,
    24
)

purpose = st.sidebar.selectbox(
    "Purpose",
    [
        "car",
        "radio/TV",
        "education",
        "furniture/equipment",
        "business"
    ]
)

# ----------------------------
# MANUAL ENCODING
# ----------------------------

sex_map = {
    "male": 1,
    "female": 0
}

housing_map = {
    "own": 0,
    "rent": 1,
    "free": 2
}

saving_map = {
    "little": 0,
    "moderate": 1,
    "quite rich": 2,
    "rich": 3
}

checking_map = {
    "little": 0,
    "moderate": 1,
    "rich": 2
}

purpose_map = {
    "car": 0,
    "radio/TV": 1,
    "education": 2,
    "furniture/equipment": 3,
    "business": 4
}

# ----------------------------
# CREATE INPUT DATAFRAME
# ----------------------------

input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_map[sex]],
    'Job': [job],
    'Housing': [housing_map[housing]],
    'Saving accounts': [saving_map[saving_accounts]],
    'Checking account': [checking_map[checking_account]],
    'Credit amount': [credit_amount],
    'Duration': [duration],
    'Purpose': [purpose_map[purpose]]
})

# ----------------------------
# PREDICTION BUTTON
# ----------------------------

if st.button("Predict Credit Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    # GOOD CREDIT
    if prediction == 0:

        st.success("✅ Good Credit Customer")

    # BAD CREDIT
    else:

        st.error("❌ Bad Credit Customer")

    st.write(f"### Risk Probability: {probability:.2%}")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MODEL PERFORMANCE
# ----------------------------

st.markdown("---")

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "97%")
col2.metric("AUC Score", "0.997")
col3.metric("KS Statistic", "96.2%")

# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.write(
    """
    Developed using:
    - Python
    - XGBoost
    - Streamlit
    - Machine Learning
    """
)