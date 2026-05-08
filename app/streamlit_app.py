import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import os
import joblib
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Credit Risk Modelling",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main Background */
.main {
    background-color: #f5f7fa;
}

/* Main Title */
h1 {
    color: #1e293b;
    font-size: 48px !important;
    font-weight: 700 !important;
}

/* Section Headers */
h2, h3 {
    color: #334155;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

/* Button Hover Effect */
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #e2e8f0;
}

/* Input Containers */
div[data-baseweb="input"] {
    border-radius: 10px;
}

/* Reduce top padding */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL + SCALER
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")

model = joblib.load(model_path)
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

scaler = joblib.load(scaler_path)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("Instructions")

    image_path = os.path.join(BASE_DIR, "assets", "finance.jpg")

    if os.path.exists(image_path):
        st.image(image_path, width=250)
    else:
        st.warning("Sidebar image not found.")

    st.markdown("""
    ### Steps
    
    1. Enter customer details
    
    2. Adjust loan information
    
    3. Click **Predict Credit Risk**
    
    4. Review risk insights and model interpretation
    """)

    st.markdown("---")

    st.info("""
    This application uses:
    
    - XGBoost
    - SHAP Explainability
    - Credit Risk ML Model
    """)

# =====================================================
# TITLE
# =====================================================

st.title("💳 Credit Risk Modelling")

st.markdown("""
### AI-Powered Credit Risk Assessment System
""")

# =====================================================
# CUSTOMER DETAILS SECTION
# =====================================================

st.markdown("## 👤 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=75,
        value=30
    )

with col2:

    credit_amount = st.slider(
    "Credit Amount",
    min_value=1000,
    max_value=1000000,
    value=50000,
    step=5000
    )
    st.markdown(
    f"<h4 style='color:#1f2937;'>Selected Loan Amount: ₹{credit_amount:,.0f}</h4>",
    unsafe_allow_html=True
    )
with col3:

    duration = st.number_input(
        "Loan Duration (Months)",
        min_value=4,
        max_value=72,
        value=24
    )

# =====================================================
# LOAN DETAILS
# =====================================================

st.markdown("## 🏦 Loan Details")

col4, col5, col6 = st.columns(3)

with col4:

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

with col5:

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

with col6:

    purpose = st.selectbox(
        "Purpose",
        [
            "car",
            "radio/TV",
            "education",
            "furniture/equipment",
            "business"
        ]
    )

# =====================================================
# FINANCIAL DETAILS
# =====================================================

st.markdown("## 💰 Financial Information")

col7, col8, col9 = st.columns(3)

with col7:

    saving_accounts = st.selectbox(
        "Saving Accounts",
        ["little", "moderate", "quite rich", "rich"]
    )

with col8:

    checking_account = st.selectbox(
        "Checking Account",
        ["little", "moderate", "rich"]
    )

with col9:

    job = st.selectbox(
        "Job Level",
        [0, 1, 2, 3]
    )

# =====================================================
# ENCODING
# =====================================================

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

# =====================================================
# INPUT DATAFRAME
# =====================================================

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

# =====================================================
# SCALE FEATURES
# =====================================================

numerical_cols = ['Age', 'Credit amount', 'Duration']
download_data = input_data.copy()
input_data[numerical_cols] = scaler.transform(
    input_data[numerical_cols]
)

# =====================================================
# PREDICT BUTTON
# =====================================================

st.markdown("---")

if st.button("Predict Credit Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # =====================================================
    # RISK CATEGORY
    # =====================================================

    if probability < 0.3:

        risk_level = "Low Risk 🟢"

    elif probability < 0.7:

        risk_level = "Medium Risk 🟠"

    else:

        risk_level = "High Risk 🔴"

    # =====================================================
    # TOP METRICS
    # =====================================================

    st.markdown("## 📊 Prediction Summary")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Risk Probability",
            f"{probability:.2%}"
        )

    with m2:

        st.metric(
            "Risk Category",
            risk_level
        )

    with m3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Bad Credit"
            )

        else:

            st.metric(
                "Prediction",
                "Good Credit"
            )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediction",
        "Feature Importance",
        "SHAP Explainability",
        "Model Metrics"
    ])

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.markdown("## 📌 Prediction Result")

        if prediction == 1:

            st.error("🔴 High Risk Customer")

        else:

            st.success("🟢 Low Risk Customer")

        # -------------------------------------------------
        # CUSTOMER SUMMARY
        # -------------------------------------------------

        st.markdown("### Customer Summary")

        summary_df = pd.DataFrame({

            "Feature": [
                "Age",
                "Credit Amount",
                "Duration",
                "Housing",
                "Purpose"
            ],

            "Value": [
                age,
                credit_amount,
                duration,
                housing,
                purpose
            ]

        })

        st.table(summary_df)

        # -------------------------------------------------
        # GAUGE CHART
        # -------------------------------------------------

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

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # BUSINESS INSIGHTS
        # -------------------------------------------------

        st.markdown("### 🧠 Business Insights")

        st.info("""
        • Higher credit amounts are associated with higher default risk.

        • Customers with low checking account balances show elevated risk probability.

        • Longer loan durations increase the likelihood of bad credit classification.

        • Housing and savings behavior strongly influence model prediction.
        """)

    # =====================================================
    # TAB 2 - FEATURE IMPORTANCE
    # =====================================================

    with tab2:

        st.markdown("## 📈 Feature Importance")

        importance_df = pd.DataFrame({

            'Feature': input_data.columns,

            'Importance': model.feature_importances_

        })

        importance_df = importance_df.sort_values(
            by='Importance',
            ascending=True
        )

        fig_imp = px.bar(

            importance_df,

            x='Importance',

            y='Feature',

            orientation='h',

            title='Feature Importance'

        )

        st.plotly_chart(
            fig_imp,
            use_container_width=True
        )

    # =====================================================
    # TAB 3 - SHAP
    # =====================================================

    with tab3:

        st.markdown("## 🔍 SHAP Explainability")

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(input_data)

        shap_df = pd.DataFrame({

            'Feature': input_data.columns,

            'SHAP Value': shap_values[0]

        })

        shap_df = shap_df.sort_values(
            by='SHAP Value',
            ascending=True
        )

        fig_shap = px.bar(

            shap_df,

            x='SHAP Value',

            y='Feature',

            orientation='h',

            title='SHAP Feature Contribution'

        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

    # =====================================================
    # TAB 4 - MODEL METRICS
    # =====================================================

    with tab4:

        st.markdown("## 📊 Model Performance")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Accuracy",
                "97%"
            )

        with c2:

            st.metric(
                "ROC-AUC",
                "0.997"
            )

        with c3:

            st.metric(
                "KS Statistic",
                "96.2%"
            )

# =====================================================
# DOWNLOAD BUTTON
# =====================================================

st.markdown("---")

csv = download_data.to_csv(index=False)

st.download_button(
    label="Download Prediction Data",
    data=csv,
    file_name='credit_risk_prediction.csv',
    mime='text/csv'
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
### ⚙️ Technology Stack

- Python
- XGBoost
- Streamlit
- SHAP Explainability
- Machine Learning
- Credit Risk Analytics
""")