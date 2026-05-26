import streamlit as st
import numpy as np
import joblib

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# =========================================
# LOAD MODEL & SCALER
# =========================================

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================================
# TITLE
# =========================================

st.title("📊 Customer Churn Prediction System")

st.markdown("""
This system predicts whether a telecom customer is likely to:

- ❌ Churn (Leave)
- ✅ Stay

Enter customer details below and click on **Predict Churn**.
""")

# =========================================
# USER INPUTS
# =========================================

st.subheader("📌 Customer Details")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# =========================================
# ENCODING MAPS
# =========================================

binary_map = {
    "No": 0,
    "Yes": 1
}

gender_map = {
    "Female": 0,
    "Male": 1
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("🔍 Predict Churn"):

    # =====================================
    # INPUT DATA (19 FEATURES)
    # =====================================

    input_data = np.array([[

        gender_map[gender],                         # gender
        binary_map[senior_citizen],                # SeniorCitizen
        binary_map[partner],                       # Partner
        binary_map[dependents],                    # Dependents
        tenure,                                    # tenure
        binary_map[phone_service],                 # PhoneService
        binary_map[multiple_lines],                # MultipleLines
        internet_map[internet_service],            # InternetService
        binary_map[online_security],               # OnlineSecurity
        binary_map[online_backup],                 # OnlineBackup
        binary_map[device_protection],             # DeviceProtection
        binary_map[tech_support],                  # TechSupport
        binary_map[streaming_tv],                  # StreamingTV
        binary_map[streaming_movies],              # StreamingMovies
        contract_map[contract],                    # Contract
        binary_map[paperless_billing],             # PaperlessBilling
        payment_map[payment_method],               # PaymentMethod
        monthly_charges,                           # MonthlyCharges
        total_charges                              # TotalCharges

    ]])

    # =====================================
    # SCALING
    # =====================================

    input_data = scaler.transform(input_data)

    # =====================================
    # PREDICTION
    # =====================================

    prediction = model.predict(input_data)

    prediction_probability = model.predict_proba(input_data)

    # =====================================
    # RESULT
    # =====================================

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.error("⚠ Customer is likely to churn.")

        st.write(
            f"Churn Probability: {prediction_probability[0][1] * 100:.2f}%"
        )

    else:

        st.success("✅ Customer is likely to stay.")

        st.write(
            f"Stay Probability: {prediction_probability[0][0] * 100:.2f}%"
        )

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption("Built with Streamlit & Machine Learning 🚀")