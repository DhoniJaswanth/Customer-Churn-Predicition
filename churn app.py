import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------

# Page configuration

# ---------------------------------------------------

st.set_page_config(
page_title="Customer Churn Prediction",
page_icon="📊",
layout="wide"
)

# ---------------------------------------------------

# Load saved model, encoders and scaler

# ---------------------------------------------------

@st.cache_resource
def load_artifacts():


with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

return model, encoders, scaler


model, encoders, scaler = load_artifacts()

# ---------------------------------------------------

# Prediction function

# ---------------------------------------------------

def make_prediction(input_data):


input_df = pd.DataFrame([input_data])

# Encode categorical columns
for col, encoder in encoders.items():

    if col in input_df.columns:
        try:
            input_df[col] = encoder.transform(
                input_df[col].astype(str)
            )
        except ValueError as e:
            st.error(f"Encoding error in {col}: {e}")
            return None, None

# Scale numerical columns
numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

input_df[numerical_cols] = scaler.transform(
    input_df[numerical_cols]
)

# Prediction
prediction = model.predict(input_df)[0]

probability = model.predict_proba(input_df)[0][1]

if prediction == 1:
    result = "Churn"
else:
    result = "No Churn"

return result, probability


# ---------------------------------------------------

# Header

# ---------------------------------------------------

st.title("📊 Customer Churn Prediction")
st.write(
"Enter customer information below to predict whether "
"the customer is likely to churn."
)

st.divider()

# ---------------------------------------------------

# Customer Information

# ---------------------------------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:


gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


with col2:


tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=1
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

with col3:


internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


# ---------------------------------------------------

# Services

# ---------------------------------------------------

st.subheader("🌐 Internet & Services")

col1, col2, col3 = st.columns(3)

with col1:

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)


with col2:


device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)


with col3:


streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# ---------------------------------------------------

# Billing Information

# ---------------------------------------------------

st.subheader("💰 Billing Information")

col1, col2 = st.columns(2)

with col1:

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=29.85,
    step=0.01
)


with col2:


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=29.85,
    step=0.01
)


payment_method = st.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

# ---------------------------------------------------

# Prediction Button

# ---------------------------------------------------

st.divider()

if st.button(
"🔮 Predict Customer Churn",
use_container_width=True
):

example_input = {

    "gender": gender,

    "SeniorCitizen": senior_citizen,

    "Partner": partner,

    "Dependents": dependents,

    "tenure": tenure,

    "PhoneService": phone_service,

    "MultipleLines": multiple_lines,

    "InternetService": internet_service,

    "OnlineSecurity": online_security,

    "OnlineBackup": online_backup,

    "DeviceProtection": device_protection,

    "TechSupport": tech_support,

    "StreamingTV": streaming_tv,

    "StreamingMovies": streaming_movies,

    "Contract": contract,

    "PaperlessBilling": paperless_billing,

    "PaymentMethod": payment_method,

    "MonthlyCharges": monthly_charges,

    "TotalCharges": total_charges
}

prediction, probability = make_prediction(
    example_input
)

if prediction is not None:

    st.subheader("Prediction Result")

    if prediction == "Churn":

        st.error(
            f"⚠️ Customer is likely to CHURN"
        )

    else:

        st.success(
            f"✅ Customer is NOT likely to CHURN"
        )

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.write(
        f"**Prediction:** {prediction}"
    )

