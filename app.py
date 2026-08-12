import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

# -----------------------------
# Load Model, Encoder and Scaler
# -----------------------------
@st.cache_resource
def load_files():
    with open("best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("encoder.pkl", "rb") as encoder_file:
        encoders = pickle.load(encoder_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, encoders, scaler


try:
    model, encoders, scaler = load_files()
except Exception as e:
    st.error("Unable to load the model files.")
    st.error(str(e))
    st.stop()


# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
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
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
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


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Churn"):

    input_data = pd.DataFrame({
        "Gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    try:
        # Apply saved encoders
        for column, encoder in encoders.items():
            if column in input_data.columns:
                input_data[column] = encoder.transform(
                    input_data[column].astype(str)
                )

        # Apply scaler
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = None

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is unlikely to churn.")

        if probability is not None:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)