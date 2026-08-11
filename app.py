import streamlit as st
import requests

API_URL = "https://customer-churn-predicition.onrender.com"

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, value=12)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=500.0
)

if st.button("Predict Churn"):

    data = {
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed!")

            st.write("Prediction:", result["prediction"])
            st.write("Probability:", result["probability"])

        else:
            st.error(
                f"API Error: {response.status_code}"
            )

    except Exception as e:
        st.error(f"Connection error: {e}")