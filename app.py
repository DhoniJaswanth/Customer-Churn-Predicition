import streamlit as st
import pandas as pd
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to churn."
)


# ============================================================
# LOAD PICKLE FILES
# ============================================================

@st.cache_resource
def load_files():

    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, encoders, scaler


try:
    model, encoders, scaler = load_files()

except Exception as e:
    st.error("❌ Could not load model files.")
    st.exception(e)
    st.stop()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.header("Customer Information")

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

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

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=100,
        value=12
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# OTHER CUSTOMER INFORMATION
# ============================================================

st.subheader("Subscription Information")

col4, col5 = st.columns(2)

with col4:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
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


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # CREATE ORIGINAL 19 FEATURE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior_citizen],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [phone_service],

            "MultipleLines": [multiple_lines],

            "InternetService": [internet_service],

            "OnlineSecurity": [online_security],

            "OnlineBackup": [online_backup],

            "DeviceProtection": [device_protection],

            "TechSupport": [tech_support],

            "StreamingTV": [streaming_tv],

            "StreamingMovies": [streaming_movies],

            "Contract": [contract],

            "PaperlessBilling": [paperless_billing],

            "PaymentMethod": [payment_method],

            "MonthlyCharges": [monthly_charges],

            "TotalCharges": [total_charges]

        })


        # ----------------------------------------------------
        # ENCODE CATEGORICAL VARIABLES
        # ----------------------------------------------------

        for column, encoder in encoders.items():

            # Churn is the target, not an input
            if column == "Churn":
                continue

            if column in input_data.columns:

                input_data[column] = encoder.transform(
                    input_data[column].astype(str)
                )


        # ----------------------------------------------------
        # SCALE ONLY NUMERICAL FEATURES
        # ----------------------------------------------------

        numerical_features = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]


        input_data[numerical_features] = scaler.transform(
            input_data[numerical_features]
        )


        # ----------------------------------------------------
        # CHECK MODEL FEATURES
        # ----------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

            input_data = input_data[
                expected_features
            ]

        else:

            if input_data.shape[1] != model.n_features_in_:

                st.error(
                    f"Model expects {model.n_features_in_} "
                    f"features, but received "
                    f"{input_data.shape[1]}."
                )

                st.stop()


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_data
            )[0][1]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn."
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn."
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if probability is not None:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


        # ----------------------------------------------------
        # SHOW DATA USED FOR PREDICTION
        # ----------------------------------------------------

        with st.expander(
            "🔍 View Processed Input"
        ):

            st.dataframe(input_data)


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)