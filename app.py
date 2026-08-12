import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.markdown(
    "Enter the customer information below to predict the likelihood of churn."
)

# ============================================================
# LOAD PICKLE FILES
# ============================================================

@st.cache_resource
def load_pickle_files():

    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, encoders, scaler


try:

    model, encoders, scaler = load_pickle_files()

except Exception as e:

    st.error("❌ Could not load the pickle files.")

    st.exception(e)

    st.stop()


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

with st.expander("🔎 Model Information"):

    st.write("Model:", type(model).__name__)

    if hasattr(model, "n_features_in_"):
        st.write(
            "Number of features expected by model:",
            model.n_features_in_
        )

    if hasattr(scaler, "feature_names_in_"):

        st.write(
            "Features expected by scaler:"
        )

        st.write(
            list(scaler.feature_names_in_)
        )


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.header("Customer Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Categorical inputs
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Numerical inputs
# ------------------------------------------------------------

with col2:

    tenure = st.number_input(
        "Tenure",
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


# ============================================================
# CREATE USER INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "Gender": [gender],

    "SeniorCitizen": [senior_citizen],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]

})


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # COPY INPUT DATA
        # ----------------------------------------------------

        processed_data = input_data.copy()


        # ----------------------------------------------------
        # APPLY SAVED ENCODERS
        # ----------------------------------------------------

        if isinstance(encoders, dict):

            for column, encoder in encoders.items():

                if column in processed_data.columns:

                    try:

                        processed_data[column] = encoder.transform(
                            processed_data[column].astype(str)
                        )

                    except Exception:

                        st.warning(
                            f"Could not encode column: {column}"
                        )


        # ----------------------------------------------------
        # GET FEATURES EXPECTED BY SCALER
        # ----------------------------------------------------

        if hasattr(scaler, "feature_names_in_"):

            required_features = list(
                scaler.feature_names_in_
            )

        else:

            required_features = list(
                processed_data.columns
            )


        # ----------------------------------------------------
        # CHECK REQUIRED FEATURES
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in required_features

            if feature not in processed_data.columns

        ]


        if missing_features:

            st.error(
                "❌ Missing features required by the scaler:"
            )

            st.write(missing_features)

            st.stop()


        # ----------------------------------------------------
        # KEEP ONLY FEATURES USED DURING TRAINING
        # ----------------------------------------------------

        processed_data = processed_data[
            required_features
        ]


        # ----------------------------------------------------
        # APPLY SCALER
        # ----------------------------------------------------

        processed_scaled = scaler.transform(
            processed_data
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            processed_scaled
        )[0]


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                processed_scaled
            )

            probability = probabilities[0][1]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.header("Prediction Result")


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
        # SHOW PROCESSED DATA
        # ----------------------------------------------------

        with st.expander(
            "🔍 View Processed Input"
        ):

            st.dataframe(
                processed_data
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)