from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    # Convert JSON to DataFrame
    input_data = pd.DataFrame([data])

    # Apply encoders
    for column, encoder in encoders.items():
        if column in input_data.columns:
            input_data[column] = encoder.transform(input_data[column])

    # Apply scaler
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0].max()

    return {
        "prediction": int(prediction),
        "probability": float(probability) if probability is not None else None
    }