# 📊 Customer Churn Prediction

A Machine Learning based **Customer Churn Prediction** web application built using **Python, Scikit-learn, Streamlit, and FastAPI**.

The application predicts whether a customer is likely to **churn (leave the service)** based on customer demographic, service, contract, and billing information.


Live Demo : https://customer-churn-predicition-cpul9eudiishwflzzyspqk.streamlit.app/


The project provides:

* 🎨 Interactive Streamlit web application
* 🤖 Machine Learning prediction model
* ⚡ FastAPI REST API
* 📈 Churn probability
* 🔐 Saved model, encoder, and scaler
* ☁️ Deployment-ready architecture



# ✨ Features

### 👤 Customer Information

The application accepts:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

### 💳 Subscription Information

The application also accepts:

* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

### 🔮 Prediction

After entering the customer information, the application displays:

* ✅ Customer is unlikely to churn
* ⚠️ Customer is likely to churn
* 📊 Churn probability percentage
* 🔍 Processed input used by the model

The Streamlit application also displays the processed features used for prediction.

---

# 🛠️ Technologies Used

| Technology      | Purpose              |
| --------------- | -------------------- |
| Python          | Programming language |
| Pandas          | Data processing      |
| NumPy           | Numerical operations |
| Scikit-learn    | Machine Learning     |
| Pickle          | Model serialization  |
| Streamlit       | Web interface        |
| FastAPI         | REST API             |
| Pydantic        | Request validation   |
| Uvicorn         | FastAPI server       |
| Git             | Version control      |
| GitHub          | Source code hosting  |
| Render          | Backend deployment   |
| Streamlit Cloud | Frontend deployment  |

---

# 🧠 Machine Learning Pipeline

The application uses three saved components:

```text
best_model.pkl
encoder.pkl
scaler.pkl
```

### Model

`best_model.pkl` contains the trained machine learning model.

### Encoder

`encoder.pkl` contains the encoders used to transform categorical features into numerical values.

### Scaler

`scaler.pkl` contains the scaler used for the numerical features:

```text
tenure
MonthlyCharges
TotalCharges
```

The Streamlit application loads these three files before making predictions.

---

# 🔄 Prediction Workflow

```text
Customer Input
      ↓
Create DataFrame
      ↓
Encode Categorical Features
      ↓
Scale Numerical Features
      ↓
Arrange Model Features
      ↓
Machine Learning Model
      ↓
Prediction
      ↓
Churn Probability
      ↓
Display Result
```

---

# 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── fastapi_app.py
│
├── best_model.pkl
├── encoder.pkl
├── scaler.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 💻 Run the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
```

Go into the project folder:

```bash
cd Customer-Churn-Prediction
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
fastapi
uvicorn
pandas
numpy
scikit-learn
pydantic
```

---

# ▶️ Run Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# ⚡ Run FastAPI Backend

The project also includes a FastAPI backend.

Start the server using:

```bash
uvicorn fastapi_app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# 📚 FastAPI Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the prediction endpoint directly from Swagger UI.

---

# 🔌 API Endpoint

## POST `/predict`

The FastAPI backend provides a prediction endpoint:

```http
POST /predict
```

The API accepts customer information and returns the prediction and probability.

### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 55.50,
  "TotalCharges": 666.00
}
```

### Example Response

```json
{
  "prediction": "Churn",
  "probability": 0.94
}
```

The backend converts the model prediction into either `"Churn"` or `"No Churn"` and returns the probability from `predict_proba()`.

---

# 🌐 Deployment

This project can be deployed using two separate services:

```text
                    User
                     │
                     ▼
            ┌─────────────────┐
            │ Streamlit Cloud │
            │    Frontend     │
            └────────┬────────┘
                     │
                     │ API Request
                     ▼
            ┌─────────────────┐
            │     Render      │
            │    FastAPI      │
            │     Backend     │
            └────────┬────────┘
                     │
                     ▼
             Machine Learning
                  Model
```

---

# ☁️ Deploy Streamlit on Streamlit Cloud

## Step 1 — Push Project to GitHub

Make sure your repository contains:

```text
app.py
best_model.pkl
encoder.pkl
scaler.pkl
requirements.txt
README.md
```

Then:

```bash
git add .
git commit -m "Add customer churn prediction app"
git push origin main
```

---

## Step 2 — Open Streamlit Cloud

Go to Streamlit Cloud and create a new application.

Select your GitHub repository.

Set:

```text
Repository: YOUR_USERNAME/Customer-Churn-Prediction
Branch: main
Main file: app.py
```

Then click **Deploy**.

---

## Step 3 — Streamlit URL

After deployment, you will receive a URL similar to:

```text
https://customer-churn-prediction.streamlit.app
```

Add your real URL to the top of this README.

---

# 🚀 Deploy FastAPI on Render

## Step 1 — Create a Render Web Service

Connect your GitHub repository to Render.

Choose:

```text
Environment: Python
```

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT
```

---

## Step 2 — Deploy

Click **Create Web Service**.

Render will build and start your FastAPI application.

Your API URL will look similar to:

```text
https://customer-churn-api.onrender.com
```

---

# 🧪 Test Deployed API

After deployment, open:

```text
https://customer-churn-api.onrender.com/docs
```

FastAPI Swagger UI will allow you to test:

```text
POST /predict
```

You can enter customer information and receive the prediction directly from the deployed API.

---

# 🔐 Security

Do not upload sensitive information or private credentials to GitHub.

Use `.gitignore`:

```text
venv/
__pycache__/
*.pyc
.env
```

If you use environment variables in the future, store them in the deployment platform's environment-variable settings rather than directly inside the source code.

---

# 📊 Example Prediction

### Input

```text
Gender: Female
Senior Citizen: 0
Partner: Yes
Dependents: No
Tenure: 12
Phone Service: Yes
Internet Service: DSL
Contract: Month-to-month
Paperless Billing: Yes
Payment Method: Electronic check
Monthly Charges: 55.50
Total Charges: 666.00
```

### Output

```text
⚠️ Customer is likely to churn.

Churn Probability: 94.00%
```

> The probability shown above is only an example. Actual predictions depend on the trained model and input values.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Predict customer churn using Machine Learning.
* Process categorical and numerical customer information.
* Build an interactive ML application using Streamlit.
* Develop a REST API using FastAPI.
* Serve predictions through an API.
* Display churn probability.
* Deploy a Machine Learning application to the cloud.
* Demonstrate an end-to-end ML deployment workflow.

---

# 🔮 Future Improvements

Possible improvements include:

* 📈 Add model performance metrics
* 📊 Add confusion matrix and classification reports
* 📉 Add feature importance visualization
* 👥 Add customer segmentation
* 📧 Add automated retention recommendations
* 🔐 Add user authentication
* 🗄️ Add database support
* 📱 Improve mobile responsiveness
* 🔄 Add automated model retraining
* 📦 Dockerize the application
* ☁️ Add CI/CD deployment

---

# 👨‍💻 Author

## Jaswanth

GitHub:


https://github.com/DhoniJaswanth


---

# ⭐ Support

If you found this project useful, please consider giving the repository a ⭐ on GitHub.

**Thank you for checking out my Customer Churn Prediction project!** 🚀
