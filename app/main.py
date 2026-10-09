import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn & LTV Prediction API",
    description="REST API for predicting customer churn probability and Lifetime Value (LTV)",
    version="1.0.0",
)

# Model file paths
CHURN_MODEL_PATH = os.path.join("models", "churn_pipeline.joblib")
LTV_MODEL_PATH = os.path.join("models", "ltv_pipeline.joblib")

# Load pre-trained model pipelines
try:
    churn_pipeline = joblib.load(CHURN_MODEL_PATH)
    ltv_pipeline = joblib.load(LTV_MODEL_PATH)
    print("Models loaded successfully!")
except Exception as e:
    churn_pipeline = None
    ltv_pipeline = None
    print(f"Error loading models: {e}")


# Define Pydantic Schema for Request Body Validation
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Customer Churn & LTV Prediction API is up and running!",
    }


@app.post("/predict")
def predict_churn_and_ltv(customer: CustomerData):
    """Prediction endpoint for evaluating Churn risk and LTV for a customer."""
    if churn_pipeline is None or ltv_pipeline is None:
        raise HTTPException(
            status_code=500, detail="Model pipelines are not loaded properly."
        )

    # Convert incoming Pydantic model to Pandas DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    # Calculate TotalCharges for Churn model input if not provided
    input_df["TotalCharges"] = input_df["tenure"] * input_df["MonthlyCharges"]

    # 1. Churn Prediction
    churn_pred = int(churn_pipeline.predict(input_df)[0])
    churn_prob = float(churn_pipeline.predict_proba(input_df)[0][1])

    # 2. Add predicted Churn status for LTV prediction model
    input_df_ltv = input_df.drop(columns=["TotalCharges"])
    input_df_ltv["Churn"] = churn_pred

    # LTV Prediction
    predicted_ltv = float(ltv_pipeline.predict(input_df_ltv)[0])

    # Construct Response
    risk_level = (
        "High Risk"
        if churn_prob > 0.6
        else ("Medium Risk" if churn_prob > 0.3 else "Low Risk")
    )

    return {
        "churn_prediction": "Yes" if churn_pred == 1 else "No",
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "predicted_ltv": round(predicted_ltv, 2),
    }