# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("models/loan_risk_classifier.pkl")

# Initialize FastAPI
app = FastAPI(title="Loan Risk Classifier API")

# Define request schema
class BusinessData(BaseModel):
    Avg_Monthly_Sales: float
    Avg_Monthly_Expenses: float
    Volatility: float
    Existing_Debt: float
    Loan_Request: float
    Net_Operating_Cash_Flow: float
    DSR: float

# Define API endpoint
@app.post("/predict")
def predict_risk(data: BusinessData):
    # Convert input to model format
    input_data = np.array([[
        data.Avg_Monthly_Sales,
        data.Avg_Monthly_Expenses,
        data.Volatility,
        data.Existing_Debt,
        data.Loan_Request,
        data.Net_Operating_Cash_Flow,
        data.DSR
    ]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {"Risk_Category": prediction}
