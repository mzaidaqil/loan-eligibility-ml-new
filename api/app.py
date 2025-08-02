from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

app = FastAPI(title="Loan Eligibility API", version="1.0.0")

# Global variable to store model
model_artifacts = None

class LoanApplication(BaseModel):
    company_id: str
    dso_days: float = 30.0
    cash_collection_ratio: float = 0.85
    avg_payment_delay_days: float = 15.0
    immediate_payment_rate: float = 0.6
    high_delay_payment_rate: float = 0.1
    current_ratio: float = 1.5
    quick_ratio: float = 1.2
    avg_net_cash_flow: float = 10000.0
    inflow_outflow_ratio: float = 1.1
    expense_coverage_ratio: float = 0.8
    avg_quarterly_revenue: float = 150000.0
    revenue_trend: float = 0.02
    revenue_consistency: float = 0.85
    company_age_years: float = 5.0
    employee_count: float = 25.0
    sector: str = "Food and Beverages"

def load_model():
    """Load the trained model artifacts"""
    global model_artifacts
    try:
        # Try multiple possible paths for Render deployment
        possible_paths = [
            "/opt/render/project/src/models/loan_eligibility_model.pkl",
            "models/loan_eligibility_model.pkl",
            "./models/loan_eligibility_model.pkl",
            "../models/loan_eligibility_model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("❌ Model file not found in any expected location")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            return False
        
        model_artifacts = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Model name: {model_artifacts['model_name']}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Current working directory: {os.getcwd()}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("⚠️ Model not loaded - API will return default predictions")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Loan Eligibility API is running",
        "status": "healthy",
        "model_loaded": model_artifacts is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "model_name": model_artifacts['model_name'] if model_artifacts else None
    }

@app.post("/predict")
async def predict_loan_eligibility(application: LoanApplication):
    """Predict loan eligibility for a company"""
    
    if model_artifacts is None:
        # Return a default prediction if model is not loaded
        return {
            "company_id": application.company_id,
            "loan_eligible": True,  # Default optimistic prediction
            "confidence": 0.5,
            "risk_level": "medium",
            "message": "Model not available - returning default prediction"
        }
    
    try:
        # Prepare features
        features = {
            "dso_days": application.dso_days,
            "cash_collection_ratio": application.cash_collection_ratio,
            "avg_payment_delay_days": application.avg_payment_delay_days,
            "immediate_payment_rate": application.immediate_payment_rate,
            "high_delay_payment_rate": application.high_delay_payment_rate,
            "current_ratio": application.current_ratio,
            "quick_ratio": application.quick_ratio,
            "avg_net_cash_flow": application.avg_net_cash_flow,
            "inflow_outflow_ratio": application.inflow_outflow_ratio,
            "expense_coverage_ratio": application.expense_coverage_ratio,
            "avg_quarterly_revenue": application.avg_quarterly_revenue,
            "revenue_trend": application.revenue_trend,
            "revenue_consistency": application.revenue_consistency,
            "company_age_years": application.company_age_years,
            "employee_count": application.employee_count,
            "sector_encoded": 0  # Food and Beverages = 0
        }
        
        # Get model components
        model = model_artifacts['model']
        scaler = model_artifacts.get('scaler')
        feature_columns = model_artifacts['feature_columns']
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select and order features
        X = df[feature_columns]
        
        # Handle missing/infinite values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale features if needed
        if scaler is not None and model_artifacts['model_name'] == 'Logistic Regression':
            X = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Determine risk level
        confidence = max(prediction_proba)
        if confidence > 0.8:
            risk_level = "low" if prediction else "high"
        elif confidence > 0.6:
            risk_level = "medium"
        else:
            risk_level = "high" if prediction else "very_high"
        
        return {
            "company_id": application.company_id,
            "loan_eligible": bool(prediction),
            "confidence": float(confidence),
            "risk_level": risk_level,
            "probability_eligible": float(prediction_proba[1]),
            "probability_not_eligible": float(prediction_proba[0]),
            "model_used": model_artifacts['model_name']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model_artifacts is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    return {
        "model_name": model_artifacts['model_name'],
        "performance_metrics": model_artifacts.get('performance_metrics', {}),
        "feature_columns": model_artifacts.get('feature_columns', []),
        "training_date": model_artifacts.get('training_date', 'Unknown')
    }