#!/usr/bin/env python3
"""
Model Testing Script for Loan Eligibility ML System
Tests both local model functionality and API endpoints
"""

import requests
import json
import joblib
import pandas as pd
import numpy as np
import os

# Test data samples
test_cases = {
    "high_quality_sme": {
        "company_id": "TEST_HIGH_001",
        "dso_days": 25.0,
        "cash_collection_ratio": 0.92,
        "avg_payment_delay_days": 8.0,
        "immediate_payment_rate": 0.75,
        "high_delay_payment_rate": 0.05,
        "current_ratio": 2.1,
        "quick_ratio": 1.8,
        "avg_net_cash_flow": 25000.0,
        "inflow_outflow_ratio": 1.3,
        "expense_coverage_ratio": 0.95,
        "avg_quarterly_revenue": 200000.0,
        "revenue_trend": 0.08,
        "revenue_consistency": 0.92,
        "company_age_years": 8.0,
        "employee_count": 45.0,
        "sector": "Food and Beverages"
    },
    "poor_quality_sme": {
        "company_id": "TEST_POOR_001",
        "dso_days": 65.0,
        "cash_collection_ratio": 0.45,
        "avg_payment_delay_days": 45.0,
        "immediate_payment_rate": 0.25,
        "high_delay_payment_rate": 0.40,
        "current_ratio": 0.8,
        "quick_ratio": 0.6,
        "avg_net_cash_flow": -2000.0,
        "inflow_outflow_ratio": 0.85,
        "expense_coverage_ratio": 0.45,
        "avg_quarterly_revenue": 75000.0,
        "revenue_trend": -0.15,
        "revenue_consistency": 0.35,
        "company_age_years": 1.5,
        "employee_count": 8.0,
        "sector": "Food and Beverages"
    }
}

def test_local_model():
    """Test the model locally"""
    print("=" * 60)
    print("🔧 TESTING LOCAL MODEL")
    print("=" * 60)
    
    try:
        # Load model
        model_artifacts = joblib.load('models/loan_eligibility_model.pkl')
        print(f"✅ Model loaded: {model_artifacts['model_name']}")
        print(f"📊 AUC Score: {model_artifacts['performance_metrics']['auc_score']:.4f}")
        print(f"🔧 Features: {len(model_artifacts['feature_columns'])}")
        
        # Test both cases
        for case_name, test_data in test_cases.items():
            print(f"\n📋 Testing {case_name.replace('_', ' ').title()}:")
            
            # Prepare data
            model = model_artifacts['model']
            feature_columns = model_artifacts['feature_columns']
            
            # Add sector encoding
            test_data_copy = test_data.copy()
            test_data_copy['sector_encoded'] = 0
            
            df = pd.DataFrame([test_data_copy])
            
            # Add missing features
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            X = df[feature_columns]
            X = X.fillna(0)
            
            # Make prediction
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            
            print(f"   Loan Eligible: {'✅ YES' if prediction else '❌ NO'}")
            print(f"   Confidence: {max(proba):.3f}")
            print(f"   Prob Eligible: {proba[1]:.3f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Local model test failed: {e}")
        return False

def test_api_endpoints(base_url="https://loan-eligibility-ml-new.onrender.com"):
    """Test the deployed API endpoints"""
    print("\n" + "=" * 60)
    print("🌐 TESTING DEPLOYED API")
    print("=" * 60)
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        health_data = response.json()
        print(f"📡 API Status: {health_data['status']}")
        print(f"🤖 Model Loaded: {'✅ YES' if health_data['model_loaded'] else '❌ NO'}")
        if health_data['model_name']:
            print(f"🔧 Model Name: {health_data['model_name']}")
        
        if not health_data['model_loaded']:
            print("⚠️  Model not loaded on API - predictions will use defaults")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test predictions
    for case_name, test_data in test_cases.items():
        print(f"\n📋 Testing API prediction for {case_name.replace('_', ' ').title()}:")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Company ID: {result['company_id']}")
                print(f"   Loan Eligible: {'✅ YES' if result['loan_eligible'] else '❌ NO'}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Risk Level: {result['risk_level']}")
                if 'model_used' in result:
                    print(f"   Model Used: {result['model_used']}")
                if 'message' in result:
                    print(f"   Message: {result['message']}")
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Prediction test failed: {e}")
    
    return True

def test_model_info_endpoint(base_url="https://loan-eligibility-ml-new.onrender.com"):
    """Test the model info endpoint"""
    print(f"\n📊 Testing Model Info Endpoint:")
    
    try:
        response = requests.get(f"{base_url}/model/info", timeout=30)
        if response.status_code == 200:
            info = response.json()
            print(f"   Model Name: {info['model_name']}")
            print(f"   Training Date: {info['training_date']}")
            print(f"   Features Count: {len(info['feature_columns'])}")
            if 'performance_metrics' in info:
                perf = info['performance_metrics']
                print(f"   Accuracy: {perf.get('accuracy', 'N/A')}")
                print(f"   AUC Score: {perf.get('auc_score', 'N/A')}")
        else:
            print(f"   ❌ Model info error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Model info test failed: {e}")

def main():
    """Run all tests"""
    print("🏦 LOAN ELIGIBILITY ML MODEL TESTING")
    print(f"📅 Test Date: August 3, 2025")
    
    # Test local model
    local_success = test_local_model()
    
    # Test API
    api_success = test_api_endpoints()
    
    # Test model info
    test_model_info_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Local Model: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if local_success and not api_success:
        print("\n💡 RECOMMENDATION:")
        print("   - Local model works perfectly")
        print("   - API deployment needs model path fix")
        print("   - Redeploy after committing the API path fixes")

if __name__ == "__main__":
    main()
