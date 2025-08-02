import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

def load_trained_model():
    """
    Load the trained model and its artifacts
    """
    try:
        model_artifacts = joblib.load('/Users/zayed/loan_eligibility_ml/models/loan_eligibility_model.pkl')
        print(f"✅ Model loaded successfully: {model_artifacts['model_name']}")
        print(f"📊 Model performance: AUC = {model_artifacts['performance_metrics']['auc_score']:.4f}")
        return model_artifacts
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_loan_eligibility(model_artifacts, company_data):
    """
    Predict loan eligibility for a single company
    """
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_columns = model_artifacts['feature_columns']
    
    # Convert to DataFrame
    df = pd.DataFrame([company_data])
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in df.columns:
            if col == 'sector_encoded':
                df[col] = 0  # Food and Beverages = 0
            else:
                df[col] = 0  # Default value
    
    # Select and order features
    X = df[feature_columns]
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Scale features if needed (for Logistic Regression)
    if model_artifacts['model_name'] == 'Logistic Regression':
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
    else:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'Low Risk' if prediction == 1 else 'High Risk',
        'confidence': f"{probability*100:.1f}%" if prediction == 1 else f"{(1-probability)*100:.1f}%"
    }

def create_test_cases():
    """
    Create comprehensive test cases for different company profiles
    """
    test_cases = {
        "test_companies": [
            {
                "name": "Excellent Company (Should be Approved)",
                "description": "High-performing food & beverage company with strong financials",
                "data": {
                    "dso_days": 25.0,
                    "cash_collection_ratio": 0.95,
                    "avg_payment_delay_days": 15.0,
                    "immediate_payment_rate": 0.60,
                    "high_delay_payment_rate": 0.05,
                    "dso_trend": -2.0,
                    "collection_ratio_trend": 0.02,
                    "quarters_of_data": 4,
                    "current_ratio": 2.5,
                    "quick_ratio": 1.8,
                    "total_net_cash_flow": 80000.0,
                    "avg_net_cash_flow": 20000.0,
                    "cash_flow_volatility": 5000.0,
                    "liquidity_stress_ratio": 0.10,
                    "cash_flow_trend": 2000.0,
                    "liquidity_trend": 0.1,
                    "inflow_outflow_ratio": 1.35,
                    "expense_coverage_ratio": 1.35,
                    "quarterly_inflows": 800000.0,
                    "quarterly_outflows": 600000.0,
                    "cash_flow_synchronization": 0.85,
                    "sync_trend": 0.05,
                    "avg_quarterly_revenue": 200000.0,
                    "revenue_trend": 5000.0,
                    "revenue_consistency": 0.90,
                    "sales_receipts_lag": 20.0,
                    "revenue_growth_rate": 0.15,
                    "quarters_tracked": 4,
                    "company_age_years": 8.0,
                    "employee_count": 45,
                    "sector_encoded": 0
                },
                "expected_result": "APPROVED"
            },
            {
                "name": "Good Company (Likely Approved)",
                "description": "Solid food & beverage company with good fundamentals",
                "data": {
                    "dso_days": 35.0,
                    "cash_collection_ratio": 0.88,
                    "avg_payment_delay_days": 25.0,
                    "immediate_payment_rate": 0.45,
                    "high_delay_payment_rate": 0.12,
                    "dso_trend": -0.5,
                    "collection_ratio_trend": 0.01,
                    "quarters_of_data": 4,
                    "current_ratio": 2.0,
                    "quick_ratio": 1.4,
                    "total_net_cash_flow": 60000.0,
                    "avg_net_cash_flow": 15000.0,
                    "cash_flow_volatility": 8000.0,
                    "liquidity_stress_ratio": 0.20,
                    "cash_flow_trend": 1000.0,
                    "liquidity_trend": 0.05,
                    "inflow_outflow_ratio": 1.25,
                    "expense_coverage_ratio": 1.25,
                    "quarterly_inflows": 600000.0,
                    "quarterly_outflows": 480000.0,
                    "cash_flow_synchronization": 0.75,
                    "sync_trend": 0.02,
                    "avg_quarterly_revenue": 150000.0,
                    "revenue_trend": 2000.0,
                    "revenue_consistency": 0.80,
                    "sales_receipts_lag": 30.0,
                    "revenue_growth_rate": 0.08,
                    "quarters_tracked": 4,
                    "company_age_years": 5.0,
                    "employee_count": 30,
                    "sector_encoded": 0
                },
                "expected_result": "LIKELY APPROVED"
            },
            {
                "name": "Average Company (Borderline)",
                "description": "Average food & beverage company with mixed performance",
                "data": {
                    "dso_days": 45.0,
                    "cash_collection_ratio": 0.78,
                    "avg_payment_delay_days": 40.0,
                    "immediate_payment_rate": 0.35,
                    "high_delay_payment_rate": 0.25,
                    "dso_trend": 0.0,
                    "collection_ratio_trend": 0.0,
                    "quarters_of_data": 4,
                    "current_ratio": 1.5,
                    "quick_ratio": 1.0,
                    "total_net_cash_flow": 30000.0,
                    "avg_net_cash_flow": 7500.0,
                    "cash_flow_volatility": 12000.0,
                    "liquidity_stress_ratio": 0.35,
                    "cash_flow_trend": 0.0,
                    "liquidity_trend": 0.0,
                    "inflow_outflow_ratio": 1.10,
                    "expense_coverage_ratio": 1.10,
                    "quarterly_inflows": 440000.0,
                    "quarterly_outflows": 400000.0,
                    "cash_flow_synchronization": 0.60,
                    "sync_trend": 0.0,
                    "avg_quarterly_revenue": 110000.0,
                    "revenue_trend": 0.0,
                    "revenue_consistency": 0.65,
                    "sales_receipts_lag": 45.0,
                    "revenue_growth_rate": 0.02,
                    "quarters_tracked": 4,
                    "company_age_years": 3.0,
                    "employee_count": 20,
                    "sector_encoded": 0
                },
                "expected_result": "BORDERLINE"
            },
            {
                "name": "Poor Company (Likely Rejected)",
                "description": "Struggling food & beverage company with financial issues",
                "data": {
                    "dso_days": 65.0,
                    "cash_collection_ratio": 0.65,
                    "avg_payment_delay_days": 55.0,
                    "immediate_payment_rate": 0.20,
                    "high_delay_payment_rate": 0.45,
                    "dso_trend": 2.0,
                    "collection_ratio_trend": -0.02,
                    "quarters_of_data": 4,
                    "current_ratio": 1.1,
                    "quick_ratio": 0.7,
                    "total_net_cash_flow": 5000.0,
                    "avg_net_cash_flow": 1250.0,
                    "cash_flow_volatility": 15000.0,
                    "liquidity_stress_ratio": 0.60,
                    "cash_flow_trend": -1000.0,
                    "liquidity_trend": -0.05,
                    "inflow_outflow_ratio": 1.02,
                    "expense_coverage_ratio": 1.02,
                    "quarterly_inflows": 306000.0,
                    "quarterly_outflows": 300000.0,
                    "cash_flow_synchronization": 0.40,
                    "sync_trend": -0.02,
                    "avg_quarterly_revenue": 76500.0,
                    "revenue_trend": -1000.0,
                    "revenue_consistency": 0.45,
                    "sales_receipts_lag": 60.0,
                    "revenue_growth_rate": -0.05,
                    "quarters_tracked": 4,
                    "company_age_years": 2.0,
                    "employee_count": 12,
                    "sector_encoded": 0
                },
                "expected_result": "LIKELY REJECTED"
            },
            {
                "name": "High Risk Company (Should be Rejected)",
                "description": "High-risk food & beverage company with serious financial problems",
                "data": {
                    "dso_days": 80.0,
                    "cash_collection_ratio": 0.50,
                    "avg_payment_delay_days": 70.0,
                    "immediate_payment_rate": 0.10,
                    "high_delay_payment_rate": 0.65,
                    "dso_trend": 5.0,
                    "collection_ratio_trend": -0.05,
                    "quarters_of_data": 4,
                    "current_ratio": 0.8,
                    "quick_ratio": 0.4,
                    "total_net_cash_flow": -10000.0,
                    "avg_net_cash_flow": -2500.0,
                    "cash_flow_volatility": 20000.0,
                    "liquidity_stress_ratio": 0.80,
                    "cash_flow_trend": -2000.0,
                    "liquidity_trend": -0.1,
                    "inflow_outflow_ratio": 0.95,
                    "expense_coverage_ratio": 0.95,
                    "quarterly_inflows": 190000.0,
                    "quarterly_outflows": 200000.0,
                    "cash_flow_synchronization": 0.20,
                    "sync_trend": -0.05,
                    "avg_quarterly_revenue": 47500.0,
                    "revenue_trend": -2000.0,
                    "revenue_consistency": 0.30,
                    "sales_receipts_lag": 75.0,
                    "revenue_growth_rate": -0.15,
                    "quarters_tracked": 4,
                    "company_age_years": 1.5,
                    "employee_count": 8,
                    "sector_encoded": 0
                },
                "expected_result": "REJECTED"
            }
        ]
    }
    
    return test_cases

def run_tests():
    """
    Run all test cases and display results
    """
    print("🧪 Loading Loan Eligibility Model Test Suite")
    print("=" * 50)
    
    # Load model
    model_artifacts = load_trained_model()
    if not model_artifacts:
        return
    
    # Load test cases
    test_cases = create_test_cases()
    
    print(f"\n🎯 Running {len(test_cases['test_companies'])} test cases...\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases['test_companies'], 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected: {test_case['expected_result']}")
        
        # Make prediction
        prediction = predict_loan_eligibility(model_artifacts, test_case['data'])
        
        # Determine actual result
        if prediction['prediction'] == 1:
            if prediction['probability'] > 0.8:
                actual_result = "APPROVED"
            else:
                actual_result = "LIKELY APPROVED"
        else:
            if prediction['probability'] < 0.2:
                actual_result = "REJECTED"
            else:
                actual_result = "LIKELY REJECTED"
        
        print(f"Actual: {actual_result}")
        print(f"Probability: {prediction['probability']:.3f} ({prediction['confidence']})")
        
        # Check if prediction matches expectation
        expected_approved = test_case['expected_result'] in ['APPROVED', 'LIKELY APPROVED']
        actual_approved = prediction['prediction'] == 1
        
        if expected_approved == actual_approved:
            print("✅ Result: PASS")
        else:
            print("❌ Result: FAIL")
        
        results.append({
            'test_name': test_case['name'],
            'expected': test_case['expected_result'],
            'actual': actual_result,
            'probability': prediction['probability'],
            'passed': expected_approved == actual_approved
        })
        
        print("-" * 50)
    
    # Summary
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"\n📊 Test Summary:")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Failed: {total-passed}/{total} ({(total-passed)/total*100:.1f}%)")
    
    return results

def save_test_json():
    """
    Save test cases as JSON for external testing
    """
    test_cases = create_test_cases()
    
    # Add API testing format
    api_test_format = {
        "model_info": {
            "model_type": "Loan Eligibility Classifier",
            "sector": "Food and Beverages",
            "assessment_period": "Quarterly",
            "features_count": 31,
            "training_companies": 10000
        },
        "api_endpoint": "/predict_loan_eligibility",
        "request_format": {
            "method": "POST",
            "content_type": "application/json",
            "body_structure": "See test_cases below"
        },
        "response_format": {
            "prediction": "0 or 1 (0=rejected, 1=approved)",
            "probability": "float between 0 and 1",
            "risk_level": "string: 'Low Risk' or 'High Risk'",
            "confidence": "string: percentage confidence"
        },
        "test_cases": test_cases["test_companies"]
    }
    
    # Save to JSON file
    output_path = '/Users/zayed/loan_eligibility_ml/tests/test_cases.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(api_test_format, f, indent=2)
    
    print(f"📄 Test cases saved to: {output_path}")
    
    # Also save a simple format for quick testing
    simple_format = {
        "instructions": "Use these test cases to validate your model API",
        "note": "Each company should be tested individually by sending the 'data' object to your API",
        "test_cases": [
            {
                "company_name": tc["name"],
                "expected_result": tc["expected_result"], 
                "test_data": tc["data"]
            } for tc in test_cases["test_companies"]
        ]
    }
    
    simple_path = '/Users/zayed/loan_eligibility_ml/tests/simple_test_cases.json'
    with open(simple_path, 'w') as f:
        json.dump(simple_format, f, indent=2)
    
    print(f"📄 Simple test cases saved to: {simple_path}")
    
    return output_path, simple_path

if __name__ == "__main__":
    import os
    
    # Run tests
    results = run_tests()
    
    # Save JSON test cases
    api_path, simple_path = save_test_json()
    
    print(f"\n🚀 Ready for deployment testing!")
    print(f"📁 API Test Cases: {api_path}")
    print(f"📁 Simple Test Cases: {simple_path}")
