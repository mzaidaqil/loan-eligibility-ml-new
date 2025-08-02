import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

def load_and_prepare_data():
    """
    Load engineered features and prepare for model training
    """
    print("Loading engineered features...")
    df = pd.read_csv('/Users/zayed/loan_eligibility_ml/data/processed/sme_financial_features.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sectors: {df['sector'].unique()}")
    print(f"Target distribution:")
    print(df['loan_eligible'].value_counts())
    
    return df

def prepare_features_and_target(df):
    """
    Prepare features and target variable for model training
    """
    # Define feature groups
    debt_recovery_features = [
        'dso_days', 'cash_collection_ratio', 'avg_payment_delay_days',
        'immediate_payment_rate', 'high_delay_payment_rate', 'dso_trend', 
        'collection_ratio_trend', 'quarters_of_data'
    ]
    
    liquidity_features = [
        'current_ratio', 'quick_ratio', 'total_net_cash_flow', 'avg_net_cash_flow',
        'cash_flow_volatility', 'liquidity_stress_ratio', 'cash_flow_trend', 'liquidity_trend'
    ]
    
    cash_flow_features = [
        'inflow_outflow_ratio', 'expense_coverage_ratio', 'quarterly_inflows', 'quarterly_outflows',
        'cash_flow_synchronization', 'sync_trend'
    ]
    
    sales_features = [
        'avg_quarterly_revenue', 'revenue_trend', 'revenue_consistency', 'sales_receipts_lag',
        'revenue_growth_rate', 'quarters_tracked'
    ]
    
    company_features = [
        'company_age_years', 'employee_count'
    ]
    
    # Combine all numerical features
    feature_columns = (debt_recovery_features + liquidity_features + 
                      cash_flow_features + sales_features + company_features)
    
    # Encode categorical sector (Food and Beverages = 0)
    df['sector_encoded'] = 0  # Since all are Food and Beverages
    feature_columns.append('sector_encoded')
    
    # Prepare features and target
    X = df[feature_columns].copy()
    y = df['loan_eligible'].copy()
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {feature_columns}")
    
    return X, y, feature_columns

def train_gradient_boosting_model(X, y):
    """
    Train Gradient Boosting model only
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    
    # Train Gradient Boosting model
    print("\nTraining Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42,
        learning_rate=0.1,
        max_depth=6
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    model_results = {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model, model_results, X_test, y_test, y_pred, y_pred_proba

def save_model_and_results(model, feature_columns, model_results):
    """
    Save Gradient Boosting model and results
    """
    models_dir = '/Users/zayed/loan_eligibility_ml/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create simplified model artifacts (no scaler needed for Gradient Boosting)
    model_artifacts = {
        'model': model,
        'feature_columns': feature_columns,
        'model_name': 'Gradient Boosting',
        'performance_metrics': model_results,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save the model
    model_path = f'{models_dir}/loan_eligibility_model.pkl'
    joblib.dump(model_artifacts, model_path)
    print(f"\nGradient Boosting model saved to: {model_path}")
    
    return model_artifacts

def validate_saved_model(model_path, X_test, y_test):
    """
    Validate that the saved model can be loaded and makes predictions
    """
    try:
        print(f"\nValidating saved model at: {model_path}")
        
        # Load the saved model
        loaded_artifacts = joblib.load(model_path)
        loaded_model = loaded_artifacts['model']
        feature_columns = loaded_artifacts['feature_columns']
        
        # Make predictions with loaded model
        y_pred_loaded = loaded_model.predict(X_test)
        y_pred_proba_loaded = loaded_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = loaded_model.score(X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba_loaded)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model type: {type(loaded_model).__name__}")
        print(f"✓ Feature count: {len(feature_columns)}")
        print(f"✓ Validation Accuracy: {accuracy:.4f}")
        print(f"✓ Validation AUC: {auc_score:.4f}")
        
        # Test with a single sample
        sample_prediction = loaded_model.predict(X_test.iloc[:1])
        sample_proba = loaded_model.predict_proba(X_test.iloc[:1])[:, 1]
        
        print(f"✓ Sample prediction: {sample_prediction[0]} (confidence: {sample_proba[0]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {str(e)}")
        return False

def main():
    """
    Main model training pipeline - Gradient Boosting only
    """
    print("Starting Loan Eligibility Model Training (Gradient Boosting)")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, feature_columns = prepare_features_and_target(df)
    
    # Train Gradient Boosting model
    model, model_results, X_test, y_test, y_pred, y_pred_proba = train_gradient_boosting_model(X, y)
    
    # Save model and results
    model_artifacts = save_model_and_results(model, feature_columns, model_results)
    
    # Validate the saved model
    model_path = '/Users/zayed/loan_eligibility_ml/models/loan_eligibility_model.pkl'
    validation_success = validate_saved_model(model_path, X_test, y_test)
    
    if validation_success:
        print(f"\n✓ Model training and validation completed successfully!")
    else:
        print(f"\n✗ Model training completed but validation failed!")
    
    print(f"Gradient Boosting AUC Score: {model_results['auc_score']:.4f}")
    print(f"Model saved to: {model_path}")
    
    return model_artifacts

if __name__ == "__main__":
    model_artifacts = main()
