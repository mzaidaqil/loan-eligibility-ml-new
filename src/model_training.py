import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
    # Encode categorical sector (though all are Food and Beverages now)
    le_sector = LabelEncoder()
    df['sector_encoded'] = le_sector.fit_transform(df['sector'])
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
    
    return X, y, feature_columns, le_sector

def train_models(X, y):
    """
    Train multiple models and compare performance
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000
        )
    }
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for logistic regression, original for tree-based models
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = model.score(X_test_scaled if name == 'Logistic Regression' else X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, 
            X_train_scaled if name == 'Logistic Regression' else X_train, 
            y_train, 
            cv=5, 
            scoring='roc_auc'
        )
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        trained_models[name] = model
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    return model_results, trained_models, scaler, X_test, y_test

def analyze_feature_importance(models, feature_columns):
    """
    Analyze and visualize feature importance
    """
    print("\nFeature Importance Analysis:")
    
    # Get feature importance from Random Forest
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Random Forest):")
    print(feature_importance.head(10))
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.xlabel('Importance Score')
    
    # Group features by category for analysis
    feature_categories = {
        'Debt Recovery': ['dso_days', 'cash_collection_ratio', 'avg_payment_delay_days', 
                         'immediate_payment_rate', 'high_delay_payment_rate', 'dso_trend', 
                         'collection_ratio_trend'],
        'Liquidity': ['current_ratio', 'quick_ratio', 'total_net_cash_flow', 'avg_net_cash_flow',
                     'cash_flow_volatility', 'liquidity_stress_ratio', 'cash_flow_trend', 'liquidity_trend'],
        'Cash Flow Sync': ['inflow_outflow_ratio', 'expense_coverage_ratio', 'quarterly_inflows', 
                          'quarterly_outflows', 'cash_flow_synchronization', 'sync_trend'],
        'Sales Performance': ['avg_quarterly_revenue', 'revenue_trend', 'revenue_consistency', 
                             'sales_receipts_lag', 'revenue_growth_rate', 'quarters_tracked'],
        'Company Info': ['company_age_years', 'employee_count', 'sector_encoded']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_importance[category] = feature_importance[
            feature_importance['feature'].isin(features)
        ]['importance'].sum()
    
    plt.subplot(2, 1, 2)
    categories = list(category_importance.keys())
    importances = list(category_importance.values())
    plt.bar(categories, importances)
    plt.title('Feature Importance by Category')
    plt.ylabel('Total Importance Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/zayed/loan_eligibility_ml/models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def create_model_evaluation_plots(model_results, y_test):
    """
    Create evaluation plots for model comparison
    """
    plt.figure(figsize=(15, 10))
    
    # ROC Curves
    plt.subplot(2, 3, 1)
    for name, results in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC: {results['auc_score']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    
    # Model Performance Comparison
    plt.subplot(2, 3, 2)
    models_names = list(model_results.keys())
    accuracies = [model_results[name]['accuracy'] for name in models_names]
    auc_scores = [model_results[name]['auc_score'] for name in models_names]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models_names, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    
    # Confusion Matrix for best model (highest AUC)
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
    best_predictions = model_results[best_model_name]['predictions']
    
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Cross-validation scores
    plt.subplot(2, 3, 4)
    cv_means = [model_results[name]['cv_mean'] for name in models_names]
    cv_stds = [model_results[name]['cv_std'] for name in models_names]
    
    plt.bar(models_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
    plt.ylabel('Cross-Validation AUC Score')
    plt.title('Cross-Validation Performance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/zayed/loan_eligibility_ml/models/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name

def save_models_and_results(models, scaler, feature_columns, model_results, feature_importance):
    """
    Save trained models and results
    """
    models_dir = '/Users/zayed/loan_eligibility_ml/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save best model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
    best_model = models[best_model_name]
    
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'model_name': best_model_name,
        'performance_metrics': model_results[best_model_name]
    }
    
    joblib.dump(model_artifacts, f'{models_dir}/loan_eligibility_model.pkl')
    print(f"\nBest model ({best_model_name}) saved to: {models_dir}/loan_eligibility_model.pkl")
    
    # Save all models
    for name, model in models.items():
        joblib.dump(model, f'{models_dir}/model_{name.lower().replace(" ", "_")}.pkl')
    
    # Save model evaluation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'{models_dir}/model_evaluation_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write("Loan Eligibility Model Evaluation Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Food and Beverages SME Companies (Quarterly Assessment)\n")
        f.write(f"Total companies: 10000\n\n")
        
        f.write("Model Performance Summary:\n")
        f.write("-" * 25 + "\n")
        for name, results in model_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  AUC Score: {results['auc_score']:.4f}\n")
            f.write(f"  CV AUC: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})\n")
        
        f.write(f"\nBest Model: {best_model_name}\n")
        f.write(f"Best AUC Score: {model_results[best_model_name]['auc_score']:.4f}\n\n")
        
        f.write("Top 10 Most Important Features:\n")
        f.write("-" * 30 + "\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    print(f"Model evaluation report saved to: {report_path}")
    
    return best_model_name, model_artifacts

def main():
    """
    Main model training pipeline
    """
    print("Starting Loan Eligibility Model Training Pipeline")
    print("=" * 50)
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, feature_columns, le_sector = prepare_features_and_target(df)
    
    # Train models
    model_results, trained_models, scaler, X_test, y_test = train_models(X, y)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(trained_models, feature_columns)
    
    # Create evaluation plots
    best_model_name = create_model_evaluation_plots(model_results, y_test)
    
    # Save models and results
    best_model_name, model_artifacts = save_models_and_results(
        trained_models, scaler, feature_columns, model_results, feature_importance
    )
    
    print(f"\nModel training completed successfully!")
    print(f"Best performing model: {best_model_name}")
    print(f"Best AUC Score: {model_results[best_model_name]['auc_score']:.4f}")
    
    return model_artifacts

if __name__ == "__main__":
    model_artifacts = main()
