import pandas as pd
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def check_overfitting():
    """
    Analyze the model for potential overfitting issues
    """
    print("Analyzing potential overfitting...")
    
    # Load the data
    df = pd.read_csv('/Users/zayed/loan_eligibility_ml/data/processed/sme_financial_features.csv')
    
    # Prepare features (simplified version)
    feature_columns = [
        'dso_days', 'cash_collection_ratio', 'avg_payment_delay_days',
        'immediate_payment_rate', 'high_delay_payment_rate', 
        'current_ratio', 'quick_ratio', 'avg_net_cash_flow',
        'inflow_outflow_ratio', 'expense_coverage_ratio',
        'avg_quarterly_revenue', 'revenue_trend', 'revenue_consistency',
        'company_age_years', 'employee_count'
    ]
    
    X = df[feature_columns].copy()
    y = df['loan_eligible'].copy()
    
    # Handle missing/infinite values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features used: {len(feature_columns)}")
    print(f"Sample to feature ratio: {len(X) / len(feature_columns):.1f}:1")
    
    # 1. Learning Curves Analysis
    print("\n1. Generating Learning Curves...")
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc', random_state=42
    )
    
    # Plot learning curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training AUC')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation AUC')
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('AUC Score')
    plt.title('Learning Curves - Overfitting Check')
    plt.legend()
    plt.grid(True)
    
    # 2. Validation Curves for Model Complexity
    print("2. Analyzing Model Complexity...")
    
    param_range = [10, 25, 50, 75, 100, 150, 200]
    train_scores, val_scores = validation_curve(
        GradientBoostingClassifier(random_state=42), X, y, 
        param_name='n_estimators', param_range=param_range, 
        cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    plt.subplot(1, 3, 2)
    plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training AUC')
    plt.plot(param_range, np.mean(val_scores, axis=1), 'o-', label='Validation AUC')
    plt.fill_between(param_range, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    plt.fill_between(param_range, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
    plt.xlabel('Number of Estimators')
    plt.ylabel('AUC Score')
    plt.title('Validation Curves - Model Complexity')
    plt.legend()
    plt.grid(True)
    
    # 3. Feature Importance Stability
    print("3. Checking Feature Importance Stability...")
    
    # Train multiple models with different random states
    feature_importances = []
    for seed in range(5):
        model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        model.fit(X, y)
        feature_importances.append(model.feature_importances_)
    
    # Calculate coefficient of variation for each feature
    feature_importances = np.array(feature_importances)
    mean_importances = np.mean(feature_importances, axis=0)
    std_importances = np.std(feature_importances, axis=0)
    cv_importances = std_importances / (mean_importances + 1e-8)  # Avoid division by zero
    
    plt.subplot(1, 3, 3)
    indices = np.argsort(mean_importances)[::-1][:10]  # Top 10 features
    plt.bar(range(10), cv_importances[indices])
    plt.xlabel('Top 10 Features (by importance)')
    plt.ylabel('Coefficient of Variation')
    plt.title('Feature Importance Stability')
    plt.xticks(range(10), [feature_columns[i] for i in indices], rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/zayed/loan_eligibility_ml/models/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Calculate overfitting metrics
    print("\n4. Overfitting Assessment:")
    
    final_train_auc = np.mean(train_scores[-1])  # AUC at full dataset size
    final_val_auc = np.mean(val_scores[-1])
    
    gap = final_train_auc - final_val_auc
    relative_gap = gap / final_val_auc * 100
    
    print(f"Training AUC (learning curve): {final_train_auc:.4f}")
    print(f"Validation AUC (learning curve): {final_val_auc:.4f}")
    print(f"Performance Gap: {gap:.4f}")
    print(f"Relative Gap: {relative_gap:.2f}%")
    
    # Assessment
    if gap > 0.05:
        print("⚠️  HIGH OVERFITTING DETECTED!")
        overfitting_level = "HIGH"
    elif gap > 0.02:
        print("⚠️  Moderate overfitting detected")
        overfitting_level = "MODERATE"
    else:
        print("✅ Low risk of overfitting")
        overfitting_level = "LOW"
    
    # 5. Data Quality Assessment
    print("\n5. Data Quality Assessment:")
    
    # Check for unrealistic patterns
    correlations = X.corr().abs()
    high_corr_pairs = []
    for i in range(len(correlations.columns)):
        for j in range(i+1, len(correlations.columns)):
            if correlations.iloc[i, j] > 0.9:
                high_corr_pairs.append((correlations.columns[i], correlations.columns[j], correlations.iloc[i, j]))
    
    print(f"High correlation pairs (>0.9): {len(high_corr_pairs)}")
    for pair in high_corr_pairs[:5]:  # Show first 5
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    
    # Check target distribution by key features
    print(f"\nTarget distribution by quartiles:")
    for feature in ['avg_net_cash_flow', 'current_ratio', 'dso_days']:
        quartiles = pd.qcut(X[feature], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        target_by_quartile = df.groupby(quartiles)['loan_eligible'].mean()
        print(f"{feature}: {target_by_quartile.to_dict()}")
    
    # 6. Recommendations
    print("\n6. Recommendations:")
    
    if overfitting_level == "HIGH":
        print("🔧 IMMEDIATE ACTIONS NEEDED:")
        print("   - Reduce model complexity (fewer estimators, max_depth)")
        print("   - Add regularization")
        print("   - Collect more diverse data")
        print("   - Use feature selection")
        print("   - Add noise to synthetic data generation")
    elif overfitting_level == "MODERATE":
        print("🔧 SUGGESTED IMPROVEMENTS:")
        print("   - Consider regularization")
        print("   - Monitor on larger dataset")
        print("   - Cross-validate with different splits")
    else:
        print("✅ Model appears well-generalized")
    
    # 7. Save analysis results
    analysis_results = {
        'overfitting_level': overfitting_level,
        'performance_gap': gap,
        'relative_gap_percent': relative_gap,
        'high_correlation_pairs': len(high_corr_pairs),
        'sample_to_feature_ratio': len(X) / len(feature_columns),
        'recommendations': []
    }
    
    if overfitting_level != "LOW":
        analysis_results['recommendations'] = [
            "Consider regularization techniques",
            "Validate on external dataset if available",
            "Monitor performance on new data",
            "Consider feature selection"
        ]
    
    # Save to file
    import json
    with open('/Users/zayed/loan_eligibility_ml/models/overfitting_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n📊 Analysis complete! Results saved to overfitting_analysis.png and .json")
    
    return analysis_results

if __name__ == "__main__":
    results = check_overfitting()
