# src/model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model(input_path="data/processed/labeled_data.xlsx", model_output="models/loan_risk_classifier.pkl"):
    # Load the data
    df = pd.read_excel(input_path)

    # Features and target
    X = df[['Avg_Monthly_Sales', 'Avg_Monthly_Expenses', 'Volatility', 'Existing_Debt', 'Loan_Request', 'Net_Operating_Cash_Flow', 'DSR']]
    y = df['Risk_Category']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    joblib.dump(clf, model_output)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    train_model()
