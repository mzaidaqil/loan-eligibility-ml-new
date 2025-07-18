# src/risk_labeling.py

import pandas as pd

def label_risk(input_path="data/processed/synthetic_data.xlsx", output_path="data/processed/labeled_data.xlsx"):
    df = pd.read_excel(input_path)

    risk_labels = []

    for _, row in df.iterrows():
        dsr = row['DSR']
        volatility = row['Volatility']

        if dsr > 1.8 and volatility < 0.15:
            risk_labels.append("Safe")
        elif dsr < 0.8 or volatility > 0.5:
            risk_labels.append("High Risk")
        else:
            risk_labels.append("Moderate")


    df['Risk_Category'] = risk_labels

    df.to_excel(output_path, index=False)
    print(f"Labeled data saved to {output_path}")
    
    # Check distribution
    print("\nRisk Category Distribution:")
    print(df['Risk_Category'].value_counts())
    print("\nRisk Category Percentages:")
    print(df['Risk_Category'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    label_risk()
