# src/data_generation.py

import numpy as np
import pandas as pd

def generate_synthetic_data(n_businesses=1000, n_months=24, random_seed=42):
    np.random.seed(random_seed)

    # Allocate proportions
    n_safe = int(n_businesses * 0.6)
    n_moderate = int(n_businesses * 0.3)
    n_high_risk = n_businesses - n_safe - n_moderate

    # Safe: Normal distributions
    safe_sales = np.random.normal(loc=50000, scale=10000, size=(n_safe, n_months)).clip(5000, 150000)
    safe_expenses = np.random.normal(loc=30000, scale=8000, size=(n_safe, n_months)).clip(3000, 100000)

    # Moderate: Slightly more unstable
    mod_sales = np.random.normal(loc=40000, scale=15000, size=(n_moderate, n_months)).clip(5000, 150000)
    mod_expenses = np.random.normal(loc=32000, scale=10000, size=(n_moderate, n_months)).clip(3000, 100000)

    # High Risk: High volatility, high expenses
    high_sales = np.random.normal(loc=30000, scale=25000, size=(n_high_risk, n_months)).clip(5000, 150000)
    high_expenses = np.random.normal(loc=35000, scale=20000, size=(n_high_risk, n_months)).clip(3000, 150000)

    # Concatenate all groups
    monthly_sales = np.vstack([safe_sales, mod_sales, high_sales])
    monthly_expenses = np.vstack([safe_expenses, mod_expenses, high_expenses])

    # Average and std deviation of sales for volatility
    avg_monthly_sales = np.mean(monthly_sales, axis=1)
    avg_monthly_expenses = np.mean(monthly_expenses, axis=1)
    volatility = np.std(monthly_sales, axis=1) / avg_monthly_sales

    # Generate existing debt and loan request with risk-appropriate ranges
    safe_debt = np.random.uniform(10000, 60000, n_safe)
    safe_loan = np.random.uniform(5000, 30000, n_safe)
    
    mod_debt = np.random.uniform(20000, 80000, n_moderate)
    mod_loan = np.random.uniform(10000, 40000, n_moderate)
    
    high_debt = np.random.uniform(40000, 120000, n_high_risk)
    high_loan = np.random.uniform(15000, 60000, n_high_risk)
    
    existing_debt = np.concatenate([safe_debt, mod_debt, high_debt])
    loan_request = np.concatenate([safe_loan, mod_loan, high_loan])

    # Net Operating Cash Flow
    net_operating_cash_flow = avg_monthly_sales - avg_monthly_expenses

    # Debt Service Ratio
    dsr = net_operating_cash_flow / ((existing_debt + loan_request) / n_months)

    # Create DataFrame
    df = pd.DataFrame({
        'Avg_Monthly_Sales': avg_monthly_sales,
        'Avg_Monthly_Expenses': avg_monthly_expenses,
        'Volatility': volatility,
        'Existing_Debt': existing_debt,
        'Loan_Request': loan_request,
        'Net_Operating_Cash_Flow': net_operating_cash_flow,
        'DSR': dsr
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_excel("data/processed/synthetic_data.xlsx", index=False)
    print("Synthetic data generated and saved to data/processed/synthetic_data.xlsx")
