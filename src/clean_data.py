import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_malaysian_sme_transaction_data(n_companies=10000, transactions_per_company=20):
    """
    Generate realistic Malaysian SME transaction data with standardized banking details
    targeting RM50,000 monthly revenue per company for loan eligibility assessment.
    """
    np.random.seed(42)
    
    # Generate transaction-level data for each company
    all_transactions = []
    
    # Company characteristics
    company_ids = [f"SME_{str(i).zfill(4)}" for i in range(1, n_companies + 1)]
    
    # Business sectors in Malaysia - Only Food and Beverages
    sectors = ['Food and Beverages']
    sector_weights = [1.0]
    
    # Standardized banking details as requested
    rfi_entity_id = "bank of sarawak"
    rfi_bank_code = "BS79334832456" 
    trxn_channel = "duitnow qr"
    trxn_type = "qr pay"
    
    for company_idx in range(n_companies):
        company_id = company_ids[company_idx]
        sector = np.random.choice(sectors, p=sector_weights)
        
        # Company characteristics
        company_age_years = max(1, np.random.exponential(5) + 1)  # 1-20 years typically
        company_age_years = min(company_age_years, 25)
        employee_count = max(5, int(np.random.lognormal(2.5, 0.8)))
        employee_count = min(employee_count, 200)
        
        # Target RM50,000 monthly revenue with sector variations
        sector_multiplier = {
            'Food and Beverages': 1.0
        }[sector]
        
        base_monthly_revenue = 50000 * sector_multiplier
        
        # Generate transactions over 12 months
        for month in range(1, 13):
            # Seasonal variation
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
            monthly_revenue_target = base_monthly_revenue * seasonal_factor
            
            # Generate individual transactions for this month
            monthly_transactions = max(20, np.random.poisson(transactions_per_company / 12))
            
            # Transaction amounts that sum to target revenue
            transaction_amounts = np.random.exponential(monthly_revenue_target / monthly_transactions, monthly_transactions)
            transaction_amounts = transaction_amounts * (monthly_revenue_target / transaction_amounts.sum())
            
            for txn_idx, amount in enumerate(transaction_amounts):
                # Transaction timing within the month
                transaction_date = datetime(2024, month, 1) + timedelta(
                    days=np.random.randint(0, 28),
                    hours=np.random.randint(8, 20),
                    minutes=np.random.randint(0, 60)
                )
                
                # Payment collection timing (affects liquidity)
                payment_delay_days = np.random.choice([0, 30, 60, 90], p=[0.4, 0.3, 0.2, 0.1])
                payment_date = transaction_date + timedelta(days=int(payment_delay_days))
                
                # Generate operating expenses for this transaction
                expense_ratio = np.random.uniform(0.3, 0.7)  # 30-70% of revenue as expenses
                operating_expense = amount * expense_ratio
                
                # Fixed costs allocation per transaction
                fixed_cost_per_txn = (employee_count * np.random.uniform(3000, 5000)) / monthly_transactions
                
                # Debt payment (if applicable)
                has_debt = np.random.choice([True, False], p=[0.7, 0.3])
                debt_payment = np.random.uniform(100, 1000) if has_debt else 0
                
                # Calculate net cash flow for this transaction
                net_cash_flow = amount - operating_expense - fixed_cost_per_txn - debt_payment
                
                # Current assets and liabilities (monthly snapshots)
                accounts_receivable = amount if payment_delay_days > 0 else 0
                inventory_value = amount * np.random.uniform(0.1, 0.3)
                cash_on_hand = max(1000, net_cash_flow + np.random.uniform(-500, 2000))
                
                current_assets = cash_on_hand + accounts_receivable + inventory_value
                
                accounts_payable = operating_expense * np.random.uniform(0.2, 0.8)
                short_term_debt = debt_payment * np.random.uniform(1, 3) if has_debt else 0
                current_liabilities = accounts_payable + short_term_debt
                
                transaction_record = {
                    # Company identifiers
                    'company_id': company_id,
                    'sector': sector,
                    'company_age_years': company_age_years,
                    'employee_count': employee_count,
                    
                    # Standardized banking details
                    'rfi_entity_id': rfi_entity_id,
                    'rfi_bank_code': rfi_bank_code,
                    'trxn_channel': trxn_channel,
                    'trxn_type': trxn_type,
                    
                    # Transaction details
                    'transaction_id': f"{company_id}_TXN_{month:02d}_{txn_idx:03d}",
                    'transaction_date': transaction_date,
                    'payment_date': payment_date,
                    'transaction_amount': round(amount, 2),
                    'payment_delay_days': payment_delay_days,
                    
                    # Financial metrics
                    'operating_expense': round(operating_expense, 2),
                    'fixed_costs': round(fixed_cost_per_txn, 2),
                    'debt_payment': round(debt_payment, 2),
                    'net_cash_flow': round(net_cash_flow, 2),
                    
                    # Balance sheet items
                    'cash_on_hand': round(cash_on_hand, 2),
                    'accounts_receivable': round(accounts_receivable, 2),
                    'inventory_value': round(inventory_value, 2),
                    'current_assets': round(current_assets, 2),
                    'accounts_payable': round(accounts_payable, 2),
                    'short_term_debt': round(short_term_debt, 2),
                    'current_liabilities': round(current_liabilities, 2),
                    
                    # Time period
                    'month': month,
                    'year': 2024,
                    'has_existing_debt': has_debt
                }
                
                all_transactions.append(transaction_record)
    
    return pd.DataFrame(all_transactions)

def clean_and_prepare_data():
    """
    Clean the synthetic transaction data and prepare it for feature engineering
    """
    print("Generating synthetic Malaysian SME transaction data...")
    # Generate transactions from 10,000 companies (20 transactions per company on average)
    df = generate_malaysian_sme_transaction_data(n_companies=10000, transactions_per_company=20)
    
    # Take only a sample if the dataset is too large (keep more for 10,000 companies)
    if len(df) > 200000:  # Allow up to 200k transactions for 10,000 companies
        df = df.sample(n=200000, random_state=42).reset_index(drop=True)
    
    print(f"Generated {len(df)} transaction records for {df['company_id'].nunique()} companies")
    
    # Basic data cleaning
    financial_cols = ['transaction_amount', 'net_cash_flow', 'current_assets', 'current_liabilities']
    for col in financial_cols:
        if col in df.columns and (df[col] < 0).any():
            print(f"Warning: Found negative values in {col}")
            # Only fix negative values for columns that shouldn't be negative
            if col in ['transaction_amount', 'current_assets', 'current_liabilities']:
                df[col] = df[col].abs()
    
    # Remove extreme outliers
    for col in ['transaction_amount', 'net_cash_flow']:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            df[col] = df[col].clip(lower=q01, upper=q99)
    
    # Sort by company and date
    df = df.sort_values(['company_id', 'transaction_date']).reset_index(drop=True)
    
    print("\nData summary:")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Average transaction amount: RM{df['transaction_amount'].mean():,.2f}")
    print(f"Transaction amount range: RM{df['transaction_amount'].min():,.2f} - RM{df['transaction_amount'].max():,.2f}")
    print(f"Banking details standardized:")
    print(f"  RFI Entity: {df['rfi_entity_id'].iloc[0]}")
    print(f"  Bank Code: {df['rfi_bank_code'].iloc[0]}")
    print(f"  Channel: {df['trxn_channel'].iloc[0]}")
    print(f"  Type: {df['trxn_type'].iloc[0]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned data to processed folder
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    # Save summary statistics
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Malaysian SME Transaction Data Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total transaction records: {len(df)}\n")
        f.write(f"Unique companies: {df['company_id'].nunique()}\n")
        f.write(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}\n\n")
        
        f.write("Banking Details (Standardized):\n")
        f.write("-" * 30 + "\n")
        f.write(f"RFI Entity ID: {df['rfi_entity_id'].iloc[0]}\n")
        f.write(f"RFI Bank Code: {df['rfi_bank_code'].iloc[0]}\n")
        f.write(f"Transaction Channel: {df['trxn_channel'].iloc[0]}\n")
        f.write(f"Transaction Type: {df['trxn_type'].iloc[0]}\n\n")
        
        f.write("Financial Metrics Summary:\n")
        f.write("-" * 25 + "\n")
        for col in ['transaction_amount', 'net_cash_flow', 'current_assets', 'current_liabilities']:
            f.write(f"{col}:\n")
            f.write(f"  Mean: RM{df[col].mean():,.2f}\n")
            f.write(f"  Median: RM{df[col].median():,.2f}\n")
            f.write(f"  Std: RM{df[col].std():,.2f}\n\n")
    
    print(f"Summary statistics saved to: {summary_path}")

if __name__ == "__main__":
    # Clean and prepare the data
    df_cleaned = clean_and_prepare_data()
    
    # Save to processed folder
    output_path = "/Users/zayed/loan_eligibility_ml/data/processed/malaysian_sme_transactions.csv"
    save_cleaned_data(df_cleaned, output_path)