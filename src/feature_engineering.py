import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def calculate_debt_recovery_efficiency(df):
    """
    Calculate debt recovery efficiency metrics on quarterly basis:
    - Days Sales Outstanding (DSO)
    - Cash collection ratio
    - Payment delay patterns
    """
    # Add quarter column
    df['quarter'] = df['transaction_date'].dt.to_period('Q')
    
    # Group by company and quarter for quarterly metrics
    quarterly_metrics = []
    
    for (company_id, quarter), quarter_data in df.groupby(['company_id', 'quarter']):
        quarter_data = quarter_data.sort_values('transaction_date')
        
        # Calculate DSO (Days Sales Outstanding) for the quarter
        # DSO = (Accounts Receivable / Total Sales) × Number of Days in Quarter
        total_sales = quarter_data['transaction_amount'].sum()
        avg_accounts_receivable = quarter_data['accounts_receivable'].mean()
        days_in_quarter = 90  # Standard quarter length
        dso = (avg_accounts_receivable / total_sales) * days_in_quarter if total_sales > 0 else 90
        
        # Cash collection ratio (how much cash collected vs revenue generated)
        total_cash_collected = total_sales - quarter_data['accounts_receivable'].iloc[-1]  # Assuming final AR is uncollected
        cash_collection_ratio = total_cash_collected / total_sales if total_sales > 0 else 0
        
        # Average payment delay for the quarter
        avg_payment_delay = quarter_data['payment_delay_days'].mean()
        
        # Percentage of immediate payments (0 days delay)
        immediate_payment_rate = (quarter_data['payment_delay_days'] == 0).mean()
        
        # Bad debt indicator (high payment delays)
        high_delay_rate = (quarter_data['payment_delay_days'] > 60).mean()
        
        quarterly_metrics.append({
            'company_id': company_id,
            'quarter': str(quarter),
            'dso_days': dso,
            'cash_collection_ratio': cash_collection_ratio,
            'avg_payment_delay_days': avg_payment_delay,
            'immediate_payment_rate': immediate_payment_rate,
            'high_delay_payment_rate': high_delay_rate
        })
    
    # Now aggregate quarterly data to company level (average across quarters)
    quarterly_df = pd.DataFrame(quarterly_metrics)
    company_metrics = quarterly_df.groupby('company_id').agg({
        'dso_days': 'mean',
        'cash_collection_ratio': 'mean', 
        'avg_payment_delay_days': 'mean',
        'immediate_payment_rate': 'mean',
        'high_delay_payment_rate': 'mean'
    }).reset_index()
    
    # Add quarterly trend analysis
    company_trends = []
    for company_id in quarterly_df['company_id'].unique():
        company_quarters = quarterly_df[quarterly_df['company_id'] == company_id].sort_values('quarter')
        
        # Calculate trends over quarters
        if len(company_quarters) > 1:
            dso_trend = np.polyfit(range(len(company_quarters)), company_quarters['dso_days'], 1)[0]
            collection_trend = np.polyfit(range(len(company_quarters)), company_quarters['cash_collection_ratio'], 1)[0]
        else:
            dso_trend = 0
            collection_trend = 0
            
        company_trends.append({
            'company_id': company_id,
            'dso_trend': dso_trend,  # Negative is better (improving DSO)
            'collection_ratio_trend': collection_trend,  # Positive is better (improving collection)
            'quarters_of_data': len(company_quarters)
        })
    
    trends_df = pd.DataFrame(company_trends)
    company_metrics = company_metrics.merge(trends_df, on='company_id')
    
    return company_metrics

def calculate_financial_liquidity(df):
    """
    Calculate financial liquidity metrics on quarterly basis:
    - Current ratio
    - Quick ratio
    - Net cash flow trends
    """
    # Add quarter column
    df['quarter'] = df['transaction_date'].dt.to_period('Q')
    
    # Group by company and quarter for quarterly metrics
    quarterly_metrics = []
    
    for (company_id, quarter), quarter_data in df.groupby(['company_id', 'quarter']):
        quarter_data = quarter_data.sort_values('transaction_date')
        
        # Current Ratio = Current Assets / Current Liabilities (end of quarter)
        end_quarter_current_assets = quarter_data['current_assets'].iloc[-1]
        end_quarter_current_liabilities = quarter_data['current_liabilities'].iloc[-1]
        current_ratio = end_quarter_current_assets / end_quarter_current_liabilities if end_quarter_current_liabilities > 0 else 0
        
        # Quick Ratio = (Current Assets - Inventory) / Current Liabilities (end of quarter)
        end_quarter_inventory = quarter_data['inventory_value'].iloc[-1]
        quick_assets = end_quarter_current_assets - end_quarter_inventory
        quick_ratio = quick_assets / end_quarter_current_liabilities if end_quarter_current_liabilities > 0 else 0
        
        # Net cash flow metrics for the quarter
        quarterly_net_cash_flow = quarter_data['net_cash_flow'].sum()
        avg_quarterly_cash_flow = quarter_data['net_cash_flow'].mean()
        cash_flow_volatility = quarter_data['net_cash_flow'].std()
        
        # Liquidity stress indicator for the quarter
        negative_cash_flow_periods = (quarter_data['net_cash_flow'] < 0).sum()
        liquidity_stress_ratio = negative_cash_flow_periods / len(quarter_data)
        
        quarterly_metrics.append({
            'company_id': company_id,
            'quarter': str(quarter),
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'quarterly_net_cash_flow': quarterly_net_cash_flow,
            'avg_quarterly_cash_flow': avg_quarterly_cash_flow,
            'cash_flow_volatility': cash_flow_volatility if not pd.isna(cash_flow_volatility) else 0,
            'liquidity_stress_ratio': liquidity_stress_ratio
        })
    
    # Aggregate quarterly data to company level
    quarterly_df = pd.DataFrame(quarterly_metrics)
    company_metrics = quarterly_df.groupby('company_id').agg({
        'current_ratio': 'mean',
        'quick_ratio': 'mean',
        'quarterly_net_cash_flow': 'sum',  # Total across all quarters
        'avg_quarterly_cash_flow': 'mean',
        'cash_flow_volatility': 'mean',
        'liquidity_stress_ratio': 'mean'
    }).reset_index()
    
    # Rename for clarity
    company_metrics.rename(columns={
        'quarterly_net_cash_flow': 'total_net_cash_flow',
        'avg_quarterly_cash_flow': 'avg_net_cash_flow'
    }, inplace=True)
    
    # Add quarterly trend analysis
    company_trends = []
    for company_id in quarterly_df['company_id'].unique():
        company_quarters = quarterly_df[quarterly_df['company_id'] == company_id].sort_values('quarter')
        
        # Calculate trends over quarters
        if len(company_quarters) > 1:
            cash_flow_trend = np.polyfit(range(len(company_quarters)), company_quarters['quarterly_net_cash_flow'], 1)[0]
            liquidity_trend = np.polyfit(range(len(company_quarters)), company_quarters['current_ratio'], 1)[0]
        else:
            cash_flow_trend = 0
            liquidity_trend = 0
            
        company_trends.append({
            'company_id': company_id,
            'cash_flow_trend': cash_flow_trend,  # Positive is better (improving cash flow)
            'liquidity_trend': liquidity_trend,  # Positive is better (improving liquidity)
        })
    
    trends_df = pd.DataFrame(company_trends)
    company_metrics = company_metrics.merge(trends_df, on='company_id')
    
    return company_metrics

def calculate_cash_flow_synchronization(df):
    """
    Calculate cash flow synchronization metrics on quarterly basis:
    - Ratio of inflows to outflows
    - Cash flow timing alignment
    """
    # Add quarter column
    df['quarter'] = df['transaction_date'].dt.to_period('Q')
    
    # Group by company and quarter for quarterly metrics
    quarterly_metrics = []
    
    for (company_id, quarter), quarter_data in df.groupby(['company_id', 'quarter']):
        # Calculate inflows and outflows for the quarter
        total_inflows = quarter_data['transaction_amount'].sum()
        total_outflows = (quarter_data['operating_expense'] + 
                         quarter_data['fixed_costs'] + 
                         quarter_data['debt_payment']).sum()
        
        # Inflow to outflow ratio for the quarter
        inflow_outflow_ratio = total_inflows / total_outflows if total_outflows > 0 else 0
        
        # Cash flow coverage ratio (ability to cover expenses) for the quarter
        expense_coverage_ratio = total_inflows / total_outflows if total_outflows > 0 else 0
        
        quarterly_metrics.append({
            'company_id': company_id,
            'quarter': str(quarter),
            'inflow_outflow_ratio': inflow_outflow_ratio,
            'expense_coverage_ratio': expense_coverage_ratio,
            'quarterly_inflows': total_inflows,
            'quarterly_outflows': total_outflows
        })
    
    # Aggregate quarterly data to company level
    quarterly_df = pd.DataFrame(quarterly_metrics)
    company_metrics = quarterly_df.groupby('company_id').agg({
        'inflow_outflow_ratio': 'mean',
        'expense_coverage_ratio': 'mean',
        'quarterly_inflows': 'sum',
        'quarterly_outflows': 'sum'
    }).reset_index()
    
    # Calculate overall synchronization metrics
    company_sync_metrics = []
    for company_id in quarterly_df['company_id'].unique():
        company_quarters = quarterly_df[quarterly_df['company_id'] == company_id].sort_values('quarter')
        
        # Cash flow synchronization score (consistency of quarterly ratios)
        if len(company_quarters) > 1:
            ratio_consistency = 1 - (company_quarters['inflow_outflow_ratio'].std() / 
                                   company_quarters['inflow_outflow_ratio'].mean()) if company_quarters['inflow_outflow_ratio'].mean() > 0 else 0
            
            # Trend in cash flow synchronization
            sync_trend = np.polyfit(range(len(company_quarters)), company_quarters['inflow_outflow_ratio'], 1)[0]
        else:
            ratio_consistency = 1.0  # Perfect consistency if only one quarter
            sync_trend = 0
        
        company_sync_metrics.append({
            'company_id': company_id,
            'cash_flow_synchronization': ratio_consistency,
            'sync_trend': sync_trend  # Positive trend is better
        })
    
    sync_df = pd.DataFrame(company_sync_metrics)
    company_metrics = company_metrics.merge(sync_df, on='company_id')
    
    return company_metrics

def calculate_sales_performance_features(df):
    """
    Calculate sales performance metrics on quarterly basis:
    - Revenue trends
    - Sales consistency
    - Revenue vs receipts lag
    """
    # Add quarter column
    df['quarter'] = df['transaction_date'].dt.to_period('Q')
    
    # Group by company and quarter for quarterly metrics
    quarterly_metrics = []
    
    for (company_id, quarter), quarter_data in df.groupby(['company_id', 'quarter']):
        quarter_data = quarter_data.sort_values('transaction_date')
        
        # Quarterly revenue
        quarterly_revenue = quarter_data['transaction_amount'].sum()
        
        # Average payment delay for the quarter
        quarterly_avg_delay = quarter_data['payment_delay_days'].mean()
        
        # Sales vs receipts lag (weighted average delay) for the quarter
        weighted_avg_delay = (quarter_data['transaction_amount'] * quarter_data['payment_delay_days']).sum() / quarter_data['transaction_amount'].sum()
        
        # Number of transactions in the quarter
        transaction_count = len(quarter_data)
        
        quarterly_metrics.append({
            'company_id': company_id,
            'quarter': str(quarter),
            'quarterly_revenue': quarterly_revenue,
            'quarterly_avg_delay': quarterly_avg_delay,
            'sales_receipts_lag': weighted_avg_delay,
            'transaction_count': transaction_count
        })
    
    # Aggregate quarterly data to company level
    quarterly_df = pd.DataFrame(quarterly_metrics)
    
    company_metrics = []
    
    for company_id in quarterly_df['company_id'].unique():
        company_quarters = quarterly_df[quarterly_df['company_id'] == company_id].sort_values('quarter')
        
        # Average quarterly revenue
        avg_quarterly_revenue = company_quarters['quarterly_revenue'].mean()
        
        # Revenue trend across quarters
        if len(company_quarters) > 1:
            revenue_trend = np.polyfit(range(len(company_quarters)), company_quarters['quarterly_revenue'], 1)[0]
        else:
            revenue_trend = 0
        
        # Revenue consistency (coefficient of variation across quarters)
        revenue_cv = (company_quarters['quarterly_revenue'].std() / 
                     company_quarters['quarterly_revenue'].mean()) if company_quarters['quarterly_revenue'].mean() > 0 else 0
        revenue_consistency = 1 / (1 + revenue_cv)  # Higher is better
        
        # Average sales receipts lag across quarters
        avg_sales_receipts_lag = company_quarters['sales_receipts_lag'].mean()
        
        # Revenue growth rate (quarter over quarter)
        if len(company_quarters) > 1:
            revenue_growth_rate = ((company_quarters['quarterly_revenue'].iloc[-1] - 
                                  company_quarters['quarterly_revenue'].iloc[0]) / 
                                 company_quarters['quarterly_revenue'].iloc[0]) if company_quarters['quarterly_revenue'].iloc[0] > 0 else 0
        else:
            revenue_growth_rate = 0
        
        company_metrics.append({
            'company_id': company_id,
            'avg_quarterly_revenue': avg_quarterly_revenue,
            'revenue_trend': revenue_trend,
            'revenue_consistency': revenue_consistency,
            'sales_receipts_lag': avg_sales_receipts_lag,
            'revenue_growth_rate': revenue_growth_rate,
            'quarters_tracked': len(company_quarters)
        })
    
    return pd.DataFrame(company_metrics)

def create_risk_scoring(df):
    """
    Create risk scoring based on quarterly financial metrics
    """
    # Calculate individual metric scores (0-100 scale, higher is better)
    df = df.copy()
    
    # Debt Recovery Efficiency Score (30% weight)
    df['debt_recovery_score'] = (
        (100 - np.clip(df['dso_days'], 0, 90) / 90 * 100) * 0.3 +  # Lower DSO is better (quarterly max 90 days)
        (df['cash_collection_ratio'] * 100) * 0.3 +  # Higher collection ratio is better
        (df['immediate_payment_rate'] * 100) * 0.2 +  # Higher immediate payment rate is better
        ((df['collection_ratio_trend'] + 0.1) / 0.2 * 100).clip(0, 100) * 0.2  # Improving collection trend
    )
    
    # Financial Liquidity Score (35% weight)
    df['liquidity_score'] = (
        np.clip(df['current_ratio'] / 2 * 100, 0, 100) * 0.25 +  # Target current ratio of 2
        np.clip(df['quick_ratio'] / 1.5 * 100, 0, 100) * 0.25 +  # Target quick ratio of 1.5
        np.clip(df['avg_net_cash_flow'] / 15000 * 100, 0, 100) * 0.2 +  # Positive cash flow (quarterly target)
        (100 - df['liquidity_stress_ratio'] * 100) * 0.15 +  # Lower stress is better
        ((df['cash_flow_trend'] / 10000 + 1) / 2 * 100).clip(0, 100) * 0.15  # Improving cash flow trend
    )
    
    # Cash Flow Synchronization Score (20% weight)
    df['cash_flow_sync_score'] = (
        np.clip(df['inflow_outflow_ratio'] / 1.2 * 100, 0, 100) * 0.4 +  # Target ratio > 1.2
        (df['cash_flow_synchronization'] * 100) * 0.3 +  # Higher synchronization is better
        np.clip(df['expense_coverage_ratio'] / 1.3 * 100, 0, 100) * 0.3  # Target coverage > 1.3
    )
    
    # Sales Performance Score (15% weight)
    df['sales_performance_score'] = (
        np.clip(df['revenue_trend'] / 15000 * 100 + 50, 0, 100) * 0.25 +  # Positive quarterly trend
        (df['revenue_consistency'] * 100) * 0.3 +  # Higher consistency
        (100 - np.clip(df['sales_receipts_lag'] / 90 * 100, 0, 100)) * 0.25 +  # Lower lag is better
        np.clip((df['revenue_growth_rate'] + 0.2) / 0.4 * 100, 0, 100) * 0.2  # Positive growth rate
    )
    
    # Overall Risk Score (weighted average)
    df['overall_risk_score'] = (
        df['debt_recovery_score'] * 0.30 +
        df['liquidity_score'] * 0.35 +
        df['cash_flow_sync_score'] * 0.20 +
        df['sales_performance_score'] * 0.15
    )
    
    # Risk Categories (adjusted for quarterly assessment)
    df['risk_category'] = pd.cut(df['overall_risk_score'], 
                                bins=[0, 45, 65, 80, 100], 
                                labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'])
    
    # Loan Eligibility (binary classification) - slightly adjusted threshold for quarterly data
    df['loan_eligible'] = (df['overall_risk_score'] >= 65).astype(int)
    
    return df

def engineer_features():
    """
    Main feature engineering function
    """
    print("Loading cleaned transaction data...")
    df = pd.read_csv('/Users/zayed/loan_eligibility_ml/data/processed/malaysian_sme_transactions.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['payment_date'] = pd.to_datetime(df['payment_date'])
    
    print(f"Processing {len(df)} transactions for {df['company_id'].nunique()} companies...")
    
    # Calculate all feature groups
    print("Calculating debt recovery efficiency features...")
    debt_recovery_features = calculate_debt_recovery_efficiency(df)
    
    print("Calculating financial liquidity features...")
    liquidity_features = calculate_financial_liquidity(df)
    
    print("Calculating cash flow synchronization features...")
    cash_flow_features = calculate_cash_flow_synchronization(df)
    
    print("Calculating sales performance features...")
    sales_features = calculate_sales_performance_features(df)
    
    # Merge all features
    print("Merging all features...")
    features_df = debt_recovery_features.merge(liquidity_features, on='company_id')
    features_df = features_df.merge(cash_flow_features, on='company_id')
    features_df = features_df.merge(sales_features, on='company_id')
    
    # Add company metadata
    company_metadata = df.groupby('company_id').agg({
        'sector': 'first',
        'company_age_years': 'first',
        'employee_count': 'first'
    }).reset_index()
    
    features_df = features_df.merge(company_metadata, on='company_id')
    
    # Create risk scoring
    print("Creating risk scoring...")
    features_df = create_risk_scoring(features_df)
    
    # Keep all 2000 companies - no sampling needed
    print(f"\nFeature engineering complete!")
    print(f"Final dataset: {len(features_df)} companies with {len(features_df.columns)} features")
    print(f"Risk distribution:")
    print(features_df['risk_category'].value_counts())
    print(f"Loan eligibility: {features_df['loan_eligible'].sum()}/{len(features_df)} companies eligible")
    
    return features_df

def save_engineered_features(df, output_path):
    """
    Save engineered features to processed folder
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nEngineered features saved to: {output_path}")
    
    # Save feature descriptions
    feature_descriptions = {
        'Debt Recovery Efficiency (Quarterly Assessment)': [
            'dso_days: Days Sales Outstanding - average time to collect receivables (quarterly basis)',
            'cash_collection_ratio: Ratio of cash collected to total sales (quarterly average)',
            'avg_payment_delay_days: Average payment delay across all transactions (quarterly average)',
            'immediate_payment_rate: Percentage of payments received immediately (quarterly average)',
            'high_delay_payment_rate: Percentage of payments delayed > 60 days (quarterly average)',
            'dso_trend: Trend in DSO across quarters (negative = improving)',
            'collection_ratio_trend: Trend in collection ratio across quarters (positive = improving)',
            'quarters_of_data: Number of quarters with data'
        ],
        'Financial Liquidity (Quarterly Assessment)': [
            'current_ratio: Current Assets / Current Liabilities (quarterly average)',
            'quick_ratio: (Current Assets - Inventory) / Current Liabilities (quarterly average)',
            'avg_net_cash_flow: Average net cash flow per quarter',
            'total_net_cash_flow: Total net cash flow across all quarters',
            'cash_flow_volatility: Standard deviation of net cash flows (quarterly average)',
            'cash_flow_trend: Trend direction of cash flows across quarters (positive = improving)',
            'liquidity_trend: Trend in current ratio across quarters (positive = improving)',
            'liquidity_stress_ratio: Proportion of periods with negative cash flow (quarterly average)'
        ],
        'Cash Flow Synchronization (Quarterly Assessment)': [
            'inflow_outflow_ratio: Total inflows / Total outflows (quarterly average)',
            'cash_flow_synchronization: Consistency of quarterly inflow/outflow ratios',
            'expense_coverage_ratio: Ability to cover expenses with inflows (quarterly average)',
            'sync_trend: Trend in synchronization across quarters (positive = improving)',
            'quarterly_inflows: Total inflows across all quarters',
            'quarterly_outflows: Total outflows across all quarters'
        ],
        'Sales Performance (Quarterly Assessment)': [
            'avg_quarterly_revenue: Average revenue per quarter',
            'revenue_trend: Trend direction of revenue across quarters (positive = growing)',
            'revenue_consistency: Inverse of coefficient of variation across quarters (higher = more consistent)',
            'sales_receipts_lag: Weighted average delay between sales and payment (quarterly average)',
            'revenue_growth_rate: Overall revenue growth rate from first to last quarter',
            'quarters_tracked: Number of quarters with revenue data'
        ],
        'Risk Scoring (Quarterly-Based)': [
            'debt_recovery_score: Score based on quarterly debt recovery efficiency (0-100)',
            'liquidity_score: Score based on quarterly financial liquidity (0-100)',
            'cash_flow_sync_score: Score based on quarterly cash flow synchronization (0-100)',
            'sales_performance_score: Score based on quarterly sales performance (0-100)',
            'overall_risk_score: Weighted average of all quarterly scores (0-100)',
            'risk_category: Categorical risk level based on quarterly assessment (High/Medium/Low/Very Low Risk)',
            'loan_eligible: Binary loan eligibility based on quarterly performance (1=eligible, 0=not eligible)'
        ]
    }
    
    desc_path = output_path.replace('.csv', '_feature_descriptions.txt')
    with open(desc_path, 'w') as f:
        f.write("Quarterly Financial Feature Engineering - Feature Descriptions\n")
        f.write("=" * 65 + "\n\n")
        f.write("This analysis aggregates transaction data into quarterly assessments\n")
        f.write("to provide more stable and meaningful financial metrics for loan eligibility.\n\n")
        f.write("Each quarter represents a 3-month period, allowing for:\n")
        f.write("- More stable financial ratio calculations\n")
        f.write("- Better trend analysis across quarters\n")
        f.write("- Reduced noise from daily transaction variations\n")
        f.write("- More realistic business cycle assessment\n\n")
        
        for category, features in feature_descriptions.items():
            f.write(f"{category}:\n")
            f.write("-" * len(category) + "\n")
            for feature in features:
                f.write(f"  • {feature}\n")
            f.write("\n")
    
    print(f"Feature descriptions saved to: {desc_path}")

if __name__ == "__main__":
    # Engineer features
    features_df = engineer_features()
    
    # Save engineered features
    output_path = "/Users/zayed/loan_eligibility_ml/data/processed/sme_financial_features.csv"
    save_engineered_features(features_df, output_path)
