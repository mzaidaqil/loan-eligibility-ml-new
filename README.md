# 🏦 Loan Eligibility ML for Malaysian SME Companies

A comprehensive machine learning system for assessing loan eligibility of Malaysian Small and Medium Enterprises (SMEs) in the Food and Beverages sector, deployed as a production-ready FastAPI service on Render.

## 📊 Project Overview

This project develops a sophisticated loan eligibility assessment system specifically designed for Malaysian SMEs, using quarterly financial performance data to predict creditworthiness. The system analyzes multiple financial dimensions to provide accurate, data-driven loan eligibility decisions.

### 🎯 Business Problem
Traditional loan assessment methods for SMEs often rely on limited financial statements and subjective evaluations. This system provides:
- **Objective assessment** based on real transaction data
- **Quarterly performance tracking** for better trend analysis
- **Multi-dimensional risk evaluation** across key financial metrics
- **Automated decision making** to reduce processing time and bias

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │ -> │ Feature Engine   │ -> │   ML Models     │
│ (Transactions)  │    │ (Quarterly Agg.) │    │ (Classification)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Cleaning   │    │ Risk Scoring     │    │   FastAPI       │
│ & Validation    │    │ & Labeling       │    │  Web Service    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📈 Machine Learning Pipeline

### 1. **Data Generation & Processing**
- **10,000 synthetic Malaysian SME companies** in Food & Beverages sector
- **240,000+ transaction records** with realistic Malaysian business patterns
- **Quarterly aggregation** for seasonal trend analysis
- **Comprehensive financial metrics** including liquidity, cash flow, and debt recovery

### 2. **Feature Engineering Framework**

#### 🔄 Debt Recovery Efficiency (30% weight)
- **Days Sales Outstanding (DSO)**: Average collection time per quarter
- **Cash Collection Ratio**: Efficiency of cash collection vs sales
- **Payment Delay Analysis**: Distribution of payment delays
- **Trend Analysis**: Quarter-over-quarter improvement tracking

#### 💰 Financial Liquidity (35% weight)
- **Current Ratio**: Short-term financial health (target: >2.0)
- **Quick Ratio**: Immediate liquidity position (target: >1.5)
- **Cash Flow Metrics**: Net cash flow trends and volatility
- **Liquidity Stress**: Periods of negative cash flow

#### ⚖️ Cash Flow Synchronization (20% weight)
- **Inflow/Outflow Ratio**: Balance of income vs expenses
- **Expense Coverage**: Ability to cover operational costs
- **Flow Consistency**: Stability of quarterly cash flows
- **Synchronization Score**: Alignment of inflows and outflows

#### 📊 Sales Performance (15% weight)
- **Revenue Trends**: Quarter-over-quarter growth patterns
- **Revenue Consistency**: Stability of sales performance
- **Sales-to-Receipt Lag**: Time between sale and payment
- **Growth Rate**: Overall revenue trajectory

### 3. **Risk Scoring & Classification**

#### Risk Assessment Framework:
```
Overall Risk Score = (Debt Recovery × 0.30) + 
                    (Liquidity × 0.35) + 
                    (Cash Flow Sync × 0.20) + 
                    (Sales Performance × 0.15)
```

#### Risk Categories:
- **Very Low Risk** (80-100): Premium borrowers
- **Low Risk** (65-79): Good borrowers (✅ **Loan Eligible**)
- **Medium Risk** (45-64): Requires additional evaluation
- **High Risk** (0-44): Not recommended for lending

### 4. **Model Training & Evaluation**

#### Models Tested:
| Model | Accuracy | AUC Score | CV AUC | Status |
|-------|----------|-----------|--------|--------|
| **Gradient Boosting** | **94.5%** | **0.986** | **0.984** | ✅ **Selected** |
| Random Forest | 92.5% | 0.974 | 0.977 | Backup |
| Logistic Regression | 91.3% | 0.978 | 0.978 | Baseline |

#### Key Performance Metrics:
- **98.6% AUC Score**: Excellent discrimination capability
- **94.5% Accuracy**: High prediction accuracy
- **Cross-Validation**: Consistent performance across folds
- **Overfitting Analysis**: Validated using learning curves

### 5. **Feature Importance Analysis**

Top risk indicators identified:
1. **Cash Collection Ratio** (18.2%): Most critical factor
2. **Current Ratio** (15.8%): Liquidity position
3. **DSO Days** (12.4%): Collection efficiency
4. **Revenue Trend** (11.7%): Growth trajectory
5. **Inflow/Outflow Ratio** (9.3%): Cash flow balance

## 🚀 Production Deployment

### FastAPI Web Service
- **Endpoint**: `https://loan-eligibility-ml-new.onrender.com`
- **Technology Stack**: FastAPI + Uvicorn
- **Deployment Platform**: Render (Free Tier)
- **Model Persistence**: Joblib serialization

### API Endpoints:
- `GET /` - Health check
- `GET /health` - Detailed system status
- `POST /predict` - Loan eligibility prediction
- `GET /model/info` - Model performance metrics

### Example API Usage:
```bash
curl -X POST "https://loan-eligibility-ml-new.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "SME_001",
    "dso_days": 25.0,
    "cash_collection_ratio": 0.92,
    "current_ratio": 2.1,
    "avg_quarterly_revenue": 200000.0,
    "revenue_trend": 0.08
  }'
```

### Response Format:
```json
{
  "company_id": "SME_001",
  "loan_eligible": true,
  "confidence": 0.94,
  "risk_level": "low",
  "probability_eligible": 0.94,
  "model_used": "Gradient Boosting"
}
```

## 📁 Project Structure

```
loan_eligibility_ml/
├── api/
│   └── app.py                    # FastAPI web service
├── data/
│   ├── raw/
│   │   └── malaysia_transactions.parquet
│   └── processed/
│       ├── malaysian_sme_transactions.csv
│       └── sme_financial_features.csv
├── models/
│   ├── loan_eligibility_model.pkl    # Best model (Gradient Boosting)
│   ├── feature_importance.png        # Feature analysis charts
│   └── model_evaluation.png          # Performance comparison
├── src/
│   ├── clean_data.py                 # Data generation & cleaning
│   ├── feature_engineering.py        # Quarterly feature calculation
│   ├── model_training.py             # ML pipeline & evaluation
│   └── validate_overfitting.py       # Model validation
├── tests/
│   └── test_model_prediction.py      # Model testing suite
├── requirements.txt                  # Production dependencies
├── render.yaml                       # Deployment configuration
└── README.md                         # This file
```

## 🔧 Technical Implementation

### Core Dependencies:
```python
pandas          # Data manipulation
numpy           # Numerical computations
scikit-learn    # Machine learning algorithms
joblib          # Model serialization
fastapi         # Web API framework
uvicorn         # ASGI server
pydantic        # Data validation
```

### Data Processing Pipeline:
1. **Data Generation**: Realistic SME transaction patterns
2. **Quarterly Aggregation**: Seasonal trend analysis
3. **Feature Engineering**: 25+ financial indicators
4. **Risk Scoring**: Multi-dimensional assessment
5. **Model Training**: Ensemble methods with cross-validation
6. **Validation**: Overfitting analysis and performance testing

### Model Features:
- **Quarterly Assessment**: Trend analysis across business quarters
- **Malaysian SME Focus**: Tailored for local business patterns
- **Robust Validation**: Cross-validation with overfitting checks
- **Production Ready**: Deployed API with error handling
- **Scalable Architecture**: Easily extensible for new features

## 📊 Business Impact

### Value Proposition:
- **Reduced Processing Time**: Automated assessment vs manual review
- **Objective Decision Making**: Data-driven vs subjective evaluation
- **Risk Mitigation**: Comprehensive financial health analysis
- **Scalability**: Handle thousands of applications efficiently

### Key Insights:
- **65% of SMEs qualify** for loans based on financial health
- **Cash collection efficiency** is the strongest predictor
- **Quarterly trends** provide better assessment than point-in-time
- **Multi-dimensional scoring** reduces false positives/negatives

## 🎯 Future Enhancements

1. **Multi-Sector Support**: Expand beyond Food & Beverages
2. **Real-time Data Integration**: Connect to banking APIs
3. **Advanced Models**: Deep learning for complex patterns
4. **Risk Monitoring**: Post-loan performance tracking
5. **Regulatory Compliance**: Bank Negara Malaysia guidelines

## 👥 Usage

### For Developers:
```bash
# Clone and setup
git clone https://github.com/mzaidaqil/loan-eligibility-ml-new.git
cd loan-eligibility-ml-new
pip install -r requirements.txt

# Run locally
uvicorn api.app:app --reload
```

### For Business Users:
- Access the deployed API at: `https://loan-eligibility-ml-new.onrender.com`
- Use POST `/predict` endpoint with company financial data
- Receive instant loan eligibility assessment with confidence scores

## 📈 Results Summary

✅ **94.5% Accuracy** in loan eligibility prediction  
✅ **98.6% AUC Score** for risk discrimination  
✅ **Production Deployed** FastAPI service on Render  
✅ **Comprehensive Feature Engineering** with 25+ financial indicators  
✅ **Robust Validation** with cross-validation and overfitting analysis  
✅ **Business-Ready** solution for Malaysian SME loan assessment  

---

*Built for Malaysian SME financial assessment | Deployed on Render | FastAPI + ML Pipeline*
