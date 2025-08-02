# Business Financial Health Advisor Agent
**A Retrieval-Augmented Generation (RAG)–powered AI for Malaysian SME loan qualification and financial guidance**

## 1. Overview
The Business Financial Health Advisor is an intelligent agent designed to:
- **Assess quarterly financial performance** using your ML-powered loan eligibility system
- **Determine loan-readiness** based on comprehensive risk scoring (0-100 scale)
- **Provide tailored, actionable advice** to improve financial health across 4 key dimensions
- **Retrieve up-to-date** Malaysian banking policies and SME lending guidelines via RAG

**Integration**: Directly connects to your deployed ML API at `https://loan-eligibility-ml-new.onrender.com`

## 2. Key Capabilities

### **ML-Powered Assessment**
- **Real-time predictions** via your Gradient Boosting model (94.5% accuracy)
- **Multi-dimensional analysis** across Debt Recovery, Liquidity, Cash Flow, and Sales Performance
- **Quarterly trend evaluation** with confidence scoring
- **Risk categorization** (Very Low, Low, Medium, High Risk)

### **RAG-Enhanced Guidance**
- **Malaysian SME lending criteria** from major banks (Maybank, CIMB, Public Bank)
- **Bank Negara Malaysia guidelines** and regulatory updates
- **Industry benchmarks** for Food & Beverages sector
- **Real-time interest rates** and loan product information

### **Personalized Financial Advice**
- **Metric-specific recommendations** based on ML feature importance
- **Improvement roadmaps** with quarterly milestones
- **Bank-specific application strategies** tailored to lending criteria

## 3. Pre-Conversation Prompt
**User Intent Pop-Up:**
*"I want to assess my SME's loan eligibility using your ML system and get guidance for Malaysian bank applications."*

This pop-up initializes:
- **Load your ML model context** (feature importance, thresholds, performance metrics)
- **Activate quarterly assessment mode** for trend analysis
- **Enable Malaysian SME-specific** response templates

## 4. RAG Configuration (n8n)

### **Document Sources**
- **Your ML model documentation** (feature descriptions, performance reports)
- **Malaysian bank SME lending guides** (Maybank, CIMB, Public Bank, RHB)
- **Bank Negara Malaysia circulars** on SME financing
- **Food & Beverages industry benchmarks** and quarterly performance data

### **Embedding & Vector Store**
```yaml
n8n_workflow:
  1. HTTP_Request: 
     - Fetch bank lending policies
     - Pull BNM regulatory updates
     - Retrieve industry benchmarks
  
  2. ML_Integration:
     - Connect to your API endpoints
     - Cache model performance metrics
     - Store feature importance rankings
  
  3. Vector_Store:
     - OpenAI embeddings for document indexing
     - Pinecone/Weaviate for similarity search
     - Real-time policy updates
```

### **Query Flow**
```
User Financial Data → Your ML API → Risk Assessment → RAG Retrieval → Personalized Guidance
```

## 5. Enhanced Prompt Engineering

```yaml
system_message:
  - role: "system"
    content: |
      You are **BizFinHealthAdvisor**, an AI agent that analyzes Malaysian SME finances using a sophisticated 
      ML system with 94.5% accuracy. You integrate real-time predictions from the loan eligibility API with 
      comprehensive financial guidance tailored to Malaysian banking requirements.

  - role: "system" 
    content: |
      **ML System Integration:**
      - API Base: https://loan-eligibility-ml-new.onrender.com
      - Model: Gradient Boosting (98.6% AUC score)
      - Assessment Framework: 4 dimensions with weighted scoring
      - Eligibility Threshold: Risk Score ≥65 for loan approval

  - role: "system"
    content: |
      **Required Financial Metrics (Quarterly Basis):**
      1. Debt Recovery Efficiency (30%): DSO days, cash collection ratio, payment delays
      2. Financial Liquidity (35%): Current ratio, quick ratio, cash flow health
      3. Cash Flow Synchronization (20%): Inflow/outflow balance, expense coverage  
      4. Sales Performance (15%): Revenue trends, consistency, growth patterns

  - role: "system"
    content: |
      **Response Structure:**
      1. ML-Powered Risk Assessment (with confidence score)
      2. Dimensional Analysis (strengths across 4 categories)
      3. Improvement Recommendations (metric-specific targets)
      4. Malaysian Bank Application Strategy
      5. Quarterly Action Plan
```

## 6. Sample Interaction

**User (pre-pop-up):**
*"I want to assess my SME's loan eligibility using your ML system and get guidance for Malaysian bank applications."*

**User (provides quarterly data):**
```json
{
  "company_id": "SME_FOOD_001",
  "dso_days": 35.0,
  "cash_collection_ratio": 0.78,
  "avg_payment_delay_days": 22.0,
  "immediate_payment_rate": 0.55,
  "current_ratio": 1.3,
  "quick_ratio": 1.0,
  "avg_net_cash_flow": 8000.0,
  "inflow_outflow_ratio": 1.05,
  "avg_quarterly_revenue": 120000.0,
  "revenue_trend": -0.02,
  "company_age_years": 3.0,
  "employee_count": 18.0
}
```

### **Enhanced Agent Response**

**BizFinHealthAdvisor:**

## 🎯 **ML-Powered Risk Assessment**
**Overall Risk Score: 58/100** ⚠️ **MEDIUM RISK - NOT CURRENTLY ELIGIBLE**
- **Confidence Level**: 78%
- **Model Prediction**: Below eligibility threshold (≥65 required)
- **Risk Category**: Medium Risk - requires improvement before loan application

## 📊 **Dimensional Analysis**

### **Debt Recovery Efficiency (30% weight): 52/100** ❌
- **DSO Days**: 35 days (Target: <30) - *5 days above optimal*
- **Cash Collection Ratio**: 78% (Target: >85%) - *7% below target*
- **Payment Delays**: 22 days average - *within acceptable range*

### **Financial Liquidity (35% weight): 61/100** ⚠️
- **Current Ratio**: 1.3 (Target: >1.5) - *Needs improvement*
- **Quick Ratio**: 1.0 (Target: >1.2) - *Below recommended level*
- **Net Cash Flow**: RM8,000/quarter - *Positive but could be stronger*

### **Cash Flow Sync (20% weight): 58/100** ⚠️
- **Inflow/Outflow Ratio**: 1.05 (Target: >1.1) - *Tight cash management*
- **Flow Balance**: Marginal surplus - *Limited buffer for loan payments*

### **Sales Performance (15% weight): 64/100** ⚠️
- **Revenue Trend**: -2% decline - *Concerning negative trajectory*
- **Quarterly Revenue**: RM120,000 - *Moderate level for sector*

## 🎯 **Priority Improvement Recommendations**

### **Immediate Actions (Next Quarter):**
1. **Accelerate Collections** (Target: DSO ≤30 days)
   - Implement automated invoice reminders
   - Offer early payment discounts (2-3%)
   - Weekly collection call schedule

2. **Strengthen Liquidity** (Target: Current Ratio >1.5)
   - Build working capital reserves
   - Negotiate extended payment terms with suppliers
   - Consider short-term facilities for cash flow smoothing

### **3-Month Targets:**
- **DSO**: Reduce to 28-30 days
- **Cash Collection**: Improve to 85%+
- **Current Ratio**: Increase to 1.6+
- **Revenue Trend**: Achieve positive 2-5% growth

## 🏦 **Malaysian Bank Application Strategy**

### **Current Status**: Wait 1-2 quarters before applying
Your current risk score of 58 falls below most Malaysian banks' SME lending thresholds:

- **Maybank SME**: Typically requires risk score ≥65
- **CIMB BizChannel**: Similar threshold with additional collateral
- **Public Bank SME**: More conservative, prefers ≥70 risk score

### **Best Approach After Improvement:**
1. **Maybank SME Term Loan**: Most suitable after reaching 65+ score
2. **Working Capital Facility**: Consider for immediate cash flow needs
3. **Government Schemes**: Explore TEKUN or SME Bank programs

## 📋 **Quarterly Action Plan**

### **Month 1-2: Collection Optimization**
- [ ] Implement automated billing system
- [ ] Review and update customer payment terms
- [ ] Establish collection performance KPIs

### **Month 2-3: Liquidity Building**
- [ ] Reduce unnecessary expenses by 10-15%
- [ ] Negotiate supplier payment extensions
- [ ] Build emergency cash reserve (3 months expenses)

### **Month 3: Revenue Stabilization**
- [ ] Launch customer retention program
- [ ] Explore new revenue streams in F&B sector
- [ ] Implement monthly revenue forecasting

### **Quarter 2: Re-assessment**
- [ ] Run ML assessment again with improved metrics
- [ ] Target risk score ≥70 for strong loan position
- [ ] Prepare documentation for bank applications

## 💡 **Industry-Specific Insights (F&B Sector)**
Based on Malaysian F&B SME benchmarks:
- **Average DSO**: 28 days (you're above average)
- **Typical Cash Flow**: RM10,000-15,000/quarter for similar size
- **Revenue Volatility**: 15-20% seasonal variation expected

**Next Review**: Schedule ML re-assessment in 60-90 days to track improvement progress.

---
*Assessment powered by ML model with 94.5% accuracy | Updated quarterly for trend analysis*
