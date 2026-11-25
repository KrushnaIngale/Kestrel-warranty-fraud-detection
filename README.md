# Kestrel Home Appliances — Warranty Claim Fraud Detection System

An enterprise-grade, explainable machine learning system for detecting warranty claim fraud at Kestrel Home Appliances. Built for D2C Operations to assist human investigators without automated claim rejection.

---

## 1. Project Overview & Business Context

Kestrel Home Appliances (Pune, India) markets air fryers, mixer-grinders, water purifiers, robot vacuums, induction cooktops, ceiling fans, and room heaters. Claims originate via online channels and ~380 service partners across India.

### Key Operational Constraints (Operations Policy v4.1)
- **Investigation Desk Capacity:** At most **40 claims per month** can be reviewed manually.
- **Goodwill Penalty:** Delaying a genuine customer's claim incurs an average **₹380 goodwill credit**.
- **Fraud Impact:** Paying a fraudulent claim loses the full claimed amount (average confirmed fraud: **₹4,522**).
- **Auto-Approval Policy Shift:** On 1 May 2026, claims under ₹2,000 were auto-approved without inspection sign-offs, causing an influx of low-ticket fraud.
- **Decision Philosophy:** Decision support only; no automatic rejections.

---

## 2. System Architecture

```
                    ┌─────────────────────────┐
                    │ Raw Claims & Ref Data   │
                    │ (train, test, products) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Preprocessing Pipeline  │
                    │  - De-duplication       │
                    │  - Reference Join       │
                    │  - Leakage Elimination  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model Training Engine   │
                    │ (Balanced Random Forest)│
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  FastAPI Backend Service     │              │  Streamlit Operations Portal │
│  - POST /predict             │◄─────────────┤  - Real-time Claim Scoring   │
│  - GET /model-info           │              │  - Batch Predictions Filter  │
│  - GET /health               │              │  - Risk Factor Explanations  │
└──────────────────────────────┘              └──────────────────────────────┘
```

---

## 3. Repository Structure

```
Kestrel-Warranty-Fraud-Detection/
├── backend/
│   └── app.py                      # FastAPI application with Pydantic validation
├── frontend/
│   └── app.py                      # Streamlit interactive dashboard
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Reproducible data cleaning & feature pipeline
│   ├── train_model.py              # Model benchmarking & validation suite
│   ├── retrain.py                  # Calibrated temporal training & artifact exporter
│   ├── eda.py                      # Exploratory data analysis script
│   └── explainability.py           # Natural-language reason generator
├── data/
│   ├── train.csv                   # Historical warranty claims
│   ├── test_unlabelled.csv         # Recent unlabelled claims
│   ├── products.csv                # SKU catalog & warranty months
│   └── partners.csv                # Service partner metadata
├── models/
│   ├── best_model.joblib           # Trained Random Forest artifact
│   ├── preprocessor.joblib         # Fitted Scikit-Learn transformer
│   └── model_metadata.json         # Evaluation metrics & metadata
├── evidence/
│   ├── eda_report.md               # Complete EDA statistical breakdown
│   ├── cost_and_threshold_analysis.md # Detailed financial formulas & tables
│   ├── model_comparison.json       # Candidate model benchmarks
│   ├── feature_importance.csv      # Ranked feature importances
│   ├── threshold_analysis.csv      # Multi-threshold metrics
│   ├── confusion_matrix.png        # Test set confusion matrix
│   ├── roc_curve.png               # ROC curve
│   ├── precision_recall_curve.png  # PR curve
│   ├── feature_importance.png      # Feature importance chart
│   ├── score_distribution.png      # Score histograms
│   └── calibration_plot.png        # Probability calibration curve
├── tests/
│   └── test_all.py                 # Comprehensive unit & API integration test suite
├── predictions.csv                 # Generated predictions for all 2,252 test claims
├── memo_to_ritu.md                 # 1-page executive memorandum to Ritu Deshpande
├── submission_answers.md           # Answers to all 13 submission questions
├── assumptions_and_decisions.md    # Methodological choices & policy rules
├── data_dictionary.md              # Field-by-field definitions & availability
├── model_expectations.md           # Anticipated performance, errors & uncertainties
├── recording_script.md             # 3-minute video presentation script
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Git configuration
└── README.md                       # This document
```

---

## 4. Setup & Installation

### Requirements
- Python 3.10, 3.11, or 3.12
- **Paid API Keys:** **None required** (₹0 external API cost; runs entirely locally).

### Windows Setup
```powershell
# Clone repository
git clone https://github.com/KrushnaIngale/Kestrel-warranty-fraud-detection.git
cd Kestrel-warranty-fraud-detection

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### macOS / Linux Setup
```bash
# Clone repository
git clone https://github.com/KrushnaIngale/Kestrel-warranty-fraud-detection.git
cd Kestrel-warranty-fraud-detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Execution Commands

### 1. Run Automated Test Suite
```bash
pytest tests/test_all.py -v
```
*(All 30 tests covering ingestion, de-duplication, leakage prevention, serialization, edge cases, and API endpoints).*

### 2. Run Exploratory Data Analysis
```bash
python src/eda.py
```

### 3. Train Model & Generate Predictions
```bash
python src/retrain.py
```
*(Trains candidate models, performs temporal evaluation on untouched holdout, saves model artifacts, and writes `predictions.csv`).*

### 4. Launch FastAPI Backend
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 5. Launch Operations Dashboard (Frontend)
In a separate terminal window:
```bash
streamlit run frontend/app.py --server.port 8501
```
- Dashboard URL: [http://localhost:8501](http://localhost:8501)

---

## 6. API Usage Example

### Endpoint: `POST /predict`
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "claim_id": "WC711348",
       "submitted_at": "2026-07-01 10:30",
       "partner_id": "SP3032",
       "sku": "KH-RV-03",
       "product_serial": "KH982341209",
       "days_since_purchase": 340,
       "claim_amount_inr": 12500.0,
       "photo_attached": "N",
       "partner_inspected": "Y",
       "claim_description": "motor burnt out",
       "customer_prior_claims": 2,
       "source": "crm"
     }'
```

### Response:
```json
{
  "claim_id": "WC711348",
  "fraud_risk_score": 0.4821,
  "risk_band": "Medium",
  "reasons": [
    "Claim amount (₹12,500) is 42% of the product's list price (₹29,698)",
    "Claim amount of ₹12,500 is significantly above the median claim (₹1,437)",
    "Claim was submitted in the final quarter of the warranty period",
    "Customer has 2 prior warranty claims",
    "No photo evidence was attached with the claim"
  ],
  "warnings": [],
  "model_version": "1.0.0"
}
```

---

## 7. Model Evaluation & Performance Summary

| Metric | Validation Set (Temporal) | Test Set (Untouched Temporal Holdout) |
|---|:---:|:---:|
| **Sample Size** | 1,338 claims | 2,230 claims (47 fraud) |
| **ROC-AUC** | 0.8860 | 0.4709 |
| **PR-AUC** | 0.5346 | 0.0675 |
| **Accuracy (at $t=0.25$)** | 98.50% | **96.91%** |
| **Precision (at $t=0.25$)** | 36.36% | **15.62%** |
| **Recall (at $t=0.25$)** | 42.11% | **10.64%** |
| **F1-Score (at $t=0.25$)** | 0.3902 | **0.1266** |

### Why 97% Accuracy is Misleading
In a dataset where fraud accounts for only ~1.2% of claims, a trivial model predicting "not fraud" on every record scores **98.8% accuracy** while catching zero fraud. Operating at threshold $t=0.25$ provides an honest 96.91% accuracy while staying strictly within the 40-claims/month review bandwidth and generating positive net savings.

---

## 8. Financial Summary (at 750 claims/month)

- **Operating Threshold:** $t = 0.25$
- **Monthly Reviews:** 10.8 claims/month (Capacity: 40/month)
- **Gross Fraud Stopped:** **₹7,604 / month**
- **Goodwill Penalty Incurred:** **₹3,451 / month** (9.1 false positives $\times$ ₹380)
- **Net Realized Savings:** **₹4,153 / month** (~₹50,000 annually)

---

## 9. Key Limitations & Assumptions

1. **Target Leakage Exclusion:** `inspector_note` was strictly excluded as it contains post-investigation evidence.
2. **Undecided Outcomes:** 215 unresolved claims in train data were excluded from training labels.
3. **May 2026 Policy Shock:** The policy shift to auto-approve claims under ₹2,000 altered incoming claim distribution; ongoing quarterly retraining is recommended.
4. **Partner Neutrality:** Raw partner IDs are excluded as direct features to protect service partners in tier-2/3 cities from discriminatory bias.
