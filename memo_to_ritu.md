# MEMORANDUM

**TO:** Ritu Deshpande, Head of D2C Operations  
**FROM:** Senior ML & Fraud Operations Engineering Team  
**DATE:** 24 November 2026  
**SUBJECT:** Delivery of Warranty Fraud Risk Scoring System & Deployment Recommendation  

---

### 1. Executive Summary & Recommended Deployment Stage
We have built and delivered an end-to-end, explainable warranty fraud-risk scoring system for Kestrel Home Appliances. Rather than an unvalidated auto-rejection mechanism that risks customer alienation, we recommend immediate deployment to **Stage 1: Supervised Human-in-the-Loop Shadow Pilot**. Under this setup, the service desk reviews high-risk claims flagged by the model without delaying standard claim turnaround times.

### 2. What Was Built
- **Risk Scoring Engine:** Calibrated Machine Learning model scoring every incoming claim from $0.0$ to $1.0$.
- **FastAPI Decision Service:** High-performance REST API (`/predict`, `/health`, `/model-info`) delivering scores, risk tiers (Low, Medium, High), plain-language contributing factors, and policy warnings in under 30 milliseconds.
- **D2C Operations Dashboard:** Dedicated web portal allowing service managers to inspect claims, evaluate real-time single-claim scores, and batch-filter unlabelled claims.
- **Unlabelled Claims Scored:** All 2,252 recent unlabelled claims scored in `predictions.csv`.

### 3. Actual Measured Performance & The 97% Accuracy Target
The board requested "accuracy above 97%." Because true fraud accounts for only ~1.2% of claims, a naive baseline predicting every claim as legitimate automatically achieves **98.8% accuracy while catching zero fraud**. 

We evaluated candidate models using realistic **temporal validation** (training on 7,578 historical claims and evaluating on an untouched temporal test holdout of **2,230 subsequent claims** with 47 confirmed fraud events):
- **Selected Model:** Balanced Random Forest Classifier.
- **Measured Accuracy:** **96.9%** at our recommended operating threshold ($t = 0.25$).
- **Test Precision & Recall:** Precision of **15.6%** (1 in every 6.4 flagged claims is confirmed fraud) and Recall of **10.6%**.
- **Discrimination:** ROC-AUC on historical validation reached **0.886** and PR-AUC reached **0.535**.

### 4. Financial Impact at 750 Claims/Month
Kestrel processes approximately 750 claims per month. Farhan's investigation team can review at most **40 claims per month**, and holding a legitimate customer's claim incurs an average goodwill cost of **₹380**.
- **Operating Threshold:** We recommend **$t = 0.25$**.
- **Monthly Workload:** Generates **~11 flagged claims per month** for manual audit (well within Farhan's 40-claim ceiling).
- **Fraud Prevented:** Catches ~1.7 confirmed fraudulent claims per month, preventing **~₹7,600 in gross fraudulent payouts**.
- **Goodwill Deduction:** Incurs ~₹3,450 in goodwill delay costs across 9.1 false positives.
- **Net Net Monthly Savings:** **₹4,153 net savings per month** (~₹50,000 annually), generating positive ROI without overloading the desk.

### 5. Partner Fairness Safeguards
Meenal correctly noted that new partners in tier-2/3 cities must not be unfairly penalized. 
- **Safeguard 1:** Partner tenure is never treated as standalone proof of fraud.
- **Safeguard 2:** Raw partner IDs are excluded from direct decision weights to prevent bias against low-volume technicians.
- **Safeguard 3:** Claims from all partners require identical evidentiary standards (invoice, photo inspection).

### 6. Action Items for Next Week
1. **Pilot Briefing:** Walk Farhan and Meenal through the Streamlit review dashboard.
2. **Review Triage:** Route only claims with score $\ge 0.25$ (~3 claims/week) to the investigation desk.
3. **Audit May Policy Impact:** Monitor the 1 May 2026 rule (auto-approving claims < ₹2,000); recent data shows small fraud surged following this change.

### 7. What is Not Yet Proven
- **Fraud Shift:** We cannot prove how quickly fraudsters will adapt to thresholds once known.
- **Kestrel Shield Differentiation:** The export lacked a dedicated Shield indicator, so extended warranty nuances remain unmeasured.
- **Undecided Outcomes:** 215 past claims marked undecided were excluded from training; their eventual resolution may uncover newer patterns.
