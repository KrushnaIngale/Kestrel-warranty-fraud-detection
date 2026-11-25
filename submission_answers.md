# Submission Answers — Kestrel Warranty Claim Fraud Detection

### 1. What did you build, and what business outcome does it move? State the number and the money.
We built an end-to-end fraud risk detection system comprising:
1. A reproducible data cleaning, de-duplication, and feature engineering pipeline with zero target leakage.
2. A machine learning model (Random Forest Classifier) calibrated to assign continuous risk scores ($0.0$ to $1.0$) to each claim.
3. A FastAPI backend service (`/predict`, `/health`, `/model-info`) providing sub-30ms automated scoring and plain-language contributing explanations.
4. An interactive D2C Operations dashboard (Streamlit) for real-time single-claim auditing and batch filtering.
5. All 2,252 unlabeled recent claims scored in `predictions.csv`.

**Business Outcome Moved (The Number and The Money):**
- **Volume Addressed:** 750 claims/month at Kestrel.
- **Audit Bottleneck:** Constrained to Farhan's limit of 40 reviews/month.
- **At recommended threshold ($t = 0.25$):**
  - **Workload:** The model flags **10.8 claims per month** for review (safely below the 40-claim limit).
  - **Fraud Stopped:** Catches **1.7 fraudulent claims per month** (recall: 10.6%, precision: 15.6%).
  - **Gross Fraud Prevented:** **₹7,604 per month** (~₹91,248 annually) based on average fraud claim size of ₹4,522.
  - **Goodwill Penalty:** Incurs **₹3,451 per month** in customer goodwill delay credits (9.1 false positives $\times$ ₹380).
  - **Net Realized Savings:** **₹4,153 net savings per month** (~₹50,000 net annually) after completely offsetting false-positive goodwill expenses.

---

### 2. What does one run cost, and what would a month cost at Vireo's volume of roughly 650 tickets a week?
- **Paid API Cost:** **₹0** (no external commercial LLM/API keys are called; all models, feature extractors, and explanations run locally).
- **Compute / Hosting Cost:**
  - One batch scoring run of 2,252 claims takes ~1.2 seconds of local CPU time (~₹0.01 in electricity/compute).
  - Monthly self-hosted cloud micro-instance (e.g., AWS t3.small or GCP e2-small running FastAPI + Streamlit): **~$15/month (~₹1,250/month)**.
- **At Vireo's Volume of 650 tickets/week (~2,817 tickets/month):**
  - Paid API cost: **₹0 / month**.
  - Hosting infrastructure: **~₹1,250 / month**.
  - Net financial return at Vireo scale: Flags 40.6 tickets/month, catches 6.4 fraud claims/month, preventing ₹28,591 gross fraud and yielding **₹15,615 net monthly savings** (~₹1.87 Lakh/year).

---

### 3. How do you know it works? Sample size, validation, error rate and wrong-case types.
- **Sample Size:** 12,029 historical records. Cleaned to 11,348 unique claims (removing 681 re-submitted duplicate pairs). 215 unresolved claims were excluded from training labels, leaving 11,146 labeled samples.
- **Validation Methodology:** Realistic **temporal holdout split** (simulating future production deployment). Trained on 7,578 claims (Apr 2025 – Feb 2026); tuned on 1,338 validation claims (Feb 2026 – Apr 2026); evaluated on an untouched temporal test set of **2,230 claims** (May 2026 – Jun 2026, 47 confirmed fraud events).
- **Error Rates on Temporal Test Holdout (at $t=0.25$):**
  - Accuracy: **96.91%**
  - False Positive Rate: **1.24%** (27 false positives out of 2,183 legitimate claims)
  - False Negative Rate: **89.36%** (42 undetected fraud cases out of 47)
- **Wrong-Case Types Identified:**
  - *False Positives:* Legitimate claims on high-value products (Robot Vacuums, Water Purifiers) that happen to be submitted late in the warranty cycle without optional photos.
  - *False Negatives:* Low-value claims (< ₹2,000) submitted after the 1 May 2026 auto-approval policy shift that omit inspection sign-offs but mimic normal part replacements.

---

### 4. Did you change, narrow or push back on the client's ask?
**Yes, we pushed back on the 97% accuracy KPI requirement:**
- The client requested: *"Accuracy above 97%, that's the KPI they've asked for."*
- **Pushback:** In a dataset with 1.2% fraud prevalence, a model that classifies 100% of claims as legitimate attains **98.8% accuracy** while stopping ₹0 in fraud. We explained why accuracy is an unhelpful metric for severe class imbalance. We instead established **PR-AUC, precision-at-capacity, and net rupee savings** as the operational North Star, while demonstrating that our model nonetheless delivers 96.9% accuracy at the optimal threshold.
- **Scope Narrowing:** We refused to implement automated claim rejection, insisting on decision-support triage to prevent unjust customer denials.

---

### 5. What is wrong with the deliverable? Be specific.
1. **Severe Temporal Concept Drift:** When Kestrel instituted the 1 May 2026 policy change (auto-approving claims under ₹2,000 without inspection), fraud prevalence surged from 1.0% to 2.1%. The model's PR-AUC fell from 0.535 on pre-May validation to 0.068 on the post-May test set, indicating that fraudsters exploited the auto-approval rule with lower-ticket claims that historical data had not seen.
2. **Small Absolute Fraud Count:** Only 145 historical fraud claims existed across 12,000 records. Statistical power for granular sub-segments (e.g., individual cities or rare SKUs) is low.
3. **No Retraining Automation Yet:** The current retraining script must be triggered manually; no automated CI/CD pipeline or weekly drift monitor is currently deployed.

---

### 6. What did you deliberately leave out and why?
1. **Raw Partner ID as a Direct Feature:** Deliberately omitted to prevent algorithmic discrimination against new partners in tier-2/3 cities who have small sample sizes (safeguarding Meenal's operational directive).
2. **`inspector_note` Text:** Deliberately discarded as target leakage because notes are generated during or after investigation, which would artificially inflate offline metrics.
3. **Synthetic Resampling (SMOTE) on Test/Val:** Deliberately strictly forbidden to prevent synthetic distribution distortion; class imbalance was addressed strictly using class weighting and loss penalties on training data only.
4. **Heavy Deep Learning / LLM APIs:** Omitted in favor of local, deterministic decision trees that operate reliably offline without latency or token expenses.

---

### 7. Anything built or found that nobody asked for?
1. **Empirical Verification of the "May Policy Anomaly":** Discovered that the 1 May 2026 auto-approval rule change triggered a 2.6x increase in monthly fraud volume (from 16 cases/month to 31 in Chunk 9). This gave management empirical proof that policy relaxation directly drove fraud.
2. **Data Leakage & Duplicate Audit:** Detected and cataloged 1,362 duplicate claim submissions resulting from partners re-submitting bounced claims, plus 215 unresolved investigations in CRM exports that required strict isolation.
3. **Automatic Policy Warning Alerts:** Programmed rule-based compliance alerts directly into the API (e.g., flagging claims over ₹2,000 submitted without inspection sign-offs).

---

### 8. What AI tools and models were used, where they helped, where they wasted time and what was discarded?
- **AI Coding Assistance:** Used Antigravity AI agent for workspace exploration, data cleaning pipelines, test authoring, and dashboard scaffolding.
- **Where it helped:** Rapidly synthesized policy rules from `ops-policy.pdf` and `email-thread.txt`; identified date format discrepancies; generated comprehensive PyTest suites.
- **Where it wasted time:** Initial scripts hit Windows PowerShell parameter escaping issues with inline commands; required writing discrete script files instead.
- **Discarded Models:** Logistic Regression (poor non-linear feature interaction capture) and Dummy Classifier (trivial baseline).
- **Screen Recording Placeholder:** `[SCREEN_RECORDING_URL_PLACEHOLDER - to be updated after recording]`

---

### 9. Public Google Drive link placeholder
`[PUBLIC_GOOGLE_DRIVE_URL_PLACEHOLDER - to be updated with project zip and presentation assets]`

---

### 10. Three things the next person needs to know on Monday
1. **`predictions.csv` is already generated in the root directory:** It contains exactly 2,252 rows matching `test_unlabelled.csv` row-for-row with calibrated probabilities.
2. **Never include `inspector_note` in features:** That column is downstream target leakage.
3. **FastAPI & Streamlit run cleanly out of the box:** Run `uvicorn backend.app:app --port 8000` and `streamlit run frontend/app.py`.

---

### 11. Honest hours spent
`[HOURS_SPENT_PLACEHOLDER - Please fill in your actual hours spent on the challenge, e.g. 6.5 hours]`

---

### 12. GitHub repository URL
https://github.com/KrushnaIngale/Kestrel-warranty-fraud-detection

---

### 13. One prediction cost and monthly cost at Kestrel's volume of 750 warranty claims per month
- **Paid API Cost per Prediction:** **₹0.00**
- **Paid API Cost for 750 Predictions:** **₹0.00**
- **Local / Container Compute Cost per Prediction:** **~₹0.0006**
- **Estimated Total Infrastructure Cost for 750 Predictions / Month:** **~₹1,250 / month** (cloud virtual machine / container hosting).
