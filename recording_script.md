# Recording Script — Kestrel Warranty Fraud Detection Demo

**Maximum duration: 3 minutes**

---

## 0:00–0:25 — Business Problem and Data Inspection (25 seconds)

> "Kestrel Home Appliances processes around 750 warranty claims per month across air fryers, mixer-grinders, water purifiers, and four other product lines. About 1.2% of historical claims were confirmed fraud. The business needed a fraud-risk scoring system — not an auto-rejector — to help their investigation desk of 40 reviews per month focus on the right claims."

*Show:* Project directory structure, train.csv row count, is_fraud distribution (11,669 legitimate / 145 fraud / 215 undecided)

---

## 0:25–0:55 — Key Discoveries and Leakage Risks (30 seconds)

> "Three critical findings:
> 1. The `inspector_note` field is a leakage risk — it contains investigation outcomes like 'approved' and 'serial verified'. We excluded it from features.
> 2. 215 claims with undecided investigations were excluded from training — not silently labelled as legitimate.
> 3. Duplicate claim_ids from partner re-submissions were de-duplicated.
> 4. Ritu's 97% accuracy target is technically trivial — a 'predict everything legitimate' model scores 98.8%. We optimised for business value instead."

*Show:* assumptions_and_decisions.md, inspector_note examples, is_fraud NaN rows

---

## 0:55–1:25 — Models Tested and Rejected (30 seconds)

> "We trained four models: DummyClassifier baseline, Logistic Regression, Random Forest, and HistGradientBoosting. All used temporal validation — training on older claims, validating on later ones — to simulate real deployment."

> "The final model is [model_name]. Key test metrics: ROC-AUC [X.XXX], PR-AUC [X.XXX], validated on [N] claims including [N] fraud cases."

*Show:* Model comparison table, final test metrics, evidence plots

---

## 1:25–2:20 — Live Dashboard and Single Claim Prediction (55 seconds)

> "Here's the Streamlit dashboard. The overview shows all 2,252 scored claims broken down by risk band."

*Action:* Navigate to Overview Dashboard, show risk band counts

> "Let me score a single claim. I'll enter a suspicious-looking one — high amount, near warranty expiry, no photo."

*Action:* Go to Score Single Claim, fill in a high-risk example, click Score. Show the result — fraud score, risk band, contributing factors, and warnings.

> "And here's the batch predictions viewer — operations can search and filter by claim ID, risk band, or product category."

*Action:* Navigate to Batch Predictions, filter by High Risk, show sorted results

---

## 2:20–2:45 — Evidence, Error Cases, Financial Impact (25 seconds)

> "Financial impact at the recommended threshold: reviewing [N] claims per month catches approximately ₹[X] in fraudulent payouts, within the 40-review monthly capacity. After accounting for ₹380 goodwill cost per false positive, the estimated net savings are ₹[X] per month."

> "Common error cases: high-value legitimate claims on expensive products get flagged as false positives. Some low-value fraud blends in and gets missed."

*Show:* Threshold analysis table, net savings calculation

---

## 2:45–3:00 — Limitations and Next Steps (15 seconds)

> "Key limitations: only 145 fraud cases in training data, so confidence intervals are wide. No Shield plan field in the data, so we couldn't distinguish extended warranty claims. The model should be retrained quarterly as fraud patterns evolve."

> "Next steps: pilot with Meenal's service desk, monitor false positive rates, expand training data as new investigations complete."

---

**Note**: Replace [X.XXX] and [N] with actual metrics from the training run. This recording has NOT been created yet — the human must record and upload it.
