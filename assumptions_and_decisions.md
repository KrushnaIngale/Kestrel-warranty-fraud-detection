# Assumptions and Decisions — Kestrel Warranty Fraud Detection

## Target Label Treatment

| Decision | Rationale |
|----------|-----------|
| **Exclude is_fraud=NaN (215 rows) from training** | These are undecided investigations. Treating them as 0 (legitimate) would contaminate the negative class; treating them as 1 (fraud) is unsupported. Excluding them is the most defensible approach. |
| **Accept legacy_zoho is_fraud=0 at face value** | Tanmay noted Zoho couldn't store blanks, so undecided cases came across as 0. However, we cannot distinguish true-negatives from undecided in legacy data. Contamination risk exists but cannot be quantified. Documented as a known limitation. |
| **Fraud definition: is_fraud=1 means confirmed fraud** | Per ops policy: "Investigation outcomes are recorded as fraud / not fraud; cases still open at export are undecided." |

## Duplicate Claims

| Decision | Rationale |
|----------|-----------|
| **De-duplicate train by keeping only the first occurrence of each claim_id** | Tanmay confirmed partners re-submit bounced claims. These are true duplicates (identical data). Keeping duplicates would inflate training data and bias toward re-submitted claims. |
| **Test data has no duplicate claim_ids** | Verified — all 2,252 test claim_ids are unique. |

## Feature Leakage

| Column | Decision | Rationale |
|--------|----------|-----------|
| `inspector_note` | **Exclude from model features** | Inspector notes are written during/after investigation. Content often reveals the outcome ("approved", "serial verified", "Photos match fault"). Using this would leak target information. |
| `is_fraud` | Target variable — never a feature | — |
| `source` | **Include as feature** | Indicates system of record (CRM vs legacy Zoho), available at claim time. Not a post-decision field. However, may proxy for time period rather than carry fraud signal. |

## Feature Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| **Derive `claim_amount_ratio` = claim_amount / list_price** | Normalises claim amounts relative to product value. A ₹5,000 claim on a ₹6,000 product is different from a ₹5,000 claim on a ₹30,000 product. |
| **Derive `warranty_remaining_pct`** | days_since_purchase relative to warranty_months × 30. Claims near expiry may have different risk profiles. |
| **Derive `partner_tenure_days`** | Days between partner onboarded_date and claim submitted_at. Captures partner experience without blanket assumptions. |
| **Do NOT use partner_id as a direct feature** | With 380 partners, direct encoding risks overfitting and raises fairness concerns. Use partner_type and aggregated features instead. |
| **Text features from claim_description** | Extract description length and word count as simple numeric features. Full NLP is out of scope. |
| **photo_attached and partner_inspected as binary features** | Convert Y/N to 1/0 |

## Policy-Driven Decisions

| Decision | Rationale |
|----------|-----------|
| **Review capacity: 40 claims/month** | Per Farhan Sheikh and ops policy §5. Threshold must respect this operational constraint. |
| **Goodwill cost of holding legitimate claim: ₹380** | Ops policy §4 |
| **Blended cost of service contact: ₹260** | Ops policy §4 |
| **Auto-approve policy change (May 2026)** | From 1 May 2026, claims under ₹2,000 auto-approved without inspection. This changes partner_inspected patterns in recent data. |
| **97% accuracy target is a hypothesis, not a promise** | Ritu's board wants 97%. With ~1.2% fraud prevalence, a DummyClassifier predicting all-legitimate achieves ~98.8% accuracy. Accuracy is misleading for imbalanced problems. We report it but optimise for business-relevant metrics. |

## Validation Strategy

| Decision | Rationale |
|----------|-----------|
| **Temporal split for validation** | Train on older claims, validate on later claims. This simulates real deployment where the model scores future claims. More realistic than random split. |
| **Stratified final test holdout** | Reserve a final test set that is never used for hyperparameter tuning or model selection. |
| **Class imbalance handled via class_weight='balanced'** | SMOTE/oversampling not used because (a) the minority class is very small (145 cases) and (b) resampling can introduce artifacts. class_weight adjusts the loss function without creating synthetic samples. |

## Partner Fairness

| Decision | Rationale |
|----------|-----------|
| **Do not treat partner tenure as evidence of misconduct** | Meenal Joshi explicitly warned: "please don't paint all the new partners with one brush, most of the new ones are fine." Partner tenure is a feature but its contribution must be evaluated empirically and reported transparently. |
| **Report fraud rates with sample sizes** | Small sample sizes can produce misleading rates. All partner-level statistics include sample counts. |

## Scope Limitations

| Item | Status |
|------|--------|
| Kestrel Shield plan analysis | No Shield-specific field in the data. Cannot distinguish Shield vs standard warranty. |
| Customer ID for repeat customer analysis | No explicit customer_id. `customer_prior_claims` provides partial information. |
| Free-text NLP on claim_description | Simple length/word-count features only. Sentiment analysis and topic modeling are out of scope. |
| Serial number validation | Serial formats are inconsistent. Used as-is for description length features. |
