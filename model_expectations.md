# Model Expectations — Kestrel Warranty Fraud Detection

## Expected Performance Characteristics

### Class Imbalance Context
- **Fraud prevalence in training data**: ~1.2% (145 fraud out of ~11,800 labeled claims)
- **Class imbalance ratio**: approximately 1:80 (fraud:legitimate)
- This extreme imbalance means accuracy alone is misleading — a model predicting everything as legitimate achieves ~98.8% accuracy

### Expected Evaluation Performance (Estimates)

| Metric | Expected Range | Rationale |
|--------|---------------|-----------|
| **Accuracy** | 96–99% | Dominated by majority class. Even poor models score high. |
| **ROC-AUC** | 0.70–0.90 | With few fraud cases (~20–25 in test), moderate discrimination expected. |
| **PR-AUC** | 0.10–0.40 | Very sensitive to class imbalance. Even 0.15 is meaningful given 1.2% prevalence. |
| **Precision** (at t=0.3) | 0.05–0.20 | Most flagged claims will be legitimate due to low prevalence. |
| **Recall** (at t=0.3) | 0.40–0.80 | Moderate — cannot guarantee catching all fraud without overwhelming false positives. |

### Why 97% Accuracy is Misleading

Ritu's board asked for 97% accuracy. This target is **technically easy to meet** but **operationally meaningless**:

1. A model that predicts every claim as legitimate achieves ~98.8% accuracy
2. Such a model catches **zero fraud** — it's useless for the business purpose
3. We will report accuracy, but optimise for **PR-AUC and business value** (how much fraud is stopped per claim reviewed)

We recommend evaluating by:
- **How many fraudulent claims does the model flag** (recall)
- **What fraction of flagged claims are actually fraud** (precision)
- **Net financial value** of fraud prevented minus false-positive review costs

### Uncertainty and Error Cases

**High uncertainty areas:**
- Claims from product categories with very few historical fraud cases
- Claims from new partners not seen in training data
- Claims near the fraud/legitimate decision boundary (scores 0.3–0.6)
- The 215 undecided investigations excluded from training could contain undetected patterns

**Likely false positives (legitimate claims flagged as suspicious):**
- High-value legitimate claims on expensive products (Robot Vacuum, Water Purifier)
- Legitimate claims filed near warranty expiry (genuine product failures)
- Claims from customers who happen to have multiple legitimate warranty issues

**Likely false negatives (fraud missed):**
- Low-value fraudulent claims that blend in with typical claim patterns
- Sophisticated fraud that mimics legitimate claim characteristics
- Fraud patterns not represented in historical data

### Temporal Validation Concern
- The model is trained on older claims and validated on later claims (temporal split)
- Fraud patterns may shift over time (e.g., after the May 2026 auto-approve policy change)
- Model performance may degrade if fraud tactics evolve
- Regular retraining is recommended (quarterly at minimum)

### Sample Size Limitations
- With only ~145 fraud cases in training, the model has limited examples to learn from
- The test set contains approximately 15–25 fraud cases — small changes in predictions can swing metrics significantly
- Confidence intervals around all metrics are wide
- These are estimates; actual measured results will come from the training run
