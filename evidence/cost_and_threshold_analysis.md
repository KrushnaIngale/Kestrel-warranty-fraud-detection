# Cost and Threshold Analysis — Kestrel Warranty Fraud Detection

## 1. Business Parameters and Operational Constraints

From Kestrel Operations Policy v4.1 and executive communications:
- **Investigation Desk Capacity**: Exactly **40 claims per month** maximum (Farhan Sheikh & Policy §5).
- **Goodwill Penalty for Delaying Legitimate Claims (False Positives)**: **₹380** on average per claim held for investigation (Policy §4).
- **Prevented Fraud Value (True Positives)**: Full claim payout prevented. Mean confirmed fraud claim value in historical data is **₹4,522** (median ₹2,265).
- **Blended Service Contact Cost**: **₹260** per contact (Policy §4).
- **Monthly Claim Volume**: Expected baseline of **750 warranty claims per month** for Kestrel Home Appliances.
- **Weekly Volume Benchmark (Vireo)**: **650 tickets per week** (~2,817 tickets per month).

---

## 2. Threshold Performance & Cost Analysis (at 750 claims/month)

Evaluation on the temporal test set (2,230 claims over ~3 months, equivalent to ~743 claims/month):

| Threshold ($t$) | Monthly Flagged Claims | Monthly True Positives (Fraud Caught) | Monthly False Positives (Goodwill Cost) | Precision | Recall | Gross Prevented Payouts | Goodwill Holding Cost | Net Monthly Savings | Within 40 Review Capacity? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.05** | 91.1 | 2.7 | 88.5 | 3.0% | 17.0% | ₹12,166 | ₹33,612 | -₹21,446 | ❌ No (>40) |
| **0.10** | 33.3 | 2.4 | 30.9 | 7.1% | 14.9% | ₹10,645 | ₹11,758 | -₹1,113 | ✅ Yes |
| **0.15** | 20.9 | 2.0 | 18.8 | 9.7% | 12.8% | ₹9,124 | ₹7,157 | +₹1,968 | ✅ Yes |
| **0.20** | 13.8 | 1.7 | 12.1 | 12.2% | 10.6% | ₹7,604 | ₹4,601 | +₹3,003 | ✅ Yes |
| **0.25 (Recommended)** | **10.8** | **1.7** | **9.1** | **15.6%** | **10.6%** | **₹7,604** | **₹3,451** | **+₹4,153** | **✅ Yes (Optimal Net)** |
| **0.30** | 8.1 | 1.3 | 6.7 | 16.7% | 8.5% | ₹6,083 | ₹2,556 | +₹3,527 | ✅ Yes |
| **0.35** | 7.1 | 1.0 | 6.1 | 14.3% | 6.4% | ₹4,562 | ₹2,300 | +₹2,262 | ✅ Yes |
| **0.40** | 5.0 | 0.7 | 4.4 | 13.3% | 4.3% | ₹3,041 | ₹1,661 | +₹1,380 | ✅ Yes |
| **0.50** | 3.0 | 0.7 | 2.4 | 22.2% | 4.3% | ₹3,041 | ₹895 | +₹2,147 | ✅ Yes |
| **0.60** | 1.7 | 0.7 | 1.0 | 40.0% | 4.3% | ₹3,041 | ₹383 | +₹2,658 | ✅ Yes |

---

## 3. Financial Formulas

1. **Gross Prevented Fraud Payout**:
   $$\text{Gross Prevented} = \text{Monthly True Positives} \times \text{Mean Fraud Amount (₹4,522)}$$

2. **Goodwill Penalty Cost (False Positives)**:
   $$\text{Goodwill Penalty} = \text{Monthly False Positives} \times \text{Goodwill Cost per Held Genuine Claim (₹380)}$$

3. **Net Monthly Savings**:
   $$\text{Net Monthly Savings} = \text{Gross Prevented} - \text{Goodwill Penalty}$$

4. **Realized Savings vs. Theoretical Savings**:
   - Theoretical savings assume 100% of confirmed fraud claims result in complete claim denial with zero litigation or dispute costs.
   - Realized savings must deduct staff review allocation: 10.8 reviews/month takes approx. 5.4 analyst hours (assuming 30 minutes/review). At internal staff cost, this fits completely within existing budgeted capacity.

---

## 4. Analysis at Vireo's Volume (650 tickets/week)

At 650 tickets/week (~2,817 tickets/month), scaling by $3.76\times$:
- **At recommended threshold ($t=0.25$)**:
  - Flagged tickets/month: **40.6 tickets** (matches the 40-ticket investigation capacity perfectly!)
  - Confirmed fraud caught: **6.4 tickets/month**
  - Gross fraud prevented: **₹28,591 / month** (~₹3.43 Lakh / year)
  - Goodwill cost incurred: **₹12,976 / month**
  - **Net monthly savings**: **₹15,615 / month** (~₹1.87 Lakh / year)
