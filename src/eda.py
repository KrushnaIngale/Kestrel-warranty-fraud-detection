"""
Exploratory Data Analysis for Kestrel Warranty Fraud Detection.
Generates comprehensive EDA report and plots.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

def run_eda():
    """Generate full EDA report."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    partners = pd.read_csv(DATA_DIR / "partners.csv")
    
    # De-duplicate for analysis
    train_dedup = train.drop_duplicates(subset="claim_id", keep="first").copy()
    
    report = []
    report.append("# EDA Report — Kestrel Warranty Fraud Detection\n")
    
    # 1. Dataset overview
    report.append("## 1. Dataset Overview\n")
    report.append(f"- **Train (raw)**: {len(train)} rows, {len(train.columns)} columns")
    report.append(f"- **Train (deduplicated)**: {len(train_dedup)} rows")
    report.append(f"- **Duplicate claim_ids**: {len(train) - len(train_dedup)} duplicates ({(len(train) - len(train_dedup)) / 2:.0f} pairs)")
    report.append(f"- **Test (unlabeled)**: {len(test)} rows")
    report.append(f"- **Products**: {len(products)} SKUs across {products['family'].nunique()} families")
    report.append(f"- **Partners**: {len(partners)} partners across {partners['city'].nunique()} cities")
    report.append(f"- **Train date range**: {train_dedup['submitted_at'].min()} to {train_dedup['submitted_at'].max()}")
    report.append(f"- **Test date range**: {test['submitted_at'].min()} to {test['submitted_at'].max()}\n")
    
    # 2. Investigation outcome distribution
    report.append("## 2. Investigation Outcome Distribution\n")
    is_fraud_dist = train_dedup['is_fraud'].value_counts(dropna=False).to_dict()
    report.append(f"| Outcome | Count | Percentage |")
    report.append(f"|---------|-------|------------|")
    total = len(train_dedup)
    for k, v in sorted(is_fraud_dist.items(), key=lambda x: str(x[0])):
        label = {0.0: "Not Fraud (0)", 1.0: "Confirmed Fraud (1)"}.get(k, "Undecided (NaN)")
        report.append(f"| {label} | {v} | {v/total*100:.2f}% |")
    report.append("")
    
    # Labeled only analysis
    labeled = train_dedup.dropna(subset=['is_fraud']).copy()
    labeled['is_fraud'] = labeled['is_fraud'].astype(int)
    fraud_count = labeled['is_fraud'].sum()
    legit_count = len(labeled) - fraud_count
    report.append(f"**Labeled claims**: {len(labeled)} (excluding {len(train_dedup) - len(labeled)} undecided)")
    report.append(f"**Fraud prevalence**: {fraud_count}/{len(labeled)} = **{fraud_count/len(labeled)*100:.2f}%**")
    report.append(f"**Class imbalance ratio**: 1:{legit_count//fraud_count} (fraud:legitimate)\n")
    
    # 3. Merge with reference data
    labeled = labeled.merge(products, on='sku', how='left')
    labeled = labeled.merge(partners, on='partner_id', how='left')
    
    # 4. Fraud rate by product family
    report.append("## 3. Fraud Rate by Product Family\n")
    fr_family = labeled.groupby('family')['is_fraud'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
    fr_family.columns = ['Fraud Count', 'Total Claims', 'Fraud Rate']
    report.append(fr_family.to_markdown())
    report.append("")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fr_family_sorted = fr_family.sort_values('Fraud Rate', ascending=True)
    bars = ax.barh(fr_family_sorted.index, fr_family_sorted['Fraud Rate'] * 100, color='steelblue')
    ax.set_xlabel('Fraud Rate (%)')
    ax.set_title('Fraud Rate by Product Family')
    for bar, count in zip(bars, fr_family_sorted['Total Claims']):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'n={int(count)}', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "fraud_rate_by_family.png", dpi=150)
    plt.close(fig)
    
    # 5. Fraud rate by partner type
    report.append("## 4. Fraud Rate by Partner Type\n")
    fr_partner = labeled.groupby('partner_type')['is_fraud'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
    fr_partner.columns = ['Fraud Count', 'Total Claims', 'Fraud Rate']
    report.append(fr_partner.to_markdown())
    report.append("")
    
    # 6. Claim amount distribution
    report.append("## 5. Claim Amount Distribution\n")
    for label, group in labeled.groupby('is_fraud'):
        label_name = "Fraud" if label == 1 else "Legitimate"
        report.append(f"**{label_name}**: mean=₹{group['claim_amount_inr'].mean():.0f}, "
                       f"median=₹{group['claim_amount_inr'].median():.0f}, "
                       f"std=₹{group['claim_amount_inr'].std():.0f}")
    report.append("")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(labeled[labeled['is_fraud']==0]['claim_amount_inr'], bins=50, alpha=0.7, label='Legitimate', color='green', density=True)
    ax.hist(labeled[labeled['is_fraud']==1]['claim_amount_inr'], bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    ax.set_xlabel('Claim Amount (INR)')
    ax.set_ylabel('Density')
    ax.set_title('Claim Amount Distribution by Fraud Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "claim_amount_distribution.png", dpi=150)
    plt.close(fig)
    
    # 7. Days since purchase
    report.append("## 6. Warranty Age and Claim Timing\n")
    for label, group in labeled.groupby('is_fraud'):
        label_name = "Fraud" if label == 1 else "Legitimate"
        report.append(f"**{label_name}**: mean={group['days_since_purchase'].mean():.0f} days, "
                       f"median={group['days_since_purchase'].median():.0f} days")
    report.append("")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(labeled[labeled['is_fraud']==0]['days_since_purchase'], bins=50, alpha=0.7, label='Legitimate', color='green', density=True)
    ax.hist(labeled[labeled['is_fraud']==1]['days_since_purchase'], bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    ax.set_xlabel('Days Since Purchase')
    ax.set_ylabel('Density')
    ax.set_title('Days Since Purchase Distribution by Fraud Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "days_since_purchase_distribution.png", dpi=150)
    plt.close(fig)
    
    # 8. Customer prior claims
    report.append("## 7. Repeat Claims and Customer Patterns\n")
    fr_prior = labeled.groupby('customer_prior_claims')['is_fraud'].agg(['sum', 'count', 'mean'])
    fr_prior.columns = ['Fraud Count', 'Total', 'Fraud Rate']
    report.append(fr_prior.to_markdown())
    report.append("")
    
    # 9. Photo and inspection patterns
    report.append("## 8. Photo and Inspection Patterns\n")
    report.append("**Photo Attached:**")
    fr_photo = labeled.groupby('photo_attached')['is_fraud'].agg(['sum', 'count', 'mean'])
    fr_photo.columns = ['Fraud Count', 'Total', 'Fraud Rate']
    report.append(fr_photo.to_markdown())
    report.append("")
    report.append("**Partner Inspected:**")
    fr_inspect = labeled.groupby('partner_inspected')['is_fraud'].agg(['sum', 'count', 'mean'])
    fr_inspect.columns = ['Fraud Count', 'Total', 'Fraud Rate']
    report.append(fr_inspect.to_markdown())
    report.append("")
    
    # 10. Source
    report.append("## 9. Source System Patterns\n")
    fr_source = labeled.groupby('source')['is_fraud'].agg(['sum', 'count', 'mean'])
    fr_source.columns = ['Fraud Count', 'Total', 'Fraud Rate']
    report.append(fr_source.to_markdown())
    report.append("")
    
    # 11. City distribution
    report.append("## 10. City Distribution\n")
    fr_city = labeled.groupby('city')['is_fraud'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
    fr_city.columns = ['Fraud Count', 'Total', 'Fraud Rate']
    report.append(fr_city.to_markdown())
    report.append("")
    
    # 12. Partner tenure analysis
    report.append("## 11. Partner Tenure Analysis\n")
    report.append("**Hypothesis test**: Are newer partners more likely to submit fraudulent claims?\n")
    labeled['onboarded_date'] = pd.to_datetime(labeled['onboarded_date'])
    labeled['submitted_at'] = pd.to_datetime(labeled['submitted_at'])
    labeled['partner_tenure_days'] = (labeled['submitted_at'] - labeled['onboarded_date']).dt.days
    
    for label, group in labeled.groupby('is_fraud'):
        label_name = "Fraud" if label == 1 else "Legitimate"
        report.append(f"**{label_name}**: mean tenure={group['partner_tenure_days'].mean():.0f} days, "
                       f"median={group['partner_tenure_days'].median():.0f} days")
    
    # Tenure bins
    labeled['tenure_bin'] = pd.cut(labeled['partner_tenure_days'], 
                                    bins=[0, 180, 365, 730, 1095, float('inf')],
                                    labels=['0-6mo', '6-12mo', '1-2yr', '2-3yr', '3yr+'])
    fr_tenure = labeled.groupby('tenure_bin', observed=True)['is_fraud'].agg(['sum', 'count', 'mean'])
    fr_tenure.columns = ['Fraud Count', 'Total', 'Fraud Rate']
    report.append("")
    report.append(fr_tenure.to_markdown())
    report.append("")
    report.append("**Note**: Sample sizes vary substantially across tenure bins. Small samples produce unreliable rates. "
                  "Partner tenure alone should not be treated as evidence of misconduct.\n")
    
    # 13. Missing data
    report.append("## 12. Missing Data Summary\n")
    report.append("### Train:")
    for col in train_dedup.columns:
        missing = train_dedup[col].isnull().sum()
        if missing > 0:
            report.append(f"- `{col}`: {missing} ({missing/len(train_dedup)*100:.1f}%)")
    report.append("")
    report.append("### Test:")
    for col in test.columns:
        missing = test[col].isnull().sum()
        if missing > 0:
            report.append(f"- `{col}`: {missing} ({missing/len(test)*100:.1f}%)")
    report.append("")
    
    # 14. Monthly claim volume
    report.append("## 13. Seasonality and Monthly Volume\n")
    labeled['submit_month'] = pd.to_datetime(labeled['submitted_at']).dt.to_period('M')
    monthly = labeled.groupby('submit_month').agg(
        claims=('is_fraud', 'count'),
        fraud=('is_fraud', 'sum')
    )
    monthly['fraud_rate'] = monthly['fraud'] / monthly['claims']
    report.append(monthly.to_markdown())
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open(EVIDENCE_DIR / "eda_report.md", "w") as f:
        f.write(report_text)
    
    print("EDA report saved to evidence/eda_report.md")
    print(f"Generated {len(list(EVIDENCE_DIR.glob('*.png')))} plots")
    
    return report_text

if __name__ == "__main__":
    run_eda()
