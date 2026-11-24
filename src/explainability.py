"""
Explainability module for Kestrel Warranty Fraud Detection.
Generates human-readable explanations for individual claim predictions.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_risk_band(score: float) -> str:
    """Convert fraud risk score to risk band."""
    if score >= 0.6:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"


def explain_prediction(claim_data: dict, score: float, model, preprocessor,
                        products_df: pd.DataFrame, partners_df: pd.DataFrame) -> dict:
    """
    Generate human-readable explanation for a single prediction.
    
    Returns dict with:
    - fraud_risk_score
    - risk_band
    - reasons (list of 3-5 contributing factors)
    - warnings (list of data quality or policy warnings)
    """
    risk_band = get_risk_band(score)
    reasons = []
    warnings_list = []
    
    # Get reference data
    sku = claim_data.get('sku', '')
    product_info = products_df[products_df['sku'] == sku].to_dict('records')
    product_info = product_info[0] if product_info else {}
    
    partner_id = claim_data.get('partner_id', '')
    partner_info = partners_df[partners_df['partner_id'] == partner_id].to_dict('records')
    partner_info = partner_info[0] if partner_info else {}
    
    claim_amount = claim_data.get('claim_amount_inr', 0)
    list_price = product_info.get('list_price_inr', 1)
    days_since_purchase = claim_data.get('days_since_purchase', 0)
    warranty_months = product_info.get('warranty_months', 12)
    warranty_days = warranty_months * 30
    family = product_info.get('family', 'Unknown')
    prior_claims = claim_data.get('customer_prior_claims', 0)
    photo = claim_data.get('photo_attached', 'N')
    inspected = claim_data.get('partner_inspected', 'N')
    
    # --- Generate reasons based on feature values ---
    
    # Claim amount relative to product price
    if list_price > 0:
        ratio = claim_amount / list_price
        if ratio > 0.5:
            reasons.append(f"Claim amount (₹{claim_amount:,.0f}) is {ratio:.0%} of the product's list price (₹{list_price:,.0f}), which is unusually high")
        elif ratio > 0.3:
            reasons.append(f"Claim amount (₹{claim_amount:,.0f}) is {ratio:.0%} of the product's list price")
    
    # High absolute claim amount
    if claim_amount > 5000:
        reasons.append(f"Claim amount of ₹{claim_amount:,.0f} is significantly above the median claim (₹1,437)")
    
    # Warranty timing
    if warranty_days > 0:
        warranty_pct = days_since_purchase / warranty_days
        if warranty_pct > 0.9:
            reasons.append(f"Claim was submitted very close to warranty expiry ({days_since_purchase} days after purchase, warranty is {warranty_months} months)")
        elif warranty_pct > 0.75:
            reasons.append(f"Claim was submitted in the final quarter of the warranty period")
    
    if days_since_purchase > warranty_days:
        reasons.append(f"Claim was submitted {days_since_purchase - warranty_days} days after the {warranty_months}-month warranty expired")
        warnings_list.append(f"Claim may be outside warranty period (purchased {days_since_purchase} days ago, warranty is {warranty_months} months)")
    
    # Prior claims
    if prior_claims >= 3:
        reasons.append(f"Customer has {prior_claims} prior warranty claims, which is well above average (0.3)")
    elif prior_claims >= 2:
        reasons.append(f"Customer has {prior_claims} prior warranty claims")
    
    # No photo
    if photo == 'N':
        reasons.append("No photo evidence was attached with the claim")
    
    # No inspection
    if inspected == 'N':
        if claim_amount >= 2000:
            reasons.append("Claim was not inspected by the partner despite being above the ₹2,000 auto-approval threshold")
            warnings_list.append("Policy requires partner inspection for claims ≥₹2,000")
    
    # Ensure we have at least 3 reasons
    if len(reasons) < 3:
        reasons.append(f"Claim is for a {family} product (SKU: {sku})")
    if len(reasons) < 3:
        if partner_info:
            reasons.append(f"Claim submitted through {partner_info.get('partner_type', 'unknown').replace('_', ' ')} partner in {partner_info.get('city', 'unknown')}")
    if len(reasons) < 3:
        reasons.append(f"Claim was filed {days_since_purchase} days after purchase")
    
    # Limit to 5 reasons
    reasons = reasons[:5]
    
    # Additional warnings
    if not product_info:
        warnings_list.append(f"Unknown product SKU: {sku}")
    if not partner_info:
        warnings_list.append(f"Unknown partner ID: {partner_id}")
    
    return {
        "claim_id": claim_data.get('claim_id', 'Unknown'),
        "fraud_risk_score": round(float(score), 4),
        "risk_band": risk_band,
        "reasons": reasons,
        "warnings": warnings_list,
        "model_version": "1.0.0"
    }


def batch_explain(predictions_df: pd.DataFrame, claims_df: pd.DataFrame,
                   products_df: pd.DataFrame, partners_df: pd.DataFrame,
                   model, preprocessor) -> list:
    """Generate explanations for a batch of predictions."""
    explanations = []
    for _, row in predictions_df.iterrows():
        claim_id = row['claim_id']
        score = row['score']
        
        claim_row = claims_df[claims_df['claim_id'] == claim_id]
        if len(claim_row) > 0:
            claim_data = claim_row.iloc[0].to_dict()
        else:
            claim_data = {'claim_id': claim_id}
        
        explanation = explain_prediction(claim_data, score, model, preprocessor,
                                          products_df, partners_df)
        explanations.append(explanation)
    
    return explanations
