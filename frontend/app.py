"""
Streamlit Dashboard for Kestrel Warranty Fraud Detection.
Professional enterprise interface for D2C Operations team.
"""
import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Kestrel Warranty Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .disclaimer { background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛡️ Kestrel Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview Dashboard",
    "🔍 Score Single Claim",
    "📋 Batch Predictions",
])
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="disclaimer">⚠️ <b>Important:</b> Risk scores are decision-support signals, '
    'not automatic grounds for rejecting claims. All flagged claims require human review.</div>',
    unsafe_allow_html=True
)

# Load predictions
@st.cache_data
def load_predictions():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return None

@st.cache_data
def load_test_data():
    path = DATA_DIR / "test_unlabelled.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_products():
    path = DATA_DIR / "products.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_model_metadata():
    path = MODELS_DIR / "model_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def get_risk_band(score):
    if score >= 0.6:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"

def risk_badge(band):
    colors = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
    text_colors = {"High": "white", "Medium": "black", "Low": "white"}
    return f'<span style="background-color:{colors.get(band,"#6c757d")}; color:{text_colors.get(band,"white")}; padding:4px 12px; border-radius:4px; font-weight:bold;">{band}</span>'


# ============= OVERVIEW DASHBOARD =============
if page == "📊 Overview Dashboard":
    st.title("📊 Warranty Claims Fraud Detection Dashboard")
    st.markdown("**Kestrel Home Appliances** — D2C Operations Decision Support")
    
    st.markdown(
        '<div class="disclaimer">⚠️ Risk scores are decision-support signals for human reviewers. '
        'They do not automatically reject claims.</div>',
        unsafe_allow_html=True
    )
    
    predictions = load_predictions()
    metadata = load_model_metadata()
    
    if predictions is not None:
        predictions['risk_band'] = predictions['score'].apply(get_risk_band)
        
        total = len(predictions)
        high = (predictions['risk_band'] == 'High').sum()
        medium = (predictions['risk_band'] == 'Medium').sum()
        low = (predictions['risk_band'] == 'Low').sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims", f"{total:,}")
        col2.metric("🔴 High Risk", f"{high:,}", f"{high/total*100:.1f}%")
        col3.metric("🟡 Medium Risk", f"{medium:,}", f"{medium/total*100:.1f}%")
        col4.metric("🟢 Low Risk", f"{low:,}", f"{low/total*100:.1f}%")
        
        st.markdown("---")
        
        # Score distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Score Distribution")
            bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 1.0]
            labels = ['0.00-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '0.30-0.40', '0.40-0.50', '0.50-0.60', '0.60-1.00']
            score_binned = pd.cut(predictions['score'], bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()
            score_binned.index = score_binned.index.astype(str)
            st.bar_chart(score_binned, use_container_width=True)
        
        with col_b:
            st.subheader("Risk Band Breakdown")
            band_order = ['Low', 'Medium', 'High']
            band_counts = predictions['risk_band'].value_counts().reindex(band_order, fill_value=0)
            band_counts.index = band_counts.index.astype(str)
            st.bar_chart(band_counts, use_container_width=True)
        
        # Model info
        if metadata:
            st.markdown("---")
            st.subheader("Model Performance Summary")
            test_metrics = metadata.get("test_metrics", {})
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.4f}")
            mc2.metric("PR-AUC", f"{test_metrics.get('pr_auc', 0):.4f}")
            mc3.metric("Precision", f"{test_metrics.get('precision', 0):.4f}")
            mc4.metric("Recall", f"{test_metrics.get('recall', 0):.4f}")
            
            st.caption(f"Model: {metadata.get('model_name', 'unknown')} | "
                       f"Test samples: {metadata.get('test_samples', 'N/A')} | "
                       f"Version: {metadata.get('model_version', 'N/A')}")
    else:
        st.warning("predictions.csv not found. Run the training pipeline first.")

# ============= SCORE SINGLE CLAIM =============
elif page == "🔍 Score Single Claim":
    st.title("🔍 Score a Single Warranty Claim")
    st.markdown(
        '<div class="disclaimer">⚠️ Risk scores are decision-support signals for human reviewers. '
        'They do not automatically reject claims.</div>',
        unsafe_allow_html=True
    )
    
    products = load_products()
    sku_options = products['sku'].tolist() if products is not None else []
    
    with st.form("claim_form"):
        st.subheader("Claim Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            claim_id = st.text_input("Claim ID", value="WC999999")
            submitted_at = st.text_input("Submitted At", value="2026-10-01 12:00")
            partner_id = st.text_input("Partner ID", value="SP3001")
            source = st.selectbox("Source", ["crm", "legacy_zoho"])
        
        with col2:
            sku = st.selectbox("Product SKU", sku_options if sku_options else ["KH-AF-01"])
            product_serial = st.text_input("Product Serial", value="KH123456789")
            days_since_purchase = st.number_input("Days Since Purchase", min_value=0, value=180)
            claim_amount = st.number_input("Claim Amount (INR)", min_value=1.0, value=2000.0)
        
        with col3:
            photo_attached = st.selectbox("Photo Attached", ["Y", "N"])
            partner_inspected = st.selectbox("Partner Inspected", ["Y", "N"])
            customer_prior_claims = st.number_input("Customer Prior Claims", min_value=0, value=0)
        
        claim_description = st.text_area("Claim Description", value="")
        
        submitted = st.form_submit_button("🔍 Score Claim", use_container_width=True)
    
    if submitted:
        payload = {
            "claim_id": claim_id,
            "submitted_at": submitted_at,
            "partner_id": partner_id,
            "sku": sku,
            "product_serial": product_serial,
            "days_since_purchase": days_since_purchase,
            "claim_amount_inr": claim_amount,
            "photo_attached": photo_attached,
            "partner_inspected": partner_inspected,
            "claim_description": claim_description,
            "customer_prior_claims": customer_prior_claims,
            "source": source
        }
        
        with st.spinner("Scoring claim..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader("Prediction Result")
                    
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.metric("Fraud Risk Score", f"{result['fraud_risk_score']:.4f}")
                    with rc2:
                        st.markdown(f"**Risk Band**: {risk_badge(result['risk_band'])}", unsafe_allow_html=True)
                    
                    st.markdown("#### Contributing Factors")
                    for reason in result.get("reasons", []):
                        st.markdown(f"- {reason}")
                    
                    if result.get("warnings"):
                        st.markdown("#### ⚠️ Warnings")
                        for warning in result["warnings"]:
                            st.warning(warning)
                    
                    st.caption(f"Model Version: {result.get('model_version', 'N/A')}")
                    
                elif response.status_code == 503:
                    st.error("Model not loaded. Start the backend with: `python backend/app.py`")
                else:
                    st.error(f"API error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Start the backend with: `cd backend && python app.py`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============= BATCH PREDICTIONS =============
elif page == "📋 Batch Predictions":
    st.title("📋 Batch Predictions Viewer")
    st.markdown(
        '<div class="disclaimer">⚠️ Risk scores are decision-support signals for human reviewers. '
        'They do not automatically reject claims.</div>',
        unsafe_allow_html=True
    )
    
    predictions = load_predictions()
    test_data = load_test_data()
    products = load_products()
    
    if predictions is not None:
        predictions['risk_band'] = predictions['score'].apply(get_risk_band)
        
        # Merge with test data for context
        if test_data is not None:
            display_df = predictions.merge(test_data, on='claim_id', how='left')
            if products is not None:
                display_df = display_df.merge(products[['sku', 'family']], on='sku', how='left')
        else:
            display_df = predictions.copy()
        
        # Filters
        st.subheader("Filters")
        fc1, fc2, fc3 = st.columns(3)
        
        with fc1:
            search_id = st.text_input("Search by Claim ID", "")
        with fc2:
            risk_filter = st.multiselect("Filter by Risk Band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        with fc3:
            if 'family' in display_df.columns:
                families = display_df['family'].dropna().unique().tolist()
                family_filter = st.multiselect("Filter by Product Category", families, default=families)
            else:
                family_filter = None
        
        # Apply filters
        filtered = display_df[display_df['risk_band'].isin(risk_filter)]
        if search_id:
            filtered = filtered[filtered['claim_id'].str.contains(search_id, case=False, na=False)]
        if family_filter and 'family' in filtered.columns:
            filtered = filtered[filtered['family'].isin(family_filter)]
        
        # Sort
        filtered = filtered.sort_values('score', ascending=False)
        
        st.markdown(f"**Showing {len(filtered):,} of {len(predictions):,} claims**")
        
        # Display columns
        display_cols = ['claim_id', 'score', 'risk_band']
        if 'claim_amount_inr' in filtered.columns:
            display_cols.append('claim_amount_inr')
        if 'family' in filtered.columns:
            display_cols.append('family')
        if 'partner_id' in filtered.columns:
            display_cols.append('partner_id')
        if 'days_since_purchase' in filtered.columns:
            display_cols.append('days_since_purchase')
        
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[available_cols].head(500),
            use_container_width=True,
            hide_index=True
        )
        
        if len(filtered) > 500:
            st.info(f"Showing first 500 of {len(filtered):,} results. Use filters to narrow.")
    else:
        st.warning("predictions.csv not found. Run the training pipeline first.")
