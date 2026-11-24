"""
Preprocessing pipeline for Kestrel Warranty Fraud Detection.
Handles data loading, cleaning, feature engineering, and train/test preparation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Leakage columns - must never be used as features
LEAKAGE_COLS = ["is_fraud", "inspector_note"]

# Columns not useful as features (identifiers, raw text)
ID_COLS = ["claim_id", "product_serial"]

# Target column
TARGET = "is_fraud"


def load_raw_data():
    """Load all raw CSV files."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    partners = pd.read_csv(DATA_DIR / "partners.csv")
    return train, test, products, partners


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    """
    Clean training data:
    1. Remove duplicate claim_ids (keep first occurrence)
    2. Remove rows with NaN target (undecided investigations)
    """
    # De-duplicate: keep first occurrence
    n_before = len(train)
    train = train.drop_duplicates(subset="claim_id", keep="first").copy()
    n_after_dedup = len(train)
    
    # Remove undecided investigations
    train = train.dropna(subset=[TARGET]).copy()
    n_after_target = len(train)
    
    train[TARGET] = train[TARGET].astype(int)
    
    print(f"Train cleaning: {n_before} -> {n_after_dedup} (dedup) -> {n_after_target} (remove NaN target)")
    return train.reset_index(drop=True)


def enrich_with_reference(df: pd.DataFrame, products: pd.DataFrame, 
                           partners: pd.DataFrame) -> pd.DataFrame:
    """
    Join claims with product and partner reference data.
    Validates join keys to prevent row multiplication.
    """
    n_before = len(df)
    
    # Merge products
    df = df.merge(products, on="sku", how="left")
    assert len(df) == n_before, f"Product join changed row count: {n_before} -> {len(df)}"
    
    # Merge partners
    df = df.merge(partners, on="partner_id", how="left")
    assert len(df) == n_before, f"Partner join changed row count: {n_before} -> {len(df)}"
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw data.
    All features use only information available at claim submission time.
    """
    df = df.copy()
    
    # Parse dates
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["onboarded_date"] = pd.to_datetime(df["onboarded_date"])
    
    # --- Claim-level features ---
    # Claim amount relative to product list price
    df["claim_amount_ratio"] = df["claim_amount_inr"] / df["list_price_inr"]
    
    # Warranty period in days
    df["warranty_days"] = df["warranty_months"] * 30
    
    # How much warranty has been used (proportion)
    df["warranty_used_pct"] = df["days_since_purchase"] / df["warranty_days"]
    
    # Whether claim was filed after warranty expiry
    df["past_warranty"] = (df["days_since_purchase"] > df["warranty_days"]).astype(int)
    
    # --- Partner features ---
    # Partner tenure at time of claim (days)
    df["partner_tenure_days"] = (df["submitted_at"] - df["onboarded_date"]).dt.days
    
    # --- Binary conversions ---
    df["photo_attached_bin"] = (df["photo_attached"] == "Y").astype(int)
    df["partner_inspected_bin"] = (df["partner_inspected"] == "Y").astype(int)
    
    # --- Text features (simple, no NLP) ---
    df["desc_length"] = df["claim_description"].fillna("").str.len()
    df["desc_word_count"] = df["claim_description"].fillna("").str.split().str.len().fillna(0).astype(int)
    
    # --- Source encoding ---
    df["source_crm"] = (df["source"] == "crm").astype(int)
    
    # --- Time features ---
    df["submit_month"] = df["submitted_at"].dt.month
    df["submit_dayofweek"] = df["submitted_at"].dt.dayofweek
    df["submit_hour"] = df["submitted_at"].dt.hour
    
    # --- Log-transform of claim amount ---
    df["log_claim_amount"] = np.log1p(df["claim_amount_inr"])
    
    return df


# Features used by the model
NUMERIC_FEATURES = [
    "days_since_purchase",
    "claim_amount_inr",
    "log_claim_amount",
    "customer_prior_claims",
    "claim_amount_ratio",
    "warranty_used_pct",
    "partner_tenure_days",
    "desc_length",
    "desc_word_count",
    "list_price_inr",
    "submit_hour",
]

BINARY_FEATURES = [
    "photo_attached_bin",
    "partner_inspected_bin",
    "source_crm",
    "past_warranty",
]

CATEGORICAL_FEATURES = [
    "family",
    "partner_type",
    "city",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    """
    Build a scikit-learn ColumnTransformer for the feature set.
    Handles numeric scaling, binary pass-through, and categorical encoding.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    
    return preprocessor


def prepare_data(temporal_split: bool = True, test_ratio: float = 0.15):
    """
    Full data preparation pipeline.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, 
        df_test_unlabeled (enriched+engineered), preprocessor
    """
    from sklearn.model_selection import train_test_split
    
    train_raw, test_raw, products, partners = load_raw_data()
    
    # Clean training data
    train_clean = clean_train(train_raw)
    
    # Enrich with reference data
    train_enriched = enrich_with_reference(train_clean, products, partners)
    test_enriched = enrich_with_reference(test_raw, products, partners)
    
    # Engineer features
    train_feat = engineer_features(train_enriched)
    test_feat = engineer_features(test_enriched)
    
    # Extract target
    y_all = train_feat[TARGET].values
    
    if temporal_split:
        # Sort by submission date, train on older, validate on newer
        train_feat = train_feat.sort_values("submitted_at").reset_index(drop=True)
        y_all = train_feat[TARGET].values
        
        n = len(train_feat)
        # Use first 70% for training, next 15% for validation, last 15% for test
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        X_train_df = train_feat.iloc[:train_end]
        X_val_df = train_feat.iloc[train_end:val_end]
        X_test_df = train_feat.iloc[val_end:]
        
        y_train = y_all[:train_end]
        y_val = y_all[train_end:val_end]
        y_test = y_all[val_end:]
    else:
        # Stratified random split
        X_temp, X_test_df, y_temp, y_test = train_test_split(
            train_feat, y_all, test_size=test_ratio, random_state=42, stratify=y_all
        )
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=test_ratio / (1 - test_ratio), 
            random_state=42, stratify=y_temp
        )
    
    # Build preprocessor and fit on training data only
    preprocessor = build_preprocessor()
    
    X_train = preprocessor.fit_transform(X_train_df[ALL_FEATURES])
    X_val = preprocessor.transform(X_val_df[ALL_FEATURES])
    X_test = preprocessor.transform(X_test_df[ALL_FEATURES])
    
    print(f"\nSplit sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Fraud rate: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            test_feat, preprocessor, X_train_df, X_val_df, X_test_df)


def prepare_test_predictions(test_feat: pd.DataFrame, preprocessor) -> np.ndarray:
    """Transform the unlabeled test data for prediction."""
    return preprocessor.transform(test_feat[ALL_FEATURES])
