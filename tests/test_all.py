"""
Tests for Kestrel Warranty Fraud Detection.
"""
import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "backend"))

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


# ======= DATA TESTS =======

class TestDataIngestion:
    """Test CSV ingestion and data quality."""
    
    def test_train_csv_loads(self):
        df = pd.read_csv(DATA_DIR / "train.csv")
        assert len(df) > 0
        assert "claim_id" in df.columns
        assert "is_fraud" in df.columns
    
    def test_test_csv_loads(self):
        df = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
        assert len(df) > 0
        assert "claim_id" in df.columns
        assert "is_fraud" not in df.columns
    
    def test_products_csv_loads(self):
        df = pd.read_csv(DATA_DIR / "products.csv")
        assert len(df) == 21
        assert set(df.columns) == {"sku", "family", "list_price_inr", "warranty_months"}
    
    def test_partners_csv_loads(self):
        df = pd.read_csv(DATA_DIR / "partners.csv")
        assert len(df) == 380
        assert set(df.columns) == {"partner_id", "city", "onboarded_date", "partner_type"}
    
    def test_train_columns_match_test(self):
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
        expected_extra = {"is_fraud"}
        assert set(train.columns) - set(test.columns) == expected_extra
    
    def test_no_negative_amounts(self):
        train = pd.read_csv(DATA_DIR / "train.csv")
        assert (train["claim_amount_inr"] >= 0).all()
    
    def test_no_negative_days(self):
        train = pd.read_csv(DATA_DIR / "train.csv")
        assert (train["days_since_purchase"] >= 0).all()


# ======= PREPROCESSING TESTS =======

class TestPreprocessing:
    """Test preprocessing pipeline."""
    
    def test_clean_train_removes_duplicates(self):
        from preprocessing import clean_train
        train = pd.read_csv(DATA_DIR / "train.csv")
        cleaned = clean_train(train)
        assert cleaned["claim_id"].is_unique
    
    def test_clean_train_removes_nan_target(self):
        from preprocessing import clean_train
        train = pd.read_csv(DATA_DIR / "train.csv")
        cleaned = clean_train(train)
        assert cleaned["is_fraud"].isna().sum() == 0
    
    def test_enrich_preserves_row_count(self):
        from preprocessing import enrich_with_reference
        products = pd.read_csv(DATA_DIR / "products.csv")
        partners = pd.read_csv(DATA_DIR / "partners.csv")
        test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
        enriched = enrich_with_reference(test, products, partners)
        assert len(enriched) == len(test)
    
    def test_feature_engineering_creates_features(self):
        from preprocessing import engineer_features, enrich_with_reference
        products = pd.read_csv(DATA_DIR / "products.csv")
        partners = pd.read_csv(DATA_DIR / "partners.csv")
        test = pd.read_csv(DATA_DIR / "test_unlabelled.csv").head(10)
        enriched = enrich_with_reference(test, products, partners)
        featured = engineer_features(enriched)
        assert "claim_amount_ratio" in featured.columns
        assert "warranty_used_pct" in featured.columns
        assert "partner_tenure_days" in featured.columns
        assert "photo_attached_bin" in featured.columns


# ======= MODEL TESTS =======

class TestModel:
    """Test model artifacts and predictions."""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model not trained yet. Run train_model.py first.")
    
    def test_model_loads(self):
        import joblib
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        assert hasattr(model, "predict_proba")
    
    def test_preprocessor_loads(self):
        import joblib
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        assert hasattr(preprocessor, "transform")
    
    def test_metadata_valid(self):
        with open(MODELS_DIR / "model_metadata.json") as f:
            metadata = json.load(f)
        assert "model_name" in metadata
        assert "model_version" in metadata
        assert "test_metrics" in metadata
    
    def test_predictions_csv_exists(self):
        pred_path = BASE_DIR / "predictions.csv"
        assert pred_path.exists(), "predictions.csv not found in project root"
    
    def test_predictions_row_count(self):
        test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert len(pred) == len(test), f"Prediction rows ({len(pred)}) != test rows ({len(test)})"
    
    def test_predictions_scores_valid(self):
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert pred["score"].between(0, 1).all(), "Scores out of range [0, 1]"
        assert pred["score"].isna().sum() == 0, "NaN scores found"
    
    def test_predictions_claim_ids_match(self):
        test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert set(pred["claim_id"]) == set(test["claim_id"])
    
    def test_predictions_order_preserved(self):
        test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert list(pred["claim_id"]) == list(test["claim_id"])


# ======= PREDICTION SCHEMA TESTS =======

class TestPredictionSchema:
    """Test prediction output schema."""
    
    @pytest.fixture(autouse=True)
    def check_predictions_exist(self):
        if not (BASE_DIR / "predictions.csv").exists():
            pytest.skip("predictions.csv not found. Run train_model.py first.")
    
    def test_has_required_columns(self):
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert "claim_id" in pred.columns
        assert "score" in pred.columns
    
    def test_no_extra_columns(self):
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert set(pred.columns) == {"claim_id", "score"}
    
    def test_score_dtype(self):
        pred = pd.read_csv(BASE_DIR / "predictions.csv")
        assert pred["score"].dtype in [np.float64, np.float32, float]


# ======= EDGE CASE TESTS =======

class TestEdgeCases:
    """Test handling of edge cases."""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model not trained yet.")
    
    def test_missing_optional_fields(self):
        """Model should handle missing optional fields."""
        import joblib
        from preprocessing import enrich_with_reference, engineer_features, ALL_FEATURES
        
        products = pd.read_csv(DATA_DIR / "products.csv")
        partners = pd.read_csv(DATA_DIR / "partners.csv")
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        
        # Minimal claim
        claim = pd.DataFrame([{
            "claim_id": "TEST001",
            "submitted_at": "2026-10-01 12:00",
            "partner_id": "SP3001",
            "sku": "KH-AF-01",
            "product_serial": "TEST123",
            "days_since_purchase": 100,
            "claim_amount_inr": 1000.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "claim_description": "",
            "inspector_note": np.nan,
            "customer_prior_claims": 0,
            "source": "crm"
        }])
        
        enriched = enrich_with_reference(claim, products, partners)
        featured = engineer_features(enriched)
        X = preprocessor.transform(featured[ALL_FEATURES])
        proba = model.predict_proba(X)[:, 1]
        assert 0 <= proba[0] <= 1
    
    def test_unseen_category(self):
        """Model should handle unseen categories."""
        import joblib
        from preprocessing import enrich_with_reference, engineer_features, ALL_FEATURES
        
        products = pd.read_csv(DATA_DIR / "products.csv")
        partners = pd.read_csv(DATA_DIR / "partners.csv")
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        
        # Claim with unknown partner (falls back to NaN -> imputed)
        claim = pd.DataFrame([{
            "claim_id": "TEST002",
            "submitted_at": "2026-10-01 12:00",
            "partner_id": "SP9999",  # Unknown
            "sku": "KH-AF-01",
            "product_serial": "TEST123",
            "days_since_purchase": 100,
            "claim_amount_inr": 1000.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "claim_description": "device not working",
            "inspector_note": np.nan,
            "customer_prior_claims": 0,
            "source": "crm"
        }])
        
        enriched = enrich_with_reference(claim, products, partners)
        featured = engineer_features(enriched)
        X = preprocessor.transform(featured[ALL_FEATURES])
        proba = model.predict_proba(X)[:, 1]
        assert 0 <= proba[0] <= 1
    
    def test_extreme_values(self):
        """Model should handle extreme values without errors."""
        import joblib
        from preprocessing import enrich_with_reference, engineer_features, ALL_FEATURES
        
        products = pd.read_csv(DATA_DIR / "products.csv")
        partners = pd.read_csv(DATA_DIR / "partners.csv")
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        
        claim = pd.DataFrame([{
            "claim_id": "TEST003",
            "submitted_at": "2026-10-01 12:00",
            "partner_id": "SP3001",
            "sku": "KH-AF-01",
            "product_serial": "TEST123",
            "days_since_purchase": 9999,  # Extreme
            "claim_amount_inr": 100000.0,  # Extreme
            "photo_attached": "N",
            "partner_inspected": "N",
            "claim_description": "everything broken " * 100,  # Very long
            "inspector_note": np.nan,
            "customer_prior_claims": 50,  # Extreme
            "source": "crm"
        }])
        
        enriched = enrich_with_reference(claim, products, partners)
        featured = engineer_features(enriched)
        X = preprocessor.transform(featured[ALL_FEATURES])
        proba = model.predict_proba(X)[:, 1]
        assert 0 <= proba[0] <= 1
        assert np.isfinite(proba[0])


# ======= API TESTS =======

class TestAPI:
    """Test FastAPI endpoints (requires running server)."""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        if not (MODELS_DIR / "best_model.joblib").exists():
            pytest.skip("Model not trained yet.")
    
    def test_health_endpoint(self):
        """Test that health endpoint responds correctly via TestClient."""
        from fastapi.testclient import TestClient
        from app import app
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self):
        from fastapi.testclient import TestClient
        from app import app
        
        with TestClient(app) as client:
            response = client.get("/model-info")
            assert response.status_code == 200
            data = response.json()
            assert "model_name" in data
    
    def test_predict_endpoint(self):
        from fastapi.testclient import TestClient
        from app import app
        
        with TestClient(app) as client:
            payload = {
                "claim_id": "WC999999",
                "submitted_at": "2026-10-01 12:00",
                "partner_id": "SP3001",
                "sku": "KH-AF-01",
                "product_serial": "KH123456789",
                "days_since_purchase": 180,
                "claim_amount_inr": 2000.0,
                "photo_attached": "Y",
                "partner_inspected": "Y",
                "claim_description": "blade jammed",
                "customer_prior_claims": 0,
                "source": "crm"
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "fraud_risk_score" in data
            assert "risk_band" in data
            assert "reasons" in data
            assert 0 <= data["fraud_risk_score"] <= 1
            assert data["risk_band"] in ["High", "Medium", "Low"]
    
    def test_predict_missing_required_field(self):
        from fastapi.testclient import TestClient
        from app import app
        
        with TestClient(app) as client:
            payload = {"claim_id": "WC999999"}  # Missing required fields
            response = client.post("/predict", json=payload)
            assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_amount(self):
        from fastapi.testclient import TestClient
        from app import app
        
        with TestClient(app) as client:
            payload = {
                "claim_id": "WC999999",
                "submitted_at": "2026-10-01 12:00",
                "partner_id": "SP3001",
                "sku": "KH-AF-01",
                "product_serial": "KH123456789",
                "days_since_purchase": 180,
                "claim_amount_inr": -100.0,  # Invalid
                "photo_attached": "Y",
                "partner_inspected": "Y",
                "claim_description": "test",
                "customer_prior_claims": 0,
                "source": "crm"
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 422
