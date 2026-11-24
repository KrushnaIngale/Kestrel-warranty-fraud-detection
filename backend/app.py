"""
FastAPI Backend for Kestrel Warranty Fraud Detection.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging - no PII
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("kestrel-fraud-api")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Add src to path for imports
sys.path.insert(0, str(BASE_DIR / "src"))
from explainability import explain_prediction, get_risk_band
from preprocessing import engineer_features, enrich_with_reference, ALL_FEATURES

# Initialize FastAPI
app = FastAPI(
    title="Kestrel Warranty Fraud Detection API",
    description="Fraud risk scoring for warranty claims. Scores are decision-support signals, not automatic rejection grounds.",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
_model = None
_preprocessor = None
_metadata = None
_products = None
_partners = None


def load_model():
    """Load model artifacts. Fails with clear message if missing."""
    global _model, _preprocessor, _metadata, _products, _partners
    
    model_path = MODELS_DIR / "best_model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            f"Run 'python src/train_model.py' first to train the model."
        )
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preprocessor_path}. "
            f"Run 'python src/train_model.py' first."
        )
    
    _model = joblib.load(model_path)
    _preprocessor = joblib.load(preprocessor_path)
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            _metadata = json.load(f)
    else:
        _metadata = {"model_version": "1.0.0", "model_name": "unknown"}
    
    _products = pd.read_csv(DATA_DIR / "products.csv")
    _partners = pd.read_csv(DATA_DIR / "partners.csv")
    
    logger.info(f"Model loaded: {_metadata.get('model_name', 'unknown')}")


@app.on_event("startup")
async def startup():
    try:
        load_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("API starting without model. /predict will return 503.")


# Eagerly load model on import if available
try:
    load_model()
except Exception as _e:
    logger.warning(f"Initial model load deferred: {_e}")


# --- Pydantic Models ---

class ClaimInput(BaseModel):
    claim_id: str = Field(..., description="Claim identifier")
    submitted_at: str = Field(..., description="Submission timestamp (YYYY-MM-DD HH:MM)")
    partner_id: str = Field(..., description="Service partner ID")
    sku: str = Field(..., description="Product SKU")
    product_serial: str = Field("", description="Product serial number")
    days_since_purchase: int = Field(..., ge=0, description="Days since purchase")
    claim_amount_inr: float = Field(..., gt=0, description="Claim amount in INR")
    photo_attached: str = Field("N", description="Y or N")
    partner_inspected: str = Field("N", description="Y or N")
    claim_description: str = Field("", description="Fault description")
    customer_prior_claims: int = Field(0, ge=0, description="Prior warranty claims by customer")
    source: str = Field("crm", description="System source: crm or legacy_zoho")
    
    @field_validator('photo_attached', 'partner_inspected')
    @classmethod
    def validate_yn(cls, v):
        if v.upper() not in ('Y', 'N'):
            raise ValueError("Must be 'Y' or 'N'")
        return v.upper()
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        if v not in ('crm', 'legacy_zoho'):
            return 'crm'  # default for unseen sources
        return v


class PredictionResponse(BaseModel):
    claim_id: str
    fraud_risk_score: float
    risk_band: str
    reasons: list[str]
    warnings: list[str]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    features: list[str]
    training_samples: int
    validation_samples: int
    test_samples: int
    test_metrics: dict


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_name=_metadata.get("model_name") if _metadata else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(claim: ClaimInput):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python src/train_model.py' first to train the model."
        )
    
    try:
        # Convert to DataFrame
        claim_dict = claim.model_dump()
        claim_df = pd.DataFrame([claim_dict])
        
        # Enrich with reference data
        claim_df = enrich_with_reference(claim_df, _products, _partners)
        
        # Engineer features
        claim_df = engineer_features(claim_df)
        
        # Predict
        X = _preprocessor.transform(claim_df[ALL_FEATURES])
        score = float(_model.predict_proba(X)[:, 1][0])
        
        # Explain
        explanation = explain_prediction(
            claim_dict, score, _model, _preprocessor,
            _products, _partners
        )
        
        logger.info(f"Prediction for claim {claim.claim_id}: score={score:.4f}, band={explanation['risk_band']}")
        
        return PredictionResponse(**explanation)
        
    except Exception as e:
        logger.error(f"Prediction error for claim {claim.claim_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    
    return ModelInfoResponse(
        model_name=_metadata.get("model_name", "unknown"),
        model_version=_metadata.get("model_version", "1.0.0"),
        features=_metadata.get("features", []),
        training_samples=_metadata.get("training_samples", 0),
        validation_samples=_metadata.get("validation_samples", 0),
        test_samples=_metadata.get("test_samples", 0),
        test_metrics=_metadata.get("test_metrics", {})
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
