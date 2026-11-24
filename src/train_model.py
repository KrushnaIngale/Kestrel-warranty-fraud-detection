"""
Model training, evaluation, and selection for Kestrel Warranty Fraud Detection.
"""
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from preprocessing import (
    prepare_data, prepare_test_predictions, ALL_FEATURES,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"
ROOT_DIR = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42


def evaluate_model(model, X, y, model_name="Model", threshold=0.5):
    """Compute comprehensive evaluation metrics."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "total_samples": int(len(y)),
        "total_fraud": int(y.sum()),
        "total_legit": int(len(y) - y.sum()),
    }
    return metrics


def train_all_models(X_train, y_train, X_val, y_val):
    """Train and compare all candidate models."""
    
    models = {}
    
    # 1. DummyClassifier (baseline)
    print("Training DummyClassifier (stratified)...")
    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    models["DummyClassifier"] = dummy
    
    # 2. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr
    
    # 3. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf
    
    # 4. HistGradientBoosting
    print("Training HistGradientBoosting...")
    # HistGBT doesn't support class_weight directly, use sample_weight
    fraud_ratio = y_train.sum() / len(y_train)
    sample_weights = np.where(y_train == 1, 1 / fraud_ratio, 1 / (1 - fraud_ratio))
    sample_weights = sample_weights / sample_weights.mean()  # normalize
    
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    hgb.fit(X_train, y_train, sample_weight=sample_weights)
    models["HistGradientBoosting"] = hgb
    
    # Evaluate all on validation set
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        results[name] = metrics
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  FPR:       {metrics['false_positive_rate']:.4f}")
        print(f"  FNR:       {metrics['false_negative_rate']:.4f}")
    
    return models, results


def select_best_model(models, results):
    """Select best model based on PR-AUC (business-relevant for imbalanced data)."""
    # Exclude dummy from selection
    candidate_results = {k: v for k, v in results.items() if k != "DummyClassifier"}
    best_name = max(candidate_results, key=lambda k: candidate_results[k]["pr_auc"])
    print(f"\nBest model by PR-AUC: {best_name} (PR-AUC={results[best_name]['pr_auc']:.4f})")
    return best_name, models[best_name]


def threshold_analysis(model, X_val, y_val, X_test, y_test):
    """Analyze performance at different thresholds including business cost impact."""
    y_proba_val = model.predict_proba(X_val)[:, 1]
    
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_results = []
    
    for t in thresholds:
        metrics = evaluate_model(model, X_val, y_val, f"threshold={t}", threshold=t)
        flagged = int((y_proba_val >= t).sum())
        metrics["flagged_for_review"] = flagged
        threshold_results.append(metrics)
    
    return threshold_results


def compute_feature_importance(model, preprocessor, model_name):
    """Extract feature importances from the model."""
    # Get feature names after preprocessing
    cat_features = []
    if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
        cat_features = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CATEGORICAL_FEATURES))
    
    all_feature_names = NUMERIC_FEATURES + BINARY_FEATURES + cat_features
    
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    
    if importances is not None and len(importances) == len(all_feature_names):
        fi = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return fi
    
    return None


def generate_evidence_plots(model, X_val, y_val, X_test, y_test, feature_importance_df):
    """Generate and save all evidence plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    EVIDENCE_DIR.mkdir(exist_ok=True)
    
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_proba_val)
    roc_auc = roc_auc_score(y_val, y_proba_val)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Validation Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    
    # 2. Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_val, y_proba_val)
    pr_auc = average_precision_score(y_val, y_proba_val)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
    ax.axhline(y=y_val.mean(), color='k', linestyle='--', alpha=0.5, label=f'Baseline (prevalence={y_val.mean():.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (Validation Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    
    # 3. Confusion Matrix (test set)
    y_pred_test = (y_proba_test >= 0.3).astype(int)
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Legit', 'Predicted Fraud'])
    ax.set_yticklabels(['Actual Legit', 'Actual Fraud'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
    ax.set_title('Confusion Matrix (Test Set, threshold=0.3)')
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    
    # 4. Feature Importance
    if feature_importance_df is not None:
        top_fi = feature_importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_fi)), top_fi['importance'].values, color='steelblue')
        ax.set_yticks(range(len(top_fi)))
        ax.set_yticklabels(top_fi['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(EVIDENCE_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
    
    # 5. Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y_proba_val[y_val == 0], bins=50, alpha=0.7, label='Legitimate', color='green')
    axes[0].hist(y_proba_val[y_val == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[0].set_xlabel('Fraud Risk Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Score Distribution (Validation Set)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(y_proba_test[y_test == 0], bins=50, alpha=0.7, label='Legitimate', color='green')
    axes[1].hist(y_proba_test[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[1].set_xlabel('Fraud Risk Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Score Distribution (Test Set)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)
    
    # 6. Calibration plot
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_val, y_proba_val, n_bins=10, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, 'bo-', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Plot (Validation Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVIDENCE_DIR / "calibration_plot.png", dpi=150)
    plt.close(fig)
    
    print(f"Evidence plots saved to {EVIDENCE_DIR}")


def generate_predictions(model, test_feat, preprocessor):
    """Generate predictions for the unlabeled test set."""
    X_test_unlabeled = prepare_test_predictions(test_feat, preprocessor)
    scores = model.predict_proba(X_test_unlabeled)[:, 1]
    
    predictions = pd.DataFrame({
        "claim_id": test_feat["claim_id"].values,
        "score": scores.round(6)
    })
    
    # Validation checks
    assert len(predictions) == len(test_feat), f"Row count mismatch: {len(predictions)} != {len(test_feat)}"
    assert predictions["score"].between(0, 1).all(), "Scores out of [0,1] range"
    assert predictions["score"].isna().sum() == 0, "NaN scores found"
    assert predictions["claim_id"].nunique() == len(predictions), "Duplicate claim_ids in predictions"
    
    return predictions


def main():
    """Full training pipeline."""
    print("=" * 70)
    print("KESTREL WARRANTY FRAUD DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    MODELS_DIR.mkdir(exist_ok=True)
    EVIDENCE_DIR.mkdir(exist_ok=True)
    
    # Prepare data
    print("\n--- Phase 1: Data Preparation ---")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     test_feat, preprocessor, X_train_df, X_val_df, X_test_df) = prepare_data(temporal_split=True)
    
    # Train models
    print("\n--- Phase 2: Model Training ---")
    models, val_results = train_all_models(X_train, y_train, X_val, y_val)
    
    # Select best model
    print("\n--- Phase 3: Model Selection ---")
    best_name, best_model = select_best_model(models, val_results)
    
    # Evaluate on final test set (ONCE)
    print("\n--- Phase 4: Final Test Evaluation ---")
    test_metrics = evaluate_model(best_model, X_test, y_test, f"{best_name} (Test Set)", threshold=0.3)
    print(f"\nFinal Test Results ({best_name}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    
    # Threshold analysis
    print("\n--- Phase 5: Threshold Analysis ---")
    threshold_results = threshold_analysis(best_model, X_val, y_val, X_test, y_test)
    for tr in threshold_results:
        t = tr['threshold']
        print(f"  t={t:.2f}: P={tr['precision']:.3f} R={tr['recall']:.3f} F1={tr['f1_score']:.3f} Flagged={tr['flagged_for_review']}")
    
    # Feature importance
    print("\n--- Phase 6: Feature Importance ---")
    fi_df = compute_feature_importance(best_model, preprocessor, best_name)
    if fi_df is not None:
        print(fi_df.head(15).to_string())
    
    # Generate evidence plots
    print("\n--- Phase 7: Evidence Plots ---")
    generate_evidence_plots(best_model, X_val, y_val, X_test, y_test, fi_df)
    
    # Save model artifacts
    print("\n--- Phase 8: Saving Artifacts ---")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    
    # Save metadata
    metadata = {
        "model_name": best_name,
        "model_version": "1.0.0",
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "val_metrics": val_results[best_name],
        "test_metrics": test_metrics,
        "threshold_analysis": threshold_results,
    }
    
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save all comparison results
    with open(EVIDENCE_DIR / "model_comparison.json", "w") as f:
        json.dump(val_results, f, indent=2)
    
    # Feature importance
    if fi_df is not None:
        fi_df.to_csv(EVIDENCE_DIR / "feature_importance.csv", index=False)
    
    # Save threshold analysis
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(EVIDENCE_DIR / "threshold_analysis.csv", index=False)
    
    # Generate predictions
    print("\n--- Phase 9: Generating Predictions ---")
    predictions = generate_predictions(best_model, test_feat, preprocessor)
    predictions.to_csv(ROOT_DIR / "predictions.csv", index=False)
    print(f"Predictions saved: {len(predictions)} rows")
    print(f"Score range: [{predictions['score'].min():.6f}, {predictions['score'].max():.6f}]")
    print(f"Mean score: {predictions['score'].mean():.6f}")
    
    # Validation report
    print("\n--- Phase 10: Validation Report ---")
    print(f"predictions.csv row count: {len(predictions)}")
    print(f"test_unlabelled.csv row count: {len(test_feat)}")
    print(f"Match: {len(predictions) == len(test_feat)}")
    print(f"All scores in [0,1]: {predictions['score'].between(0,1).all()}")
    print(f"No NaN scores: {predictions['score'].isna().sum() == 0}")
    print(f"Unique claim_ids: {predictions['claim_id'].nunique()} of {len(predictions)}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return best_model, preprocessor, metadata, predictions


if __name__ == "__main__":
    main()
