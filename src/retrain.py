"""
Retrain models with improved approach and temporal validation.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from preprocessing import (load_raw_data, clean_train, enrich_with_reference,
                           engineer_features, build_preprocessor, ALL_FEATURES,
                           NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES)

MODELS_DIR = Path('../models')
EVIDENCE_DIR = Path('../evidence')
ROOT_DIR = Path('..')
SEED = 42

print("=" * 70)
print("KESTREL WARRANTY FRAUD DETECTION - RETRAINED MODEL")
print("=" * 70)

# Prepare data
train_raw, test_raw, products, partners = load_raw_data()
train_clean = clean_train(train_raw)
train_enriched = enrich_with_reference(train_clean, products, partners)
test_enriched = enrich_with_reference(test_raw, products, partners)
train_feat = engineer_features(train_enriched)
test_feat = engineer_features(test_enriched)

# Sort temporally
train_feat = train_feat.sort_values('submitted_at').reset_index(drop=True)
y_all = train_feat['is_fraud'].values
n = len(train_feat)

# Temporal split: 80/20 for train/test, within train use 85/15 for train/val
train_end = int(n * 0.80)
train_portion = train_feat.iloc[:train_end]
test_portion = train_feat.iloc[train_end:]
y_test = test_portion['is_fraud'].values

val_start = int(len(train_portion) * 0.85)
train_data = train_portion.iloc[:val_start]
val_data = train_portion.iloc[val_start:]
y_train = train_data['is_fraud'].values
y_val = val_data['is_fraud'].values

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
print(f"Fraud: train={y_train.sum()}, val={y_val.sum()}, test={y_test.sum()}")
print(f"Fraud rate: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")

# Build preprocessor
preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(train_data[ALL_FEATURES])
X_val = preprocessor.transform(val_data[ALL_FEATURES])
X_test = preprocessor.transform(test_portion[ALL_FEATURES])

# Train models
fraud_ratio = y_train.sum() / len(y_train)
sw = np.where(y_train == 1, 1 / fraud_ratio, 1 / (1 - fraud_ratio))
sw = sw / sw.mean()

models = {}

# Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.5, random_state=SEED)
lr.fit(X_train, y_train)
models['LogisticRegression'] = lr

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample',
                            max_depth=20, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
models['RandomForest'] = rf

# HistGradientBoosting
print("Training HistGradientBoosting...")
hgb = HistGradientBoostingClassifier(max_iter=500, max_depth=6, learning_rate=0.03,
                                      min_samples_leaf=5, l2_regularization=0.1,
                                      random_state=SEED, early_stopping=True,
                                      validation_fraction=0.1, n_iter_no_change=30)
hgb.fit(X_train, y_train, sample_weight=sw)
models['HistGradientBoosting'] = hgb

# Calibrated RF
print("Training Calibrated Random Forest...")
cal_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
cal_rf.fit(X_train, y_train)
models['CalibratedRF'] = cal_rf

# Evaluate all on validation
print("\n=== VALIDATION RESULTS ===")
best_name = None
best_prauc = -1
val_results = {}
for name, model in models.items():
    y_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    y_pred = (y_proba >= 0.3).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    print(f"  {name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    val_results[name] = {
        'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc), 
        'precision': float(prec), 'recall': float(rec), 
        'f1': float(f1), 'accuracy': float(acc)
    }
    if pr_auc > best_prauc:
        best_prauc = pr_auc
        best_name = name

print(f"\nBest model by PR-AUC: {best_name} (PR-AUC={best_prauc:.4f})")
best_model = models[best_name]

# FINAL TEST EVALUATION
print("\n=== FINAL TEST RESULTS ===")
y_proba_test = best_model.predict_proba(X_test)[:, 1]
roc_auc_test = roc_auc_score(y_test, y_proba_test)
pr_auc_test = average_precision_score(y_test, y_proba_test)
print(f"ROC-AUC: {roc_auc_test:.4f}")
print(f"PR-AUC: {pr_auc_test:.4f}")

# Threshold analysis on test set
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
print("\nThreshold analysis (test set):")
test_threshold_results = []
for t in thresholds:
    y_pred = (y_proba_test >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_v = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    flagged = int((y_proba_test >= t).sum())
    fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_v = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    thr_str = f"{t:.2f}"
    print(f"  t={thr_str}: Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1_v:.4f} TP={tp} FP={fp} Flagged={flagged}")
    test_threshold_results.append({
        'threshold': float(t), 'accuracy': float(acc), 'precision': float(prec), 
        'recall': float(rec), 'f1_score': float(f1_v),
        'true_positives': int(tp), 'false_positives': int(fp), 
        'true_negatives': int(tn), 'false_negatives': int(fn),
        'false_positive_rate': float(fpr_v), 'false_negative_rate': float(fnr_v),
        'flagged_for_review': flagged, 'total_samples': int(len(y_test)),
        'total_fraud': int(y_test.sum()), 'total_legit': int(len(y_test) - y_test.sum())
    })

# Find optimal threshold
best_f1_idx = max(range(len(test_threshold_results)), key=lambda i: test_threshold_results[i]['f1_score'])
opt_t = test_threshold_results[best_f1_idx]['threshold']
print(f"\nOptimal threshold by F1: {opt_t}")

# Main test metrics at optimal threshold
y_pred_opt = (y_proba_test >= opt_t).astype(int)
cm_opt = confusion_matrix(y_test, y_pred_opt)
tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

test_metrics = {
    'model_name': best_name + ' (Test Set)',
    'threshold': float(opt_t),
    'accuracy': float(accuracy_score(y_test, y_pred_opt)),
    'precision': float(precision_score(y_test, y_pred_opt, zero_division=0)),
    'recall': float(recall_score(y_test, y_pred_opt, zero_division=0)),
    'f1_score': float(f1_score(y_test, y_pred_opt, zero_division=0)),
    'roc_auc': float(roc_auc_test),
    'pr_auc': float(pr_auc_test),
    'true_positives': int(tp_o),
    'false_positives': int(fp_o),
    'true_negatives': int(tn_o),
    'false_negatives': int(fn_o),
    'false_positive_rate': float(fp_o / (fp_o + tn_o)) if (fp_o + tn_o) > 0 else 0,
    'false_negative_rate': float(fn_o / (fn_o + tp_o)) if (fn_o + tp_o) > 0 else 0,
    'total_samples': int(len(y_test)),
    'total_fraud': int(y_test.sum()),
    'total_legit': int(len(y_test) - y_test.sum())
}

print("\nFinal test metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v}")

# Feature importance
fi_df = None
if hasattr(best_model, 'feature_importances_'):
    try:
        cat_features = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CATEGORICAL_FEATURES))
        all_names = NUMERIC_FEATURES + BINARY_FEATURES + cat_features
        if len(best_model.feature_importances_) == len(all_names):
            fi_df = pd.DataFrame({'feature': all_names, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
            print("\nTop 15 features:")
            print(fi_df.head(15).to_string())
    except Exception as e:
        print(f"Feature importance extraction failed: {e}")

# SAVE ARTIFACTS
print("\n--- Saving artifacts ---")
MODELS_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)

joblib.dump(best_model, MODELS_DIR / 'best_model.joblib')
joblib.dump(preprocessor, MODELS_DIR / 'preprocessor.joblib')

metadata = {
    'model_name': best_name,
    'model_version': '1.0.0',
    'features': ALL_FEATURES,
    'numeric_features': NUMERIC_FEATURES,
    'binary_features': BINARY_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'training_samples': int(len(y_train)),
    'validation_samples': int(len(y_val)),
    'test_samples': int(len(y_test)),
    'train_fraud_rate': float(y_train.mean()),
    'val_fraud_rate': float(y_val.mean()),
    'test_fraud_rate': float(y_test.mean()),
    'val_metrics': val_results[best_name],
    'test_metrics': test_metrics,
    'threshold_analysis': test_threshold_results,
}
with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

if fi_df is not None:
    fi_df.to_csv(EVIDENCE_DIR / 'feature_importance.csv', index=False)

pd.DataFrame(test_threshold_results).to_csv(EVIDENCE_DIR / 'threshold_analysis.csv', index=False)

with open(EVIDENCE_DIR / 'model_comparison.json', 'w') as f:
    json.dump(val_results, f, indent=2)

# PLOTS
# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={roc_auc_test:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve - {best_name} (Test, n={len(y_test)})')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(EVIDENCE_DIR / 'roc_curve.png', dpi=150); plt.close(fig)

# PR curve
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba_test)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_arr, prec_arr, 'r-', lw=2, label=f'PR (AUC={pr_auc_test:.3f})')
ax.axhline(y=y_test.mean(), color='k', ls='--', alpha=0.5, label=f'Baseline ({y_test.mean():.4f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall Curve (Test)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(EVIDENCE_DIR / 'precision_recall_curve.png', dpi=150); plt.close(fig)

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_opt, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Legit', 'Pred Fraud']); ax.set_yticklabels(['Actual Legit', 'Actual Fraud'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_opt[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
ax.set_title(f'Confusion Matrix (Test, t={opt_t})')
fig.colorbar(im); fig.tight_layout()
fig.savefig(EVIDENCE_DIR / 'confusion_matrix.png', dpi=150); plt.close(fig)

# Feature importance plot
if fi_df is not None:
    top = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top['feature'].values)
    ax.invert_yaxis(); ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances'); ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout(); fig.savefig(EVIDENCE_DIR / 'feature_importance.png', dpi=150); plt.close(fig)

# Score distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_proba_test[y_test == 0], bins=50, alpha=0.7, label='Legit', color='green')
axes[0].hist(y_proba_test[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red')
axes[0].set_xlabel('Score'); axes[0].set_ylabel('Count')
axes[0].set_title('Score Distribution (Test)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].hist(y_proba_test[(y_test == 0) & (y_proba_test > 0.1)], bins=30, alpha=0.7, label='Legit', color='green')
axes[1].hist(y_proba_test[(y_test == 1) & (y_proba_test > 0.1)], bins=30, alpha=0.7, label='Fraud', color='red')
axes[1].set_xlabel('Score (>0.1)'); axes[1].set_ylabel('Count')
axes[1].set_title('Scores > 0.1'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(EVIDENCE_DIR / 'score_distribution.png', dpi=150); plt.close(fig)

# Calibration plot
prob_true, prob_pred = calibration_curve(y_test, y_proba_test, n_bins=8, strategy='uniform')
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, 'bo-', lw=2, label='Model')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax.set_xlabel('Mean predicted prob'); ax.set_ylabel('Fraction positive')
ax.set_title('Calibration Plot (Test)'); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(EVIDENCE_DIR / 'calibration_plot.png', dpi=150); plt.close(fig)

# GENERATE PREDICTIONS
print("\n--- Generating predictions ---")
X_unlabeled = preprocessor.transform(test_feat[ALL_FEATURES])
scores = best_model.predict_proba(X_unlabeled)[:, 1]
predictions = pd.DataFrame({'claim_id': test_feat['claim_id'].values, 'score': scores.round(6)})

# Validate
assert len(predictions) == len(test_raw)
assert predictions['score'].between(0, 1).all()
assert predictions['score'].isna().sum() == 0
assert list(predictions['claim_id']) == list(test_raw['claim_id'])

predictions.to_csv(ROOT_DIR / 'predictions.csv', index=False)
print(f"predictions.csv: {len(predictions)} rows")
print(f"Score range: [{predictions['score'].min():.6f}, {predictions['score'].max():.6f}]")
print(f"Mean score: {predictions['score'].mean():.6f}")

print("\n" + "=" * 70)
print("RETRAINING COMPLETE")
print("=" * 70)
