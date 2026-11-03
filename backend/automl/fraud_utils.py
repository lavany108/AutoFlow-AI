"""
Fraud Detection Utilities
Covers:
- Imbalanced data handling (class_weight, SMOTE-based resampling)
- Decision threshold optimization (F-beta, max-precision@recall, business cost)
- Fraud-specific evaluation metrics (PR-AUC, F2, average_precision, specificity)
- Business impact calculator (savings vs. false-positive cost)
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    fbeta_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imbalance Handling
# ---------------------------------------------------------------------------

def compute_class_weights(y: pd.Series) -> Dict[Any, float]:
    """Compute balanced class weights for sklearn estimators."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def try_smote_resample(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    target_minority_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Attempt SMOTE resampling.  Falls back gracefully if imblearn is unavailable
    or the minority class is too small.

    Returns (X_res, y_res, was_resampled).
    """
    try:
        from imblearn.over_sampling import SMOTE
        counts = y.value_counts()
        minority_count = counts.min()
        majority_count = counts.max()
        ratio = minority_count / (minority_count + majority_count)

        # Only resample if imbalance ratio < target and enough samples
        if ratio < target_minority_ratio and minority_count >= 6:
            # If dataset has plenty of minority samples (>=1000) or large dataset (>25k rows),
            # balanced class weighting is preferred, much faster, and avoids heavy k-NN synthesis
            if minority_count >= 1000 or len(X) > 25000:
                logger.info(
                    "Sufficient minority samples (%d) in large dataset (%d rows); using balanced class weighting instead of SMOTE.",
                    minority_count, len(X)
                )
                return X, y, False

            desired_minority = int(majority_count * target_minority_ratio)
            sampling_strategy = {
                counts.idxmin(): max(desired_minority, minority_count)
            }
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=min(5, minority_count - 1),
            )
            X_arr, y_arr = smote.fit_resample(X, y)
            logger.info(
                "SMOTE resampled: %d → %d minority samples", minority_count, y_arr.tolist().count(counts.idxmin())
            )
            return pd.DataFrame(X_arr, columns=X.columns), pd.Series(y_arr, name=y.name), True
        return X, y, False
    except Exception as e:
        logger.warning("SMOTE failed, using original data: %s", e)
        return X, y, False


# ---------------------------------------------------------------------------
# Fraud Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_fraud_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,  # probability of positive/fraud class
    threshold: float = 0.5,
    cost_fp: float = 10.0,   # $cost of investigating a false positive (wasted analyst time)
    cost_fn: float = 500.0,  # $cost of missing a fraud (chargeback + liability)
    avg_fraud_amount: float = 500.0,
) -> Dict[str, Any]:
    """
    Compute a comprehensive suite of fraud-detection metrics.

    Args:
        y_true: Ground truth labels (0=legit, 1=fraud)
        y_proba: Model probability scores for fraud class
        threshold: Decision boundary
        cost_fp: Cost per false positive (false alarm)
        cost_fn: Cost per false negative (missed fraud)
        avg_fraud_amount: Average transaction amount of fraudulent transactions
    """
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    total = len(y_true)
    fraud_count = int(np.sum(y_true == 1))
    legit_count = total - fraud_count
    imbalance_ratio = fraud_count / max(total, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)  # sensitivity / detection rate
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # PR-AUC (Average Precision) — key metric for imbalanced fraud data
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = None

    # ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = None

    # Business cost at current threshold
    total_cost = fp * cost_fp + fn * cost_fn
    # Savings vs. no-model baseline (catching 0 frauds)
    baseline_cost = fraud_count * cost_fn
    savings = baseline_cost - total_cost
    # Amount of fraud caught
    fraud_caught_amount = tp * avg_fraud_amount

    # Precision-Recall curve data (for frontend plotting)
    try:
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_proba)
        pr_curve = {
            "precision": [round(float(p), 4) for p in prec_arr[:100]],
            "recall": [round(float(r), 4) for r in rec_arr[:100]],
            "thresholds": [round(float(t), 4) for t in thresh_arr[:99]] + [1.0],
        }
    except Exception:
        pr_curve = {"precision": [], "recall": [], "thresholds": []}

    # ROC curve data
    try:
        fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, y_proba)
        roc_curve_data = {
            "fpr": [round(float(f), 4) for f in fpr_arr[:100]],
            "tpr": [round(float(t), 4) for t in tpr_arr[:100]],
        }
    except Exception:
        roc_curve_data = {"fpr": [], "tpr": []}

    return {
        # Core counts
        "total_samples": int(total),
        "fraud_count": int(fraud_count),
        "legit_count": int(legit_count),
        "imbalance_ratio": float(round(imbalance_ratio, 4)),
        # Confusion matrix
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        # Classification metrics
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),        # = detection rate
        "specificity": float(round(specificity, 4)),
        "f1": float(round(f1, 4)),
        "f2": float(round(f2, 4)),                # recall-weighted F-score
        "roc_auc": float(round(roc_auc, 4)) if roc_auc else None,
        "pr_auc": float(round(pr_auc, 4)) if pr_auc else None,
        "threshold": float(threshold),
        # Business impact
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "total_operational_cost": float(round(total_cost, 2)),
        "baseline_cost_no_model": float(round(baseline_cost, 2)),
        "estimated_savings": float(round(savings, 2)),
        "fraud_amount_caught": float(round(fraud_caught_amount, 2)),
        # Curve data for plots
        "pr_curve": pr_curve,
        "roc_curve": roc_curve_data,
        "confusion_matrix": {"matrix": cm.tolist(), "labels": ["Legit", "Fraud"]},
    }


# ---------------------------------------------------------------------------
# Threshold Optimization
# ---------------------------------------------------------------------------

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f2",   # 'f1', 'f2', 'cost', 'precision_at_recall'
    min_recall: float = 0.80,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
) -> Dict[str, Any]:
    """
    Find optimal decision threshold under various strategies:
    - f1: maximize F1
    - f2: maximize F2 (recall-weighted)
    - cost: minimize cost_fp*FP + cost_fn*FN
    - precision_at_recall: max precision subject to recall >= min_recall
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresh)+1 precision/recall values
    # Last entry is precision=1.0, recall=0.0 with no corresponding threshold
    thresh_extended = np.append(thresh_arr, 1.0)

    best_threshold = 0.5
    best_score = -np.inf
    threshold_scores = []

    for i, t in enumerate(thresh_extended[:-1]):
        y_pred = (y_proba >= t).astype(int)
        p = prec_arr[i]
        r = rec_arr[i]

        if strategy == "f2":
            score = (1 + 4) * p * r / max(4 * p + r, 1e-10)
        elif strategy == "f1":
            score = 2 * p * r / max(p + r, 1e-10)
        elif strategy == "cost":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            score = -(fp * cost_fp + fn * cost_fn)
        elif strategy == "precision_at_recall":
            if r >= min_recall:
                score = p
            else:
                score = -np.inf
        else:
            score = (1 + 4) * p * r / max(4 * p + r, 1e-10)

        threshold_scores.append({
            "threshold": float(round(float(t), 4)),
            "precision": float(round(float(p), 4)),
            "recall": float(round(float(r), 4)),
            "score": float(round(float(score), 6)) if np.isfinite(score) else None,
        })

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_threshold = float(t)

    # Compute metrics at optimized threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    f2 = fbeta_score(y_true, y_pred_opt, beta=2, zero_division=0)
    f1 = fbeta_score(y_true, y_pred_opt, beta=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision_opt = tp / max(tp + fp, 1)
    recall_opt = tp / max(tp + fn, 1)

    return {
        "strategy": strategy,
        "optimal_threshold": float(round(best_threshold, 4)),
        "optimal_precision": float(round(precision_opt, 4)),
        "optimal_recall": float(round(recall_opt, 4)),
        "optimal_f1": float(round(f1, 4)),
        "optimal_f2": float(round(f2, 4)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold_scores": threshold_scores[:200],  # cap for JSON size
    }


# ---------------------------------------------------------------------------
# Imbalance statistics helper
# ---------------------------------------------------------------------------

def compute_imbalance_stats(y: pd.Series) -> Dict[str, Any]:
    """Return class distribution stats for display."""
    counts = y.value_counts()
    total = len(y)

    fraud_count = 0
    legit_count = 0

    for k, v in counts.items():
        k_str = str(k).strip().lower()
        if k_str in ["1", "true", "yes", "fraud", "y"]:
            fraud_count = int(v)
        elif k_str in ["0", "false", "no", "legit", "n"]:
            legit_count = int(v)

    if fraud_count == 0 and legit_count == 0 and len(counts) == 2:
        fraud_count = int(counts.min())
        legit_count = int(counts.max())
    elif fraud_count == 0 and legit_count > 0:
        fraud_count = total - legit_count
    elif legit_count == 0 and fraud_count > 0:
        legit_count = total - fraud_count

    ratio = fraud_count / max(total, 1)

    if ratio < 0.01:
        severity = "Extreme (< 1%)"
    elif ratio < 0.05:
        severity = "High (1-5%)"
    elif ratio < 0.15:
        severity = "Moderate (5-15%)"
    else:
        severity = "Mild (> 15%)"

    return {
        "total": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_ratio": float(round(ratio, 4)),
        "imbalance_severity": severity,
        "recommended_strategy": "SMOTE + class_weight" if ratio < 0.10 else "class_weight only",
    }
