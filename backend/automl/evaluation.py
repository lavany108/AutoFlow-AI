import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[Any]] = None,
    is_fraud_task: bool = False,
) -> Dict[str, Any]:
    """
    Computes classification metrics.
    When is_fraud_task=True, adds F2, PR-AUC, specificity, and imbalance-aware metrics.
    """
    metrics = {
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
        "recall": float(round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
        "f1": float(round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)),
        "f1_macro": float(round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)),
    }

    # Binary-specific fraud metrics
    n_classes = len(np.unique(y_true))
    if is_fraud_task or n_classes == 2:
        # F2 score — weighted towards recall (catching fraud matters more than false alarms)
        try:
            f2 = float(round(fbeta_score(y_true, y_pred, beta=2, average="binary", zero_division=0), 4))
        except Exception:
            f2 = float(round(fbeta_score(y_true, y_pred, beta=2, average="weighted", zero_division=0), 4))
        metrics["f2"] = f2

        # Specificity (true negative rate)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                metrics["specificity"] = float(round(tn / max(tn + fp, 1), 4))
                metrics["detection_rate"] = float(round(tp / max(tp + fn, 1), 4))  # same as recall for binary
        except Exception:
            pass

    # ROC AUC
    if y_proba is not None:
        try:
            if n_classes == 2:
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    score = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    score = roc_auc_score(y_true, y_proba)
                metrics["roc_auc"] = float(round(score, 4))
            elif n_classes > 2 and y_proba.ndim == 2:
                score = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                metrics["roc_auc"] = float(round(score, 4))
            else:
                metrics["roc_auc"] = None
        except Exception as e:
            logger.debug("ROC AUC skipped: %s", e)
            metrics["roc_auc"] = None

        # PR-AUC — primary metric for fraud (imbalanced binary)
        if n_classes == 2:
            try:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                unique_vals = sorted(np.unique(y_true).tolist(), key=str)
                if set(unique_vals).issubset({0, 1}):
                    y_true_bin = (y_true == 1).astype(int)
                else:
                    pos_candidate = next((c for c in unique_vals if str(c).strip().lower() in ["1", "true", "yes", "fraud"]), unique_vals[-1])
                    y_true_bin = (y_true == pos_candidate).astype(int)

                pr_auc = float(round(average_precision_score(y_true_bin, proba_pos), 4))
                metrics["pr_auc"] = pr_auc

                # Compute Precision-Recall curve for threshold optimization & visualization
                precisions, recalls, thresholds = precision_recall_curve(y_true_bin, proba_pos)
                step = max(1, len(thresholds) // 50)
                metrics["pr_curve"] = {
                    "precision": [float(round(p, 4)) for p in precisions[::step].tolist()],
                    "recall": [float(round(r, 4)) for r in recalls[::step].tolist()],
                    "thresholds": [float(round(t, 4)) for t in thresholds[::step].tolist()],
                }
                metrics["fraud_count"] = int(np.sum(y_true_bin == 1))
                metrics["legit_count"] = int(np.sum(y_true_bin == 0))
            except Exception as e:
                logger.debug("PR-AUC skipped: %s", e)
                metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    # Confusion Matrix
    try:
        label_set = classes if classes is not None else sorted(np.unique(y_true).tolist(), key=str)
        cm = confusion_matrix(y_true, y_pred, labels=label_set)
        metrics["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": [str(c) for c in label_set],
        }
    except Exception as e:
        logger.debug("Confusion matrix error: %s", e)
        metrics["confusion_matrix"] = None

    return metrics


def get_primary_metric(task_type: str = "classification", metrics: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
    """
    Returns (primary_metric_name, value) for ranking fraud detection models.
    For fraud classification: PR-AUC > F2 > F1 > Accuracy.
    """
    import math
    if metrics is None:
        metrics = {}

    for k in ["pr_auc", "f2", "f1", "accuracy"]:
        val = metrics.get(k)
        if val is not None:
            try:
                f = float(val)
                if not (math.isnan(f) or math.isinf(f)):
                    return k, f
            except (ValueError, TypeError):
                pass
    return "f1", 0.0
