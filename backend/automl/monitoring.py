import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from backend.automl.evaluation import evaluate_classification

logger = logging.getLogger(__name__)

def detect_numerical_drift(
    ref_samples: List[float],
    current_values: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs 2-sample Kolmogorov-Smirnov test between reference and current values.
    """
    clean_ref = [x for x in ref_samples if x is not None and not np.isnan(x)]
    clean_curr = [x for x in current_values if x is not None and not np.isnan(x)]

    if len(clean_curr) < 5 or len(clean_ref) < 5:
        # Not enough data for meaningful KS test
        return {
            "statistic_name": "Kolmogorov-Smirnov",
            "statistic_value": 0.0,
            "p_value": 1.0,
            "is_drifted": False,
            "ref_mean_or_mode": float(round(np.mean(clean_ref), 2)) if clean_ref else 0.0,
            "curr_mean_or_mode": float(round(np.mean(clean_curr), 2)) if clean_curr else 0.0,
            "sample_size": len(clean_curr),
            "note": "Insufficient sample size (minimum 5 predictions required)."
        }

    try:
        ks_res = stats.ks_2samp(clean_ref, clean_curr)
        stat_val = float(round(ks_res.statistic, 4))
        p_val = float(round(ks_res.pvalue, 4))
        # Practical drift check: p-value < alpha AND statistic > 0.15 to avoid false positives on large samples
        is_drifted = (p_val < alpha) and (stat_val > 0.15)

        return {
            "statistic_name": "Kolmogorov-Smirnov",
            "statistic_value": stat_val,
            "p_value": p_val,
            "is_drifted": is_drifted,
            "ref_mean_or_mode": float(round(np.mean(clean_ref), 2)),
            "curr_mean_or_mode": float(round(np.mean(clean_curr), 2)),
            "sample_size": len(clean_curr),
            "note": "Drift detected" if is_drifted else "Distribution stable"
        }
    except Exception as e:
        logger.warning("KS test failed: %s", e)
        return {
            "statistic_name": "Kolmogorov-Smirnov",
            "statistic_value": 0.0,
            "p_value": 1.0,
            "is_drifted": False,
            "ref_mean_or_mode": 0.0,
            "curr_mean_or_mode": 0.0,
            "sample_size": len(clean_curr),
            "note": f"Test error: {str(e)}"
        }


def detect_categorical_drift(
    ref_frequencies: Dict[str, float],
    ref_mode: str,
    current_values: List[Any],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs Chi-square goodness-of-fit test comparing observed category counts
    against reference category probability distribution.
    """
    clean_curr = [str(x) for x in current_values if x is not None and str(x) != "nan"]

    if len(clean_curr) < 10:
        return {
            "statistic_name": "Chi-Square",
            "statistic_value": 0.0,
            "p_value": 1.0,
            "is_drifted": False,
            "ref_mean_or_mode": ref_mode,
            "curr_mean_or_mode": str(clean_curr[0]) if clean_curr else "",
            "sample_size": len(clean_curr),
            "note": "Insufficient sample size (minimum 10 predictions required)."
        }

    # Count observed occurrences
    curr_counts = pd.Series(clean_curr).value_counts()
    curr_mode = str(curr_counts.index[0]) if not curr_counts.empty else ""

    all_categories = list(set(list(ref_frequencies.keys()) + list(curr_counts.index)))

    observed = []
    expected = []
    total_curr = len(clean_curr)

    for cat in all_categories:
        obs = curr_counts.get(cat, 0)
        # Expected probability from reference
        prob = ref_frequencies.get(cat, 0.001)  # small epsilon for unseen categories
        exp = prob * total_curr
        observed.append(obs)
        expected.append(exp)

    # Normalize expected to sum exactly to sum(observed)
    expected = np.array(expected)
    observed = np.array(observed)
    expected = expected * (np.sum(observed) / np.sum(expected))

    try:
        chi_res = stats.chisquare(observed, expected)
        stat_val = float(round(chi_res.statistic, 4))
        p_val = float(round(chi_res.pvalue, 4))
        is_drifted = (p_val < alpha)

        return {
            "statistic_name": "Chi-Square",
            "statistic_value": stat_val,
            "p_value": p_val,
            "is_drifted": is_drifted,
            "ref_mean_or_mode": ref_mode,
            "curr_mean_or_mode": curr_mode,
            "sample_size": total_curr,
            "note": "Categorical frequency shift detected" if is_drifted else "Category distribution stable"
        }
    except Exception as e:
        logger.warning("Chi-square test failed: %s", e)
        return {
            "statistic_name": "Chi-Square",
            "statistic_value": 0.0,
            "p_value": 1.0,
            "is_drifted": False,
            "ref_mean_or_mode": ref_mode,
            "curr_mean_or_mode": curr_mode,
            "sample_size": total_curr,
            "note": f"Test error: {str(e)}"
        }


def analyze_drift_and_latency(
    reference_distribution: Dict[str, Any],
    recent_prediction_records: List[Dict[str, Any]],
    threshold_pvalue: float = 0.05
) -> Dict[str, Any]:
    """
    Computes drift across all features and summarizes prediction latency.
    """
    total_predictions = len(recent_prediction_records)
    if total_predictions == 0:
        return {
            "recent_predictions_analyzed": 0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "drifted_features_count": 0,
            "total_features_monitored": len(reference_distribution),
            "drift_detected": False,
            "feature_drift": []
        }

    # Latency calculations
    latencies = [rec.get("latency_ms", 0.0) for rec in recent_prediction_records]
    mean_latency = float(round(np.mean(latencies), 2)) if latencies else 0.0
    p95_latency = float(round(np.percentile(latencies, 95), 2)) if latencies else 0.0

    # Build DataFrame from incoming prediction features
    feature_rows = [rec.get("features", {}) for rec in recent_prediction_records]
    curr_df = pd.DataFrame(feature_rows)

    drift_items = []
    drifted_count = 0

    for feat_name, ref_info in reference_distribution.items():
        feat_type = ref_info.get("type", "numerical")

        if feat_name not in curr_df.columns:
            continue

        feat_values = curr_df[feat_name].tolist()

        if feat_type == "numerical":
            ref_samples = ref_info.get("sample_values", [])
            drift_res = detect_numerical_drift(ref_samples, feat_values, alpha=threshold_pvalue)
        else:
            ref_freq = ref_info.get("frequencies", {})
            ref_mode = ref_info.get("mode", "")
            drift_res = detect_categorical_drift(ref_freq, ref_mode, feat_values, alpha=threshold_pvalue)

        is_drift = drift_res["is_drifted"]
        if is_drift:
            drifted_count += 1

        drift_items.append({
            "feature": feat_name,
            "feature_type": feat_type,
            "statistic_name": drift_res["statistic_name"],
            "statistic_value": drift_res["statistic_value"],
            "p_value": drift_res["p_value"],
            "threshold": threshold_pvalue,
            "is_drifted": is_drift,
            "ref_mean_or_mode": drift_res["ref_mean_or_mode"],
            "curr_mean_or_mode": drift_res["curr_mean_or_mode"],
            "note": drift_res.get("note", "")
        })

    # Overall drift is flagged if 2 or more features or >= 25% of monitored features drifted
    drift_detected = (drifted_count >= 2) or (len(drift_items) > 0 and (drifted_count / len(drift_items)) >= 0.25)

    return {
        "recent_predictions_analyzed": total_predictions,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "drifted_features_count": drifted_count,
        "total_features_monitored": len(drift_items),
        "drift_detected": drift_detected,
        "feature_drift": drift_items
    }


def evaluate_concept_drift(
    logged_predictions_with_labels: List[Dict[str, Any]],
    task_type: str,
    baseline_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates concept drift when delayed ground-truth labels are provided.
    Returns:
    - concept_drift_status: 'unknown', 'evaluated', 'degraded'
    - current_metrics
    - degradation
    """
    if len(logged_predictions_with_labels) < 5:
        return {
            "status": "unknown",
            "message": "Fewer than 5 labeled outcomes available. Concept drift status cannot be determined.",
            "metrics": None
        }

    y_true = [rec["actual_label"] for rec in logged_predictions_with_labels]
    y_pred = [rec["prediction"] for rec in logged_predictions_with_labels]

    curr_metrics = evaluate_classification(np.array(y_true), np.array(y_pred), is_fraud_task=True)
    baseline_f1 = baseline_metrics.get("f1", 0.0)
    curr_f1 = curr_metrics.get("f1", 0.0)
    drop = baseline_f1 - curr_f1
    is_degraded = drop > 0.10  # >10% drop in F1-score

    return {
        "status": "degraded" if is_degraded else "evaluated",
        "is_degraded": is_degraded,
        "current_metrics": curr_metrics,
        "baseline_metrics": baseline_metrics,
        "performance_drop": float(round(drop, 4)),
        "message": f"Performance drop: {drop:.4f}. {'Significant concept drift detected!' if is_degraded else 'Performance remains within acceptable limits.'}"
    }
