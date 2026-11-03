import io
import base64
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def generate_shap_explainability(
    pipeline: Any,
    X_sample: pd.DataFrame,
    task_type: str,
    algorithm_name: str,
    max_samples: int = 100
) -> Dict[str, Any]:
    """
    Computes global feature importance using SHAP.
    Safely unwraps the ColumnTransformer and estimator.
    Returns:
    - explainer_type
    - sample_size
    - feature_importance list (feature, importance, mean_abs_shap, rank)
    - summary_plot_base64
    - limitations note
    """
    # Sample data to keep computation fast and laptop-friendly
    if len(X_sample) > max_samples:
        X_eval = X_sample.sample(max_samples, random_state=42)
    else:
        X_eval = X_sample.copy()

    preprocessor = pipeline.named_steps.get("preprocessor")
    estimator = pipeline.named_steps.get("estimator")

    if not preprocessor or not estimator:
        raise ValueError("Pipeline must have 'preprocessor' and 'estimator' steps.")

    # Transform features through preprocessor
    X_transformed = preprocessor.transform(X_eval)
    
    # Retrieve transformed feature names
    feature_names = []
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback names
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Clean prefix names like 'num__' or 'cat__' for cleaner display
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "") for name in feature_names
    ]

    explainer_type = "shap.Explainer"
    shap_values = None
    limitations = None

    try:
        # Tree-based explainer
        if any(tree_algo in algorithm_name for tree_algo in ["Random Forest", "Extra Trees", "HistGradientBoosting"]):
            explainer_type = "TreeExplainer"
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_transformed)
        elif any(linear_algo in algorithm_name for linear_algo in ["Logistic", "Linear"]):
            explainer_type = "LinearExplainer"
            # Use small background sample for linear explainer
            background = X_transformed[:min(20, len(X_transformed))]
            explainer = shap.LinearExplainer(estimator, background)
            shap_values = explainer.shap_values(X_transformed)
        else:
            # Fallback to model agnostic KernelExplainer or general Explainer
            explainer_type = "KernelExplainer (Sampled)"
            background = shap.sample(X_transformed, min(15, len(X_transformed)), random_state=42)
            if hasattr(estimator, "predict_proba") and task_type == "classification":
                explainer = shap.KernelExplainer(estimator.predict_proba, background)
            else:
                explainer = shap.KernelExplainer(estimator.predict, background)
            shap_values = explainer.shap_values(X_transformed[:min(30, len(X_transformed))], nsamples=50)

    except Exception as e:
        logger.warning("Primary SHAP explainer failed (%s). Attempting fallback shap.Explainer: %s", explainer_type, e)
        try:
            explainer_type = "shap.Explainer (Permutation Fallback)"
            background = shap.sample(X_transformed, min(20, len(X_transformed)), random_state=42)
            explainer = shap.Explainer(estimator.predict, background)
            explanation = explainer(X_transformed[:min(30, len(X_transformed))])
            shap_values = explanation.values
        except Exception as e2:
            logger.error("All SHAP explainers failed: %s", e2)
            return {
                "explainer_type": "Unavailable",
                "sample_size": len(X_eval),
                "feature_importance": [],
                "summary_plot_base64": None,
                "limitations": f"SHAP explanation could not be computed for {algorithm_name}: {str(e2)}. Model predictions remain valid."
            }

    # Extract mean absolute SHAP values per feature
    if isinstance(shap_values, list):
        # Multiclass classification: average across classes or take positive class
        if len(shap_values) == 2:
            # Binary classification: positive class
            abs_shap = np.abs(shap_values[1])
        else:
            abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            abs_shap = np.mean(np.abs(shap_values), axis=2)
        else:
            abs_shap = np.abs(shap_values)
    else:
        abs_shap = np.abs(np.array(shap_values))

    mean_abs_shap = np.mean(abs_shap, axis=0)
    total_impact = np.sum(mean_abs_shap)
    normalized_importance = (mean_abs_shap / total_impact) if total_impact > 0 else np.zeros_like(mean_abs_shap)

    # Combine into sorted feature importance list
    importance_items = []
    for idx, (feat_name, val, norm_val) in enumerate(zip(clean_feature_names, mean_abs_shap, normalized_importance)):
        importance_items.append({
            "feature": feat_name,
            "mean_abs_shap": float(round(val, 5)),
            "importance": float(round(norm_val * 100, 2)),
        })

    # Sort descending by importance
    importance_items.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for rank, item in enumerate(importance_items, start=1):
        item["rank"] = rank

    # Top 15 features for clean visualization
    top_items = importance_items[:15]

    # Generate summary plot base64
    summary_plot_base64 = None
    try:
        plt.figure(figsize=(8, 5))
        features_to_plot = [it["feature"] for it in reversed(top_items)]
        vals_to_plot = [it["mean_abs_shap"] for it in reversed(top_items)]

        plt.barh(features_to_plot, vals_to_plot, color="#3b82f6", alpha=0.85)
        plt.xlabel("Mean |SHAP value| (Average Impact on Model Output)")
        plt.title(f"SHAP Global Feature Importance — {algorithm_name}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        summary_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as plot_err:
        logger.warning("Failed to render SHAP plot image: %s", plot_err)

    if explainer_type.startswith("Kernel"):
        limitations = "Computed on a representative subsample using kernel approximation for computational efficiency."

    return {
        "explainer_type": explainer_type,
        "sample_size": len(X_eval),
        "feature_importance": top_items,
        "summary_plot_base64": summary_plot_base64,
        "limitations": limitations
    }
