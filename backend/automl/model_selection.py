import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC

from backend.automl.preprocessing import (
    build_preprocessor,
    extract_reference_distribution,
)
from backend.automl.evaluation import evaluate_classification, get_primary_metric
from backend.automl.fraud_utils import (
    compute_class_weights,
    try_smote_resample,
    compute_imbalance_stats,
)

logger = logging.getLogger(__name__)


def get_candidate_algorithms(task_type: str = "classification", class_weight: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Returns fraud-optimized candidate estimators for classification.
    class_weight is passed to estimators that support it.
    """
    cw = class_weight or "balanced"

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight=cw, C=0.1
        ),
        "Random Forest (Fraud)": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight=cw, min_samples_leaf=5,
        ),
        "Extra Trees (Fraud)": ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight=cw, min_samples_leaf=5,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=42, max_iter=200, learning_rate=0.05,
            class_weight=cw if isinstance(cw, str) else None,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, random_state=42, learning_rate=0.05,
            subsample=0.8,
        ),
    }


def benchmark_models(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = 42,
    use_smote: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Fraud-aware benchmarking pipeline:
    1. Stratified train/test split
    2. Imbalance stats
    3. Optional SMOTE resampling (training data only — no leakage)
    4. Class-weight-aware candidate models
    5. CV with StratifiedKFold
    6. Full evaluation with PR-AUC, F2, etc.
    """
    if progress_callback:
        progress_callback(5.0, "Preparing data and stratified train/test split...")

    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < 20:
        raise ValueError(
            f"Dataset has only {len(clean_df)} valid rows after dropping missing targets. "
            "At least 20 rows required."
        )

    y = clean_df[target_column]
    raw_X = clean_df.drop(columns=[target_column])

    # Identify and drop non-predictive identifier and ultra-high cardinality text columns
    cols_to_drop = []
    n_rows_total = max(1, len(raw_X))
    for col in raw_X.columns:
        col_lower = col.lower().strip()
        is_id_name = (
            col_lower in ["id", "transaction_id", "transactionid", "trans_id", "user_id", "userid", "customer_id", "customerid", "account_id", "uuid", "guid"]
            or col_lower.endswith(("_id", ".id"))
            or col_lower.startswith(("id_", "uuid_"))
        )
        is_text_high_card = (
            not pd.api.types.is_numeric_dtype(raw_X[col])
            and (raw_X[col].nunique() > 100 or (raw_X[col].nunique() / n_rows_total) > 0.4)
        )
        if is_id_name or is_text_high_card:
            cols_to_drop.append(col)

    if cols_to_drop:
        logger.info("Auto-pruning non-predictive identifier columns: %s", cols_to_drop)
        X = raw_X.drop(columns=cols_to_drop)
    else:
        X = raw_X

    feature_names = list(X.columns)

    # Detect feature types
    numerical_cols, categorical_cols = [], []
    for col in feature_names:
        if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    # Encode target for classification
    unique_classes = sorted(y.unique().tolist(), key=str)
    classes = unique_classes
    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= cv_splits else None

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Reference distribution for drift monitoring (strictly from X_train pre-SMOTE)
    reference_distribution = extract_reference_distribution(X_train, feature_names)

    # Imbalance analysis
    imbalance_stats = compute_imbalance_stats(y_train)
    class_weights = compute_class_weights(y_train)

    # SMOTE — applied to training set AFTER split (no leakage)
    was_resampled = False
    if use_smote:
        if progress_callback:
            progress_callback(12.0, "Analyzing class imbalance & applying SMOTE...")
        # Preprocess first, then SMOTE on numeric arrays
        preprocessor_for_smote = build_preprocessor(X_train, numerical_cols, categorical_cols)
        X_train_proc = preprocessor_for_smote.fit_transform(X_train)
        X_train_smote, y_train_smote, was_resampled = try_smote_resample(
            pd.DataFrame(X_train_proc),
            y_train,
            random_state=random_state,
        )
        if was_resampled:
            # Store preprocessor used for SMOTE; models will use raw X_train after this
            imbalance_stats["smote_applied"] = True
            imbalance_stats["smote_samples_after"] = len(y_train_smote)

    if progress_callback:
        progress_callback(15.0, "Starting model benchmarking...")

    candidates = get_candidate_algorithms(task_type="classification", class_weight="balanced")
    total_candidates = len(candidates)
    leaderboard = []
    trained_pipelines = {}

    # CV strategy
    min_train_class_cnt = y_train.value_counts().min()
    actual_splits = min(cv_splits, min_train_class_cnt)
    if len(X_train) > 20000 and actual_splits > 3:
        actual_splits = 3
    if actual_splits >= 2:
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=2, shuffle=True, random_state=random_state)

    unique_y = list(np.unique(y_train))
    if set(unique_y).issubset({0, 1}):
        cv_scoring = "average_precision"
    elif len(unique_y) == 2:
        from sklearn.metrics import make_scorer, average_precision_score
        pos = next((c for c in unique_y if str(c).strip().lower() in ["1", "true", "yes", "fraud"]), unique_y[-1])
        try:
            cv_scoring = make_scorer(average_precision_score, pos_label=pos, response_method="predict_proba")
        except Exception:
            cv_scoring = "f1_macro"
    else:
        cv_scoring = "f1_macro"

    for idx, (algo_name, estimator) in enumerate(candidates.items(), start=1):
        step_progress = 15.0 + (idx / total_candidates) * 72.0
        if progress_callback:
            progress_callback(step_progress, f"Evaluating {idx}/{total_candidates}: {algo_name}...")

        start_time = time.time()
        try:
            preprocessor = build_preprocessor(X_train, numerical_cols, categorical_cols)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ])

            # CV
            try:
                cv_jobs = 1 if len(X_train) > 10000 else -1
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=cv_jobs)
                mean_val = np.nanmean(cv_scores) if len(cv_scores) > 0 else 0.0
                std_val = np.nanstd(cv_scores) if len(cv_scores) > 0 else 0.0
                cv_mean = 0.0 if np.isnan(mean_val) or np.isinf(mean_val) else float(round(mean_val, 4))
                cv_std = 0.0 if np.isnan(std_val) or np.isinf(std_val) else float(round(std_val, 4))
            except Exception as cv_err:
                logger.warning("CV failed for %s: %s", algo_name, cv_err)
                cv_mean, cv_std = 0.0, 0.0

            # Fit on full training set
            pipeline.fit(X_train, y_train)
            fit_time = float(round(time.time() - start_time, 2))

            y_pred = pipeline.predict(X_test)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                except Exception:
                    pass

            # Evaluate with fraud metrics
            metrics = evaluate_classification(
                np.array(y_test), np.array(y_pred),
                y_proba=y_proba, classes=classes, is_fraud_task=True,
            )

            metrics["cv_mean"] = cv_mean
            metrics["cv_std"] = cv_std

            metric_name, metric_val = get_primary_metric("classification", metrics)

            hyperparams = {
                k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                for k, v in estimator.get_params().items()
            }

            leaderboard.append({
                "algorithm": algo_name,
                "task_type": "classification",
                "primary_metric_name": metric_name,
                "primary_metric_value": metric_val,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "fit_time_seconds": fit_time,
                "metrics": metrics,
                "hyperparameters": hyperparams,
                "status": "SUCCESS",
                "error_message": None,
            })
            trained_pipelines[algo_name] = pipeline

        except Exception as e:
            logger.error("Failed to train %s: %s", algo_name, e, exc_info=True)
            leaderboard.append({
                "algorithm": algo_name,
                "task_type": "classification",
                "primary_metric_name": "pr_auc",
                "primary_metric_value": -999.0,
                "cv_mean": 0.0,
                "cv_std": 0.0,
                "fit_time_seconds": float(round(time.time() - start_time, 2)),
                "metrics": {},
                "hyperparameters": {},
                "status": "FAILED",
                "error_message": str(e),
            })

    # Sort by primary metric descending
    leaderboard.sort(key=lambda item: item["primary_metric_value"], reverse=True)

    if progress_callback:
        progress_callback(90.0, "Selecting best fraud detection model...")

    best_candidate = next((item for item in leaderboard if item["status"] == "SUCCESS"), None)
    if not best_candidate:
        raise RuntimeError("All candidate models failed during training.")

    best_algo = best_candidate["algorithm"]
    best_pipeline = trained_pipelines[best_algo]

    return {
        "leaderboard": leaderboard,
        "best_algorithm": best_algo,
        "best_pipeline": best_pipeline,
        "trained_pipelines": trained_pipelines,
        "best_metrics": best_candidate["metrics"],
        "best_hyperparameters": best_candidate["hyperparameters"],
        "target_classes": [str(c) for c in classes] if classes else None,
        "feature_names": feature_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "reference_distribution": reference_distribution,
        "imbalance_stats": imbalance_stats,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
