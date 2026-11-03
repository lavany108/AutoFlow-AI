import time
import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC

from backend.automl.preprocessing import build_preprocessor
from backend.automl.evaluation import (
    evaluate_classification,
    get_primary_metric,
)

# Silence optuna verbose logging to keep terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

def instantiate_estimator_with_params(algo_name: str, task_type: str = "classification", params: Dict[str, Any] = None):
    """Creates a new classification estimator instance with specified parameters for fraud detection."""
    if params is None:
        params = {}
    if "Random Forest" in algo_name:
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight="balanced")
    elif "Extra Trees" in algo_name:
        return ExtraTreesClassifier(**params, random_state=42, n_jobs=-1, class_weight="balanced")
    elif "HistGradientBoosting" in algo_name:
        return HistGradientBoostingClassifier(**params, random_state=42)
    elif "Gradient Boosting" in algo_name:
        return GradientBoostingClassifier(**params, random_state=42)
    elif "Logistic" in algo_name:
        return LogisticRegression(**params, max_iter=1000, random_state=42, class_weight="balanced")
    elif "Support Vector" in algo_name:
        return SVC(**params, probability=True, random_state=42, class_weight="balanced")
    else:
        return RandomForestClassifier(**params, random_state=42, class_weight="balanced")


def sample_hyperparameters(algo_name: str, task_type: str = "classification", trial: optuna.Trial = None) -> Dict[str, Any]:
    """Sample hyperparameters for the given fraud classification algorithm within safe laptop bounds."""
    params = {}
    if "Random Forest" in algo_name or "Extra Trees" in algo_name:
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150, step=25)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 14)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 4)
    elif "HistGradientBoosting" in algo_name:
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 50, 150, step=25)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 15, 63)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 40)
    elif "Gradient Boosting" in algo_name:
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150, step=25)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
    elif "Logistic" in algo_name:
        params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
    elif "Support Vector" in algo_name:
        params["C"] = trial.suggest_float("C", 0.1, 10.0, log=True)
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    else:
        # Default fallback
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 100)
    return params


def tune_model(
    algo_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numerical_cols: List[str],
    categorical_cols: List[str],
    baseline_metrics: Dict[str, Any],
    n_trials: int = 10,
    timeout_seconds: int = 180,
    classes: Optional[List[Any]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Runs Optuna hyperparameter optimization.
    Returns:
    - best_params
    - best_pipeline (fitted with best params)
    - tuned_metrics
    - trial_history
    - improvement
    """
    start_time = time.time()
    _, baseline_primary = get_primary_metric(task_type, baseline_metrics)

    # Set up CV for classification
    min_class_cnt = y_train.value_counts().min()
    splits = min(3, min_class_cnt)
    if splits >= 2:
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
    scoring = "f1_weighted"

    trials_info: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(algo_name, "classification", trial)
        estimator = instantiate_estimator_with_params(algo_name, "classification", params)
        preprocessor = build_preprocessor(X_train, numerical_cols, categorical_cols)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", estimator)
        ])

        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = float(np.mean(scores))
            # Record trial
            trials_info.append({
                "trial_number": trial.number + 1,
                "value": float(round(mean_score, 4)),
                "params": params
            })
            if progress_callback:
                pct = 10.0 + ((trial.number + 1) / n_trials) * 80.0
                progress_callback(pct, f"Optuna trial {trial.number + 1}/{n_trials} complete (Score: {mean_score:.4f})")
            return mean_score
        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return -999.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    best_params = study.best_params
    best_cv_score = float(round(study.best_value, 4))

    # Fit final model with best params on full X_train
    best_estimator = instantiate_estimator_with_params(algo_name, "classification", best_params)
    final_preprocessor = build_preprocessor(X_train, numerical_cols, categorical_cols)
    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("estimator", best_estimator)
    ])
    final_pipeline.fit(X_train, y_train)

    # Evaluate on X_test
    y_pred = final_pipeline.predict(X_test)
    y_proba = None
    if hasattr(final_pipeline, "predict_proba"):
        try:
            y_proba = final_pipeline.predict_proba(X_test)
        except Exception:
            pass

    tuned_metrics = evaluate_classification(
        np.array(y_test), np.array(y_pred), y_proba=y_proba, classes=classes, is_fraud_task=True
    )

    tuned_metrics["cv_mean"] = best_cv_score
    tuned_metrics["cv_std"] = 0.0

    _, tuned_primary = get_primary_metric(task_type, tuned_metrics)
    improvement = float(round(tuned_primary - baseline_primary, 4))
    duration = float(round(time.time() - start_time, 2))

    return {
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "best_pipeline": final_pipeline,
        "tuned_metrics": tuned_metrics,
        "baseline_score": baseline_primary,
        "tuned_score": tuned_primary,
        "improvement": improvement,
        "trials_count": len(study.trials),
        "duration_seconds": duration,
        "trials": trials_info
    }
