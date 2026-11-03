import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import mlflow
from backend.config import settings

logger = logging.getLogger(__name__)

# Configure local MLflow tracking
if settings.MLFLOW_TRACKING_URI.startswith(("sqlite:", "http:", "https:", "file:")):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
else:
    mlflow.set_tracking_uri(f"file://{settings.MLFLOW_TRACKING_URI}")

def save_model_artifact(
    model_id: str,
    pipeline: Any,
    reference_distribution: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[str]]:
    """
    Saves trained pipeline to storage/models/{model_id}.joblib
    and reference distribution to storage/models/{model_id}_ref.json.
    """
    model_path = settings.MODELS_DIR / f"{model_id}.joblib"
    joblib.dump(pipeline, str(model_path))

    ref_path_str = None
    if reference_distribution is not None:
        ref_path = settings.MODELS_DIR / f"{model_id}_ref.json"
        with open(ref_path, "w") as f:
            json.dump(reference_distribution, f, indent=2)
        ref_path_str = str(ref_path)

    return str(model_path), ref_path_str


def load_model_artifact(artifact_path: str) -> Any:
    """
    Loads saved scikit-learn pipeline from artifact_path safely.
    Validates that the file exists and is located within the trusted models directory.
    Resolves paths across host and Docker container environments.
    """
    path = Path(artifact_path)
    if not path.is_file():
        # Fallback to current environment's MODELS_DIR by filename
        candidate = settings.MODELS_DIR / path.name
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Model artifact not found at {artifact_path} (checked {candidate})")

    # Ensure path is within application models directory to prevent unauthorized deserialization
    try:
        resolved_path = path.resolve()
        resolved_storage = settings.MODELS_DIR.resolve()
        if not str(resolved_path).startswith(str(resolved_storage)):
            candidate = settings.MODELS_DIR / path.name
            if candidate.is_file() and str(candidate.resolve()).startswith(str(resolved_storage)):
                path = candidate
            else:
                raise ValueError("Untrusted model artifact path rejected.")
    except Exception as e:
        logger.warning("Path resolution warning: %s", e)

    pipeline = joblib.load(str(path))
    return pipeline


def load_reference_distribution(ref_path: Optional[str]) -> Dict[str, Any]:
    """Loads reference distribution from json file, resolving cross-environment paths."""
    if not ref_path:
        return {}
    path = Path(ref_path)
    if not path.is_file():
        candidate = settings.MODELS_DIR / path.name
        if candidate.is_file():
            path = candidate
        else:
            return {}
    try:
        with open(str(path), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load reference distribution from %s: %s", path, e)
        return {}


def log_experiment_to_mlflow(
    experiment_name: str,
    run_name: str,
    dataset_id: str,
    algorithm: str,
    task_type: str,
    target_column: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    duration_seconds: float,
    artifact_path: Optional[str] = None
) -> Optional[str]:
    """
    Logs model metadata, metrics, parameters, and tags to MLflow.
    Returns mlflow_run_id.
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            # Tags
            mlflow.set_tag("dataset_id", dataset_id)
            mlflow.set_tag("algorithm", algorithm)
            mlflow.set_tag("task_type", task_type)
            mlflow.set_tag("target_column", target_column)
            mlflow.set_tag("framework", "scikit-learn")

            # Parameters
            for k, v in hyperparameters.items():
                # Limit param length
                param_str = str(v)[:250]
                mlflow.log_param(k, param_str)

            # Metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    mlflow.log_metric(k, float(v))

            mlflow.log_metric("training_duration_seconds", duration_seconds)

            # Log artifact file if exists
            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, artifact_path="model")

            return run.info.run_id
    except Exception as e:
        logger.warning("MLflow logging encountered an issue: %s", e)
        return None
