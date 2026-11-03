import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
from sqlalchemy.orm import Session

from backend.models_db import MLModelRecord, RetrainJobRecord, DatasetModel, MonitoringEventRecord
from backend.automl.model_selection import benchmark_models
from backend.automl.model_registry import save_model_artifact, log_experiment_to_mlflow
from backend.automl.evaluation import get_primary_metric

logger = logging.getLogger(__name__)

def execute_retraining_workflow(
    db: Session,
    retrain_job_id: str,
    performance_tolerance: float = 0.05
) -> RetrainJobRecord:
    """
    Executes an autonomous retraining pipeline:
    1. Loads dataset and active model
    2. Runs model training on latest data
    3. Evaluates candidate model vs active model
    4. Validates and conditionally promotes candidate
    5. Updates retrain job audit record
    """
    job = db.query(RetrainJobRecord).filter(RetrainJobRecord.id == retrain_job_id).first()
    if not job:
        raise ValueError(f"RetrainJob {retrain_job_id} not found.")

    job.status = "RUNNING"
    job.updated_at = datetime.utcnow()
    db.commit()

    active_model = db.query(MLModelRecord).filter(MLModelRecord.id == job.model_id).first()
    if not active_model:
        job.status = "FAILED"
        job.decision_reason = "Active model referenced in retrain job does not exist."
        db.commit()
        return job

    dataset = db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first()
    if not dataset or not dataset.file_path:
        job.status = "FAILED"
        job.decision_reason = "Dataset file not found for retraining."
        db.commit()
        return job

    try:
        # Load dataset
        df = pd.read_csv(getattr(dataset, "resolved_file_path", dataset.file_path))
        target_col = active_model.target_column
        task_type = active_model.task_type

        # Benchmark/train new candidates
        benchmark_res = benchmark_models(
            df=df,
            target_column=target_col,
            task_type=task_type,
            test_size=0.2,
            cv_splits=5,
            random_state=int(datetime.utcnow().timestamp()) % 100000
        )

        candidate_algo = benchmark_res["best_algorithm"]
        candidate_pipeline = benchmark_res["best_pipeline"]
        candidate_metrics = benchmark_res["best_metrics"]
        candidate_hyperparams = benchmark_res["best_hyperparameters"]

        # Primary metrics comparison
        _, active_score = get_primary_metric(task_type, active_model.metrics)
        primary_name, candidate_score = get_primary_metric(task_type, candidate_metrics)

        # Decision threshold: candidate must achieve at least (active_score - tolerance)
        required_score = active_score - performance_tolerance
        should_promote = candidate_score >= required_score

        # Save candidate model artifact
        new_model_id = uuid.uuid4().hex
        artifact_path, ref_path = save_model_artifact(
            new_model_id,
            candidate_pipeline,
            benchmark_res.get("reference_distribution")
        )

        # MLflow logging
        mlflow_run_id = log_experiment_to_mlflow(
            experiment_name="AutoFlow_Retraining",
            run_name=f"Retrain_{candidate_algo[:15]}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            dataset_id=dataset.id,
            algorithm=candidate_algo,
            task_type=task_type,
            target_column=target_col,
            hyperparameters=candidate_hyperparams,
            metrics=candidate_metrics,
            duration_seconds=10.0,
            artifact_path=artifact_path
        )

        # Find latest version number
        latest_version = db.query(MLModelRecord).filter(
            MLModelRecord.dataset_id == dataset.id
        ).count() + 1

        # Register candidate model in DB
        candidate_model = MLModelRecord(
            id=new_model_id,
            name=f"{candidate_algo} (Retrained v{latest_version})",
            version=latest_version,
            dataset_id=dataset.id,
            algorithm=candidate_algo,
            task_type=task_type,
            target_column=target_col,
            feature_names=benchmark_res["feature_names"],
            target_classes=benchmark_res.get("target_classes"),
            hyperparameters=candidate_hyperparams,
            metrics=candidate_metrics,
            artifact_path=artifact_path,
            reference_distribution_path=ref_path,
            mlflow_run_id=mlflow_run_id,
            is_active=False,
            is_tuned=False,
            training_duration_seconds=10.0
        )
        db.add(candidate_model)
        db.commit()

        job.candidate_model_id = candidate_model.id
        job.old_model_id = active_model.id
        job.comparison_metrics = {
            "primary_metric": primary_name,
            "active_model_score": active_score,
            "candidate_model_score": candidate_score,
            "threshold_required": required_score,
            "candidate_metrics": candidate_metrics,
            "active_metrics": active_model.metrics
        }

        if should_promote:
            # Demote active model
            active_model.is_active = False
            # Promote candidate model
            candidate_model.is_active = True
            job.promoted = True
            job.decision_reason = (
                f"Candidate model ({candidate_algo}) achieved {primary_name}={candidate_score:.4f}, "
                f"which meets the required threshold ({required_score:.4f}). "
                f"Promoted to active model (v{latest_version}). Previous model retained for rollback."
            )
            # Create monitoring audit event
            event = MonitoringEventRecord(
                model_id=candidate_model.id,
                event_type="RETRAIN_TRIGGERED",
                severity="INFO",
                details={
                    "retrain_job_id": job.id,
                    "promoted": True,
                    "reason": job.decision_reason
                }
            )
            db.add(event)
        else:
            job.promoted = False
            job.decision_reason = (
                f"Candidate model ({candidate_algo}) scored {primary_name}={candidate_score:.4f}, "
                f"falling below the minimum promotion threshold ({required_score:.4f}). "
                f"Active model (v{active_model.version}) was preserved."
            )

        job.status = "COMPLETED"
        job.updated_at = datetime.utcnow()
        db.commit()
        return job

    except Exception as e:
        logger.error("Retraining workflow failed: %s", e, exc_info=True)
        job.status = "FAILED"
        job.decision_reason = f"Pipeline execution failed: {str(e)}"
        job.updated_at = datetime.utcnow()
        db.commit()
        return job
