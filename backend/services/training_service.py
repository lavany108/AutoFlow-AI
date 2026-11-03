import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models_db import DatasetModel, MLModelRecord, TrainingJobRecord, MonitoringEventRecord
from backend.automl.model_selection import benchmark_models
from backend.automl.tuning import tune_model
from backend.automl.model_registry import save_model_artifact, log_experiment_to_mlflow
from backend.automl.evaluation import get_primary_metric

logger = logging.getLogger(__name__)

def run_benchmark_background(
    job_id: str,
    dataset_id: str,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = 42,
    use_smote: bool = True
):
    """
    Background worker function for model benchmarking.
    Runs in separate thread/task with independent DB session.
    """
    db: Session = SessionLocal()
    start_total_time = time.time()
    try:
        job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
        if not job:
            return

        job.status = "RUNNING"
        job.progress = 5.0
        job.current_step = "Loading dataset..."
        db.commit()

        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset or not dataset.target_column:
            job.status = "FAILED"
            job.error_message = "Dataset or target column not configured."
            db.commit()
            return

        df = pd.read_csv(getattr(dataset, "resolved_file_path", dataset.file_path))

        def update_progress(pct: float, step_msg: str):
            try:
                job.progress = pct
                job.current_step = step_msg
                db.commit()
            except Exception:
                pass

        # Execute benchmarking across candidate algorithms
        benchmark_results = benchmark_models(
            df=df,
            target_column=dataset.target_column,
            task_type=dataset.task_type or "classification",
            test_size=test_size,
            cv_splits=cv_splits,
            random_state=random_state,
            use_smote=use_smote,
            progress_callback=update_progress
        )

        update_progress(92.0, "Persisting models and logging experiments...")

        leaderboard = benchmark_results["leaderboard"]
        best_algo = benchmark_results["best_algorithm"]
        best_pipeline = benchmark_results["best_pipeline"]
        trained_pipelines = benchmark_results.get("trained_pipelines", {})
        ref_dist = benchmark_results["reference_distribution"]
        feature_names = benchmark_results["feature_names"]
        target_classes = benchmark_results.get("target_classes")
        task_type = dataset.task_type or "classification"

        # Check existing models for version numbering
        existing_models_count = db.query(MLModelRecord).filter(
            MLModelRecord.dataset_id == dataset.id
        ).count()

        # Demote existing active models for this dataset if any
        db.query(MLModelRecord).filter(
            MLModelRecord.dataset_id == dataset.id,
            MLModelRecord.is_active == True
        ).update({"is_active": False})

        # Save all successful models and assign IDs
        best_model_db_id = None
        for item in leaderboard:
            if item["status"] != "SUCCESS":
                continue

            algo = item["algorithm"]
            model_db_id = uuid.uuid4().hex
            is_best = (algo == best_algo)
            
            # Save actual trained pipeline for each model
            pipeline_to_save = trained_pipelines.get(algo, best_pipeline if is_best else None)
            if pipeline_to_save is None:
                pipeline_to_save = best_pipeline

            if is_best:
                best_model_db_id = model_db_id

            artifact_path, ref_path = save_model_artifact(
                model_db_id, pipeline_to_save, ref_dist if is_best else None
            )

            # Log to MLflow
            run_id = log_experiment_to_mlflow(
                experiment_name=f"AutoFlow_{dataset.original_filename}",
                run_name=f"{algo}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                dataset_id=dataset.id,
                algorithm=algo,
                task_type=task_type,
                target_column=dataset.target_column,
                hyperparameters=item["hyperparameters"],
                metrics=item["metrics"],
                duration_seconds=item["fit_time_seconds"],
                artifact_path=artifact_path if is_best else None
            )

            existing_models_count += 1
            model_record = MLModelRecord(
                id=model_db_id,
                name=f"{algo} (v{existing_models_count})",
                version=existing_models_count,
                dataset_id=dataset.id,
                algorithm=algo,
                task_type=task_type,
                target_column=dataset.target_column,
                feature_names=feature_names,
                target_classes=target_classes,
                hyperparameters=item["hyperparameters"],
                metrics=item["metrics"],
                artifact_path=artifact_path,
                reference_distribution_path=ref_path if is_best else None,
                mlflow_run_id=run_id,
                is_active=is_best,
                is_tuned=False,
                training_duration_seconds=item["fit_time_seconds"]
            )
            db.add(model_record)
            item["model_id"] = model_db_id

        # Update job record
        job.status = "COMPLETED"
        job.progress = 100.0
        job.current_step = f"Benchmarking completed. Best algorithm: {best_algo}"
        job.leaderboard = leaderboard
        job.best_model_id = best_model_db_id
        job.result_summary = {
            "best_algorithm": best_algo,
            "models_evaluated": len(leaderboard),
            "total_time_seconds": float(round(time.time() - start_total_time, 2))
        }
        db.commit()

    except Exception as e:
        logger.error("Benchmark background job failed: %s", e, exc_info=True)
        job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.progress = 100.0
            job.current_step = "Failed"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


def run_tuning_background(
    job_id: str,
    original_model_id: str,
    n_trials: int = 10,
    timeout_seconds: int = 180
):
    """
    Background worker for Optuna hyperparameter optimization.
    """
    db: Session = SessionLocal()
    try:
        job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
        if not job:
            return

        job.status = "RUNNING"
        job.progress = 5.0
        job.current_step = "Preparing dataset for Optuna tuning..."
        db.commit()

        model_rec = db.query(MLModelRecord).filter(MLModelRecord.id == original_model_id).first()
        if not model_rec:
            job.status = "FAILED"
            job.error_message = "Target model not found."
            db.commit()
            return

        dataset = db.query(DatasetModel).filter(DatasetModel.id == model_rec.dataset_id).first()
        if not dataset:
            job.status = "FAILED"
            job.error_message = "Associated dataset not found."
            db.commit()
            return

        df = pd.read_csv(getattr(dataset, "resolved_file_path", dataset.file_path)).dropna(subset=[model_rec.target_column])
        y = df[model_rec.target_column]
        X = df.drop(columns=[model_rec.target_column])

        # Feature groupings
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 10]
        cat_cols = [c for c in X.columns if c not in num_cols]

        from sklearn.model_selection import train_test_split
        stratify = y if model_rec.task_type == "classification" and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        def update_progress(pct: float, msg: str):
            try:
                job.progress = pct
                job.current_step = msg
                db.commit()
            except Exception:
                pass

        tune_res = tune_model(
            algo_name=model_rec.algorithm,
            task_type=model_rec.task_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            baseline_metrics=model_rec.metrics,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            classes=model_rec.target_classes,
            progress_callback=update_progress
        )

        # Save tuned model
        tuned_model_id = uuid.uuid4().hex
        artifact_path, ref_path = save_model_artifact(
            tuned_model_id,
            tune_res["best_pipeline"],
            None
        )

        # Count versions
        latest_ver = db.query(MLModelRecord).filter(
            MLModelRecord.dataset_id == dataset.id
        ).count() + 1

        # MLflow logging
        mlflow_run_id = log_experiment_to_mlflow(
            experiment_name=f"AutoFlow_{dataset.original_filename}_Tuning",
            run_name=f"Tuned_{model_rec.algorithm}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            dataset_id=dataset.id,
            algorithm=f"{model_rec.algorithm} (Optuna Tuned)",
            task_type=model_rec.task_type,
            target_column=model_rec.target_column,
            hyperparameters=tune_res["best_params"],
            metrics=tune_res["tuned_metrics"],
            duration_seconds=tune_res["duration_seconds"],
            artifact_path=artifact_path
        )

        # Decide whether to activate
        is_improved = tune_res["improvement"] >= 0
        if is_improved:
            # Demote older active model
            db.query(MLModelRecord).filter(
                MLModelRecord.dataset_id == dataset.id,
                MLModelRecord.is_active == True
            ).update({"is_active": False})

        tuned_record = MLModelRecord(
            id=tuned_model_id,
            name=f"{model_rec.algorithm} (Tuned v{latest_ver})",
            version=latest_ver,
            dataset_id=dataset.id,
            algorithm=model_rec.algorithm,
            task_type=model_rec.task_type,
            target_column=model_rec.target_column,
            feature_names=model_rec.feature_names,
            target_classes=model_rec.target_classes,
            hyperparameters=tune_res["best_params"],
            metrics=tune_res["tuned_metrics"],
            artifact_path=artifact_path,
            reference_distribution_path=model_rec.reference_distribution_path,
            mlflow_run_id=mlflow_run_id,
            is_active=is_improved,
            is_tuned=True,
            training_duration_seconds=tune_res["duration_seconds"]
        )
        db.add(tuned_record)

        job.status = "COMPLETED"
        job.progress = 100.0
        job.current_step = f"Tuning finished. Score: {tune_res['tuned_score']:.4f} (Improvement: {tune_res['improvement']:+.4f})"
        job.best_model_id = tuned_model_id
        job.result_summary = {
            "original_model_id": original_model_id,
            "tuned_model_id": tuned_model_id,
            "best_params": tune_res["best_params"],
            "baseline_score": tune_res["baseline_score"],
            "tuned_score": tune_res["tuned_score"],
            "improvement": tune_res["improvement"],
            "trials_count": tune_res["trials_count"],
            "duration_seconds": tune_res["duration_seconds"],
            "trials": tune_res["trials"]
        }
        db.commit()

    except Exception as e:
        logger.error("Tuning background job failed: %s", e, exc_info=True)
        job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.progress = 100.0
            job.current_step = "Tuning failed"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()
