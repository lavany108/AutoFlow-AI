import uuid
from typing import List
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import TrainingJobRecord, DatasetModel, MLModelRecord
from backend.schemas import (
    StartBenchmarkRequest,
    StartTuningRequest,
    JobStatusResponse,
    LeaderboardItem
)
from backend.services.training_service import (
    run_benchmark_background,
    run_tuning_background
)

router = APIRouter(prefix="/training", tags=["AutoML Training"])

@router.post("/benchmark", response_model=JobStatusResponse)
def start_benchmark(
    req: StartBenchmarkRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Initiates asynchronous benchmarking across 5 candidate models.
    Returns immediately with a job_id for status polling.
    """
    dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not dataset.target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column has not been configured. Please configure target column first."
        )

    job_id = uuid.uuid4().hex
    job = TrainingJobRecord(
        id=job_id,
        dataset_id=dataset.id,
        job_type="BENCHMARK",
        status="PENDING",
        progress=0.0,
        current_step="Queued for benchmarking...",
        leaderboard=[]
    )
    db.add(job)
    db.commit()

    # Launch in background
    background_tasks.add_task(
        run_benchmark_background,
        job_id=job_id,
        dataset_id=dataset.id,
        test_size=req.test_size,
        cv_splits=req.cv_splits,
        random_state=req.random_state,
        use_smote=req.use_smote
    )

    return JobStatusResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        leaderboard=[],
        best_model_id=None,
        error_message=None,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_training_jobs(db: Session = Depends(get_db)):
    """Lists recent training and tuning jobs."""
    jobs = db.query(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc()).limit(20).all()
    return [
        JobStatusResponse(
            job_id=j.id,
            dataset_id=j.dataset_id,
            job_type=j.job_type,
            status=j.status,
            progress=j.progress,
            current_step=j.current_step,
            leaderboard=[LeaderboardItem(**item) for item in (j.leaderboard or [])],
            best_model_id=j.best_model_id,
            error_message=j.error_message,
            created_at=j.created_at,
            updated_at=j.updated_at
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Polls status, progress, and leaderboard of a training or tuning job."""
    job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")

    leaderboard_items = [
        LeaderboardItem(**item) for item in (job.leaderboard or [])
    ]

    return JobStatusResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        leaderboard=leaderboard_items,
        best_model_id=job.best_model_id,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@router.post("/tune", response_model=JobStatusResponse)
def start_tuning(
    req: StartTuningRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Launches Optuna hyperparameter tuning for a selected model.
    """
    model = db.query(MLModelRecord).filter(MLModelRecord.id == req.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    job_id = uuid.uuid4().hex
    job = TrainingJobRecord(
        id=job_id,
        dataset_id=model.dataset_id,
        job_type="TUNE",
        status="PENDING",
        progress=0.0,
        current_step="Queued for Optuna tuning...",
        leaderboard=[]
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(
        run_tuning_background,
        job_id=job_id,
        original_model_id=model.id,
        n_trials=req.n_trials,
        timeout_seconds=req.timeout_seconds or 180
    )

    return JobStatusResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        leaderboard=[],
        best_model_id=None,
        error_message=None,
        created_at=job.created_at,
        updated_at=job.updated_at
    )
