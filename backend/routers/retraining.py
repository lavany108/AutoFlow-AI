import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database import get_db, SessionLocal
from backend.models_db import RetrainJobRecord, MLModelRecord
from backend.schemas import RetrainTriggerRequest
from backend.automl.retraining import execute_retraining_workflow

router = APIRouter(prefix="/retraining", tags=["Autonomous Retraining"])

def _run_retrain_bg(retrain_job_id: str, tolerance: float):
    db: Session = SessionLocal()
    try:
        execute_retraining_workflow(db, retrain_job_id, performance_tolerance=tolerance)
    finally:
        db.close()

@router.post("/trigger")
def trigger_retraining(
    req: RetrainTriggerRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Triggers an autonomous retraining job.
    Evaluates new candidate against active model and promotes safely if criteria are met.
    """
    if req.model_id:
        active_model = db.query(MLModelRecord).filter(MLModelRecord.id == req.model_id).first()
    else:
        active_model = db.query(MLModelRecord).filter(MLModelRecord.is_active == True).first()

    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found to retrain against.")

    job_id = uuid.uuid4().hex
    job = RetrainJobRecord(
        id=job_id,
        model_id=active_model.id,
        dataset_id=active_model.dataset_id,
        triggered_by=req.reason,
        status="PENDING",
        promoted=False
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(
        _run_retrain_bg,
        retrain_job_id=job.id,
        tolerance=req.performance_tolerance
    )

    return {
        "job_id": job.id,
        "status": job.status,
        "model_id": job.model_id,
        "triggered_by": job.triggered_by,
        "message": "Retraining job queued successfully."
    }


import math

def _sanitize_val(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_val(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_val(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj

def _serialize_job(job: RetrainJobRecord) -> dict:
    return {
        "id": job.id,
        "model_id": job.model_id,
        "dataset_id": job.dataset_id,
        "triggered_by": job.triggered_by,
        "status": job.status,
        "candidate_model_id": job.candidate_model_id,
        "old_model_id": job.old_model_id,
        "comparison_metrics": _sanitize_val(job.comparison_metrics) if job.comparison_metrics else None,
        "promoted": job.promoted,
        "decision_reason": job.decision_reason,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }

@router.get("/jobs/{job_id}")
def get_retrain_job(job_id: str, db: Session = Depends(get_db)):
    """Retrieves status and decision outcome of a retraining job."""
    job = db.query(RetrainJobRecord).filter(RetrainJobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Retrain job not found.")
    return _serialize_job(job)


@router.get("/jobs")
def list_retrain_jobs(db: Session = Depends(get_db)):
    """Lists history of all autonomous retraining jobs and audit records."""
    jobs = db.query(RetrainJobRecord).order_by(desc(RetrainJobRecord.created_at)).limit(50).all()
    return [_serialize_job(j) for j in jobs]

