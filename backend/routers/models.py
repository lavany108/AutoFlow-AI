from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import MLModelRecord
from backend.schemas import MLModelSummary, MLModelDetail

router = APIRouter(prefix="/models", tags=["Model Registry"])

@router.get("", response_model=List[MLModelSummary])
def list_models(
    dataset_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Lists all registered models in the model registry."""
    query = db.query(MLModelRecord)
    if dataset_id:
        query = query.filter(MLModelRecord.dataset_id == dataset_id)

    models = query.order_by(MLModelRecord.created_at.desc()).all()
    return [
        MLModelSummary(
            id=m.id,
            name=m.name,
            version=m.version,
            dataset_id=m.dataset_id,
            algorithm=m.algorithm,
            task_type=m.task_type,
            target_column=m.target_column,
            metrics=m.metrics,
            is_active=m.is_active,
            is_tuned=m.is_tuned,
            mlflow_run_id=m.mlflow_run_id,
            created_at=m.created_at
        )
        for m in models
    ]


@router.get("/{model_id}", response_model=MLModelDetail)
def get_model(model_id: str, db: Session = Depends(get_db)):
    """Retrieves full metadata, hyperparameters, and feature schema of a model."""
    model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    return MLModelDetail(
        id=model.id,
        name=model.name,
        version=model.version,
        dataset_id=model.dataset_id,
        algorithm=model.algorithm,
        task_type=model.task_type,
        target_column=model.target_column,
        metrics=model.metrics,
        is_active=model.is_active,
        is_tuned=model.is_tuned,
        mlflow_run_id=model.mlflow_run_id,
        created_at=model.created_at,
        feature_names=model.feature_names,
        target_classes=model.target_classes,
        hyperparameters=model.hyperparameters,
        artifact_path=model.artifact_path,
        training_duration_seconds=model.training_duration_seconds
    )


@router.post("/{model_id}/activate", response_model=MLModelSummary)
def set_active_model(model_id: str, db: Session = Depends(get_db)):
    """Sets the specified model as the active model for serving predictions."""
    target_model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    if not target_model:
        raise HTTPException(status_code=404, detail="Model not found.")

    # Deactivate all models for this dataset
    db.query(MLModelRecord).filter(
        MLModelRecord.dataset_id == target_model.dataset_id
    ).update({"is_active": False})

    target_model.is_active = True
    db.commit()
    db.refresh(target_model)

    return MLModelSummary(
        id=target_model.id,
        name=target_model.name,
        version=target_model.version,
        dataset_id=target_model.dataset_id,
        algorithm=target_model.algorithm,
        task_type=target_model.task_type,
        target_column=target_model.target_column,
        metrics=target_model.metrics,
        is_active=target_model.is_active,
        is_tuned=target_model.is_tuned,
        mlflow_run_id=target_model.mlflow_run_id,
        created_at=target_model.created_at
    )


@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Deletes a model and associated artifact files."""
    import os
    model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    if model.artifact_path and os.path.exists(model.artifact_path):
        try:
            os.remove(model.artifact_path)
        except Exception:
            pass
    if model.reference_distribution_path and os.path.exists(model.reference_distribution_path):
        try:
            os.remove(model.reference_distribution_path)
        except Exception:
            pass

    db.delete(model)
    db.commit()
    return {"message": f"Model {model_id} deleted successfully."}
