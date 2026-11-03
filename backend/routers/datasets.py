from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import DatasetModel
from backend.schemas import (
    DatasetUploadResponse,
    DatasetDetailResponse,
    DatasetConfigureTargetRequest,
    DatasetPreviewResponse
)
from backend.services.dataset_service import (
    save_and_profile_dataset,
    configure_target,
    get_dataset_preview
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Uploads a CSV dataset, validates schema, profiles statistics, and stores safely."""
    dataset = await save_and_profile_dataset(file, db)
    return DatasetUploadResponse(
        id=dataset.id,
        filename=dataset.filename,
        original_filename=dataset.original_filename,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        file_size_bytes=dataset.file_size_bytes,
        created_at=dataset.created_at
    )

@router.get("", response_model=List[DatasetUploadResponse])
def list_datasets(db: Session = Depends(get_db)):
    """Lists all uploaded datasets."""
    datasets = db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()
    return [
        DatasetUploadResponse(
            id=d.id,
            filename=d.filename,
            original_filename=d.original_filename,
            row_count=d.row_count,
            column_count=d.column_count,
            file_size_bytes=d.file_size_bytes,
            created_at=d.created_at
        )
        for d in datasets
    ]

import os
import pandas as pd
from backend.automl.fraud_utils import compute_imbalance_stats

def _build_dataset_detail(dataset: DatasetModel) -> DatasetDetailResponse:
    imbalance = None
    file_path = getattr(dataset, "resolved_file_path", dataset.file_path)
    if dataset.target_column and dataset.task_type == "classification" and file_path:
        try:
            df_target = pd.read_csv(file_path, usecols=[dataset.target_column])
            imbalance = compute_imbalance_stats(df_target[dataset.target_column].dropna())
        except Exception:
            pass

    return DatasetDetailResponse(
        id=dataset.id,
        filename=dataset.filename,
        original_filename=dataset.original_filename,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        file_size_bytes=dataset.file_size_bytes,
        columns_info=dataset.columns_info,
        summary_stats=dataset.summary_stats,
        target_column=dataset.target_column,
        task_type=dataset.task_type,
        potential_targets=list(dataset.columns_info.keys()),
        imbalance_stats=imbalance,
        created_at=dataset.created_at
    )

@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieves detailed profile, schema, and statistics for a dataset."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return _build_dataset_detail(dataset)

@router.post("/{dataset_id}/target", response_model=DatasetDetailResponse)
def set_target_and_task(
    dataset_id: str,
    req: DatasetConfigureTargetRequest,
    db: Session = Depends(get_db)
):
    """Configures the target column and task type (with optional user override)."""
    dataset = configure_target(dataset_id, req.target_column, req.task_type, db)
    return _build_dataset_detail(dataset)

@router.post("/{dataset_id}/configure-target", response_model=DatasetDetailResponse)
def configure_target_alias(
    dataset_id: str,
    req: DatasetConfigureTargetRequest,
    db: Session = Depends(get_db)
):
    """Alias for /target endpoint."""
    return set_target_and_task(dataset_id, req, db)

@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def get_preview(
    dataset_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Returns top rows of dataset for table preview in the frontend."""
    return get_dataset_preview(dataset_id, limit, db)

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Deletes a dataset and its underlying file."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    target_path = getattr(dataset, "resolved_file_path", dataset.file_path)
    if target_path and os.path.exists(target_path):
        try:
            os.remove(target_path)
        except Exception:
            pass

    db.delete(dataset)
    db.commit()
    return {"message": "Dataset deleted successfully."}
