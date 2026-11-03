import os
import uuid
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models_db import DatasetModel
from backend.automl.preprocessing import profile_dataset, detect_task_type

logger = logging.getLogger(__name__)

async def save_and_profile_dataset(file: UploadFile, db: Session) -> DatasetModel:
    """
    Validates CSV upload, writes safely to storage, profiles features and schema,
    and registers dataset in the database.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are accepted.")

    # Read contents into memory safely
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size ({settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB)."
        )

    # Decode content
    df = None
    for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            from io import BytesIO
            df = pd.read_csv(BytesIO(content), encoding=encoding)
            break
        except Exception:
            continue

    if df is None:
        raise HTTPException(status_code=400, detail="Could not parse file as CSV. Please check delimiters and encoding.")

    if df.empty or df.shape[0] < 5:
        raise HTTPException(status_code=400, detail=f"Dataset must contain at least 5 rows (found {df.shape[0]}).")

    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail=f"Dataset must contain at least 2 columns (found {df.shape[1]}).")

    dataset_id = uuid.uuid4().hex
    safe_filename = f"{dataset_id}.csv"
    dest_path = settings.DATASETS_DIR / safe_filename

    # Save to disk
    with open(dest_path, "wb") as f:
        f.write(content)

    # Profile dataset
    try:
        columns_info, summary_stats, potential_targets, detected_target, detected_task = profile_dataset(df)
    except Exception as e:
        logger.error("Dataset profiling error: %s", e, exc_info=True)
        # Clean up file on failure
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error profiling dataset: {str(e)}")

    dataset_record = DatasetModel(
        id=dataset_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_path=str(dest_path),
        file_size_bytes=len(content),
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns_info=columns_info,
        summary_stats=summary_stats,
        target_column=detected_target,
        task_type=detected_task
    )

    db.add(dataset_record)
    db.commit()
    db.refresh(dataset_record)

    return dataset_record


def configure_target(
    dataset_id: str,
    target_column: str,
    task_type: Optional[str],
    db: Session
) -> DatasetModel:
    """Updates target column and task type with validation."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if target_column not in dataset.columns_info:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist in dataset."
        )

    if not task_type or task_type.lower() == "classification":
        task_type = "classification"
    else:
        raise HTTPException(
            status_code=400,
            detail="AutoFlow AI is specifically an integrated platform for fraud detection. Only 'classification' task type is supported."
        )

    dataset.target_column = target_column
    dataset.task_type = task_type
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset_preview(dataset_id: str, limit: int, db: Session) -> Dict[str, Any]:
    """Returns sample rows of dataset for frontend preview."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    resolved_path = getattr(dataset, "resolved_file_path", dataset.file_path)
    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="Dataset file missing from disk.")

    df = pd.read_csv(resolved_path, nrows=limit)
    # Fill nan with None for valid JSON serialization
    df_clean = df.where(pd.notnull(df), None)

    return {
        "id": dataset.id,
        "columns": list(df.columns),
        "rows": df_clean.to_dict(orient="records"),
        "total_rows": dataset.row_count
    }
