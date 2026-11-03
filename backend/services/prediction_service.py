import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.models_db import MLModelRecord, PredictionLogRecord, MonitoringEventRecord
from backend.automl.model_registry import load_model_artifact
from backend.config import settings

logger = logging.getLogger(__name__)

# In-memory cache for loaded pipelines to avoid reloading disk files on every single-record prediction
_LOADED_PIPELINES_CACHE: Dict[str, Any] = {}

def get_or_load_pipeline(model_record: MLModelRecord) -> Any:
    """Retrieves cached pipeline or loads from disk artifact."""
    model_id = model_record.id
    if model_id in _LOADED_PIPELINES_CACHE:
        return _LOADED_PIPELINES_CACHE[model_id]

    pipeline = load_model_artifact(model_record.artifact_path)
    _LOADED_PIPELINES_CACHE[model_id] = pipeline
    return pipeline


def predict_single_record(
    features: Dict[str, Any],
    db: Session,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes prediction for a single feature dictionary.
    Measures latency and logs to prediction_logs.
    """
    # Find model
    if model_id:
        model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    else:
        model = db.query(MLModelRecord).filter(MLModelRecord.is_active == True).first()

    if not model:
        raise HTTPException(
            status_code=404,
            detail="No active model found. Please train or activate a model first."
        )

    # Validate schema: check expected features
    missing_feats = [f for f in model.feature_names if f not in features]
    if missing_feats and len(missing_feats) > (len(model.feature_names) * 0.5):
        raise HTTPException(
            status_code=422,
            detail=f"Missing key required features: {missing_feats[:5]}"
        )

    # Construct 1-row DataFrame matching model feature order
    row_data = {feat: [features.get(feat, np.nan)] for feat in model.feature_names}
    df_row = pd.DataFrame(row_data)

    pipeline = get_or_load_pipeline(model)

    start_time = time.perf_counter()
    try:
        raw_pred = pipeline.predict(df_row)[0]
    except Exception as e:
        logger.error("Inference prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = float(round((time.perf_counter() - start_time) * 1000, 2))

    # Calculate probabilities if supported
    proba_dict = None
    if model.task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(df_row)[0]
            classes = model.target_classes or [str(c) for c in range(len(probas))]
            proba_dict = {str(cls_name): float(round(p, 4)) for cls_name, p in zip(classes, probas)}
        except Exception as p_err:
            logger.debug("Failed to obtain predict_proba: %s", p_err)

    # Serialize prediction value
    if isinstance(raw_pred, (np.floating, float)):
        pred_val = float(round(raw_pred, 4))
    elif isinstance(raw_pred, (np.integer, int)):
        pred_val = int(raw_pred)
    else:
        pred_val = str(raw_pred)

    # Check for latency spike event
    if latency_ms > settings.LATENCY_WARNING_MS:
        spike_event = MonitoringEventRecord(
            model_id=model.id,
            event_type="LATENCY_SPIKE",
            severity="WARNING",
            details={
                "latency_ms": latency_ms,
                "threshold_ms": settings.LATENCY_WARNING_MS
            }
        )
        db.add(spike_event)

    # Log prediction to DB
    pred_log = PredictionLogRecord(
        model_id=model.id,
        features=features,
        prediction=pred_val,
        probability=proba_dict,
        latency_ms=latency_ms
    )
    db.add(pred_log)
    db.commit()

    return {
        "prediction": pred_val,
        "probability": proba_dict,
        "latency_ms": latency_ms,
        "model_id": model.id,
        "model_version": model.version,
        "prediction_log_id": pred_log.id
    }


def predict_batch_records(
    records: List[Dict[str, Any]],
    db: Session,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes batch inference across a list of JSON records.
    """
    if not records:
        raise HTTPException(status_code=400, detail="Batch records list cannot be empty.")

    if len(records) > 5000:
        raise HTTPException(status_code=400, detail="Batch prediction is capped at 5000 records per request.")

    if model_id:
        model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    else:
        model = db.query(MLModelRecord).filter(MLModelRecord.is_active == True).first()

    if not model:
        raise HTTPException(status_code=404, detail="No active model found.")

    pipeline = get_or_load_pipeline(model)

    # Build DataFrame
    df_batch = pd.DataFrame(records)
    # Reindex to ensure all model features are present
    for feat in model.feature_names:
        if feat not in df_batch.columns:
            df_batch[feat] = np.nan
    df_eval = df_batch[model.feature_names]

    start_time = time.perf_counter()
    try:
        raw_preds = pipeline.predict(df_eval)
    except Exception as e:
        logger.error("Batch inference error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")

    total_latency_ms = float(round((time.perf_counter() - start_time) * 1000, 2))
    avg_per_record = float(round(total_latency_ms / len(records), 3))

    probas = None
    if model.task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(df_eval)
        except Exception:
            pass

    classes = model.target_classes or []
    results = []
    log_objects = []

    for idx, (rec, pred) in enumerate(zip(records, raw_preds)):
        if isinstance(pred, (np.floating, float)):
            val = float(round(pred, 4))
        elif isinstance(pred, (np.integer, int)):
            val = int(pred)
        else:
            val = str(pred)

        prob_map = None
        if probas is not None and idx < len(probas):
            prob_map = {str(classes[c_idx] if c_idx < len(classes) else c_idx): float(round(p, 4))
                        for c_idx, p in enumerate(probas[idx])}

        results.append({
            "index": idx,
            "prediction": val,
            "probability": prob_map
        })

        # Keep DB writes lightweight for batch (log first 50 records)
        if idx < 50:
            log_objects.append(PredictionLogRecord(
                model_id=model.id,
                features=rec,
                prediction=val,
                probability=prob_map,
                latency_ms=avg_per_record
            ))

    if log_objects:
        db.bulk_save_objects(log_objects)
        db.commit()

    return {
        "predictions": results,
        "total_records": len(records),
        "overall_latency_ms": total_latency_ms,
        "average_latency_ms": avg_per_record,
        "model_id": model.id,
        "model_version": model.version
    }


def record_ground_truth(
    prediction_log_id: str,
    actual_label: Any,
    db: Session
) -> Dict[str, Any]:
    """Records ground truth label for a previous prediction."""
    log = db.query(PredictionLogRecord).filter(PredictionLogRecord.id == prediction_log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Prediction log not found.")

    log.actual_label = actual_label
    db.commit()
    return {"status": "success", "prediction_log_id": prediction_log_id, "actual_label": actual_label}
