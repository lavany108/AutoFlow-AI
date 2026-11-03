from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import (
    SinglePredictionRequest,
    PredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ActualOutcomeRequest
)
from backend.services.prediction_service import (
    predict_single_record,
    predict_batch_records,
    record_ground_truth
)

router = APIRouter(prefix="/predictions", tags=["Inference & Predictions"])

@router.post("/single", response_model=PredictionOutput)
def predict_single(
    req: SinglePredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Real-time inference endpoint for single-record prediction with latency tracking.
    """
    res = predict_single_record(
        features=req.features,
        db=db,
        model_id=req.model_id
    )
    return PredictionOutput(
        prediction=res["prediction"],
        probability=res["probability"],
        latency_ms=res["latency_ms"],
        model_id=res["model_id"],
        model_version=res["model_version"]
    )


@router.post("/predict", response_model=PredictionOutput)
def predict_alias(
    req: SinglePredictionRequest,
    db: Session = Depends(get_db)
):
    """Alias for /single endpoint."""
    return predict_single(req, db)


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(
    req: BatchPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Batch inference endpoint for JSON records array.
    """
    res = predict_batch_records(
        records=req.records,
        db=db,
        model_id=req.model_id
    )
    return BatchPredictionResponse(
        predictions=res["predictions"],
        total_records=res["total_records"],
        overall_latency_ms=res["overall_latency_ms"],
        model_id=res["model_id"],
        model_version=res["model_version"]
    )


@router.post("/actual-outcome")
def submit_actual_outcome(
    req: ActualOutcomeRequest,
    db: Session = Depends(get_db)
):
    """
    Records delayed ground-truth outcome for concept drift evaluation.
    """
    return record_ground_truth(
        prediction_log_id=req.prediction_log_id,
        actual_label=req.actual_label,
        db=db
    )
