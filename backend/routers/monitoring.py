from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import MonitoringSummaryResponse, FeatureDriftItem
from backend.services.monitoring_service import get_monitoring_summary

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Drift"])

@router.get("/summary", response_model=MonitoringSummaryResponse)
def get_monitoring(
    model_id: Optional[str] = Query(None),
    threshold_pvalue: float = Query(0.05, ge=0.001, le=0.5),
    limit: int = Query(200, ge=10, le=1000),
    db: Session = Depends(get_db)
):
    """
    Returns live prediction latency, statistical feature drift tests, and recent monitoring events.
    """
    summary = get_monitoring_summary(
        db=db,
        model_id=model_id,
        threshold_pvalue=threshold_pvalue,
        limit_recent=limit
    )

    drift_items = [FeatureDriftItem(**it) for it in summary.get("feature_drift", [])]

    return MonitoringSummaryResponse(
        model_id=summary["model_id"],
        model_name=summary["model_name"],
        total_predictions=summary["total_predictions"],
        recent_predictions_analyzed=summary["recent_predictions_analyzed"],
        mean_latency_ms=summary["mean_latency_ms"],
        p95_latency_ms=summary["p95_latency_ms"],
        error_count=summary["error_count"],
        drifted_features_count=summary["drifted_features_count"],
        total_features_monitored=summary["total_features_monitored"],
        drift_detected=summary["drift_detected"],
        concept_drift_status=summary["concept_drift_status"],
        concept_drift_metrics=summary.get("concept_drift_metrics"),
        feature_drift=drift_items,
        recent_events=summary.get("recent_events", []),
        fraud_rate_trend=summary.get("fraud_rate_trend"),
        risk_distribution=summary.get("risk_distribution")
    )
