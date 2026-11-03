import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.models_db import MLModelRecord, PredictionLogRecord, MonitoringEventRecord
from backend.automl.model_registry import load_reference_distribution
from backend.automl.monitoring import analyze_drift_and_latency, evaluate_concept_drift
from backend.config import settings

logger = logging.getLogger(__name__)

def get_monitoring_summary(
    db: Session,
    model_id: Optional[str] = None,
    threshold_pvalue: float = settings.DEFAULT_DRIFT_THRESHOLD_PVALUE,
    limit_recent: int = 200
) -> Dict[str, Any]:
    """
    Computes real-time drift, latency metrics, concept drift, and recent monitoring events.
    """
    if model_id:
        model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    else:
        model = db.query(MLModelRecord).filter(MLModelRecord.is_active == True).first()

    if not model:
        raise HTTPException(status_code=404, detail="No active model available for monitoring.")

    # Load reference distribution
    ref_dist = load_reference_distribution(model.reference_distribution_path)

    # Fetch recent prediction logs
    logs_query = db.query(PredictionLogRecord).filter(
        PredictionLogRecord.model_id == model.id
    ).order_by(desc(PredictionLogRecord.timestamp))

    total_predictions = logs_query.count()
    recent_logs = logs_query.limit(limit_recent).all()

    # Convert logs to dict format
    recent_records = [
        {
            "id": log.id,
            "features": log.features,
            "prediction": log.prediction,
            "latency_ms": log.latency_ms,
            "timestamp": log.timestamp.isoformat(),
            "actual_label": log.actual_label
        }
        for log in recent_logs
    ]

    # Run statistical drift and latency analysis
    drift_res = analyze_drift_and_latency(
        reference_distribution=ref_dist,
        recent_prediction_records=recent_records,
        threshold_pvalue=threshold_pvalue
    )

    # If drift is detected, record event in DB if not recorded recently
    if drift_res.get("drift_detected", False):
        ten_mins_ago = datetime.utcnow() - timedelta(minutes=10)
        recent_drift_event = db.query(MonitoringEventRecord).filter(
            MonitoringEventRecord.model_id == model.id,
            MonitoringEventRecord.event_type == "DRIFT_DETECTED",
            MonitoringEventRecord.timestamp >= ten_mins_ago
        ).first()

        if not recent_drift_event:
            new_event = MonitoringEventRecord(
                model_id=model.id,
                event_type="DRIFT_DETECTED",
                severity="WARNING",
                details={
                    "drifted_features": [f["feature"] for f in drift_res["feature_drift"] if f["is_drifted"]],
                    "drifted_count": drift_res["drifted_features_count"],
                    "total_features": drift_res["total_features_monitored"]
                }
            )
            db.add(new_event)
            db.commit()

    # Concept drift check (predictions with ground truth)
    labeled_records = [r for r in recent_records if r.get("actual_label") is not None]
    concept_drift_info = evaluate_concept_drift(
        logged_predictions_with_labels=labeled_records,
        task_type=model.task_type,
        baseline_metrics=model.metrics
    )

    # Fetch recent monitoring events
    events = db.query(MonitoringEventRecord).filter(
        MonitoringEventRecord.model_id == model.id
    ).order_by(desc(MonitoringEventRecord.timestamp)).limit(20).all()

    recent_events_data = [
        {
            "id": ev.id,
            "event_type": ev.event_type,
            "severity": ev.severity,
            "details": ev.details,
            "timestamp": ev.timestamp.isoformat()
        }
        for ev in events
    ]

    # Error count (e.g. predictions with latency spikes or severe events)
    error_count = db.query(MonitoringEventRecord).filter(
        MonitoringEventRecord.model_id == model.id,
        MonitoringEventRecord.severity.in_(["WARNING", "CRITICAL"])
    ).count()

    # Calculate real fraud metrics and risk distribution from recent prediction logs
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    hourly_buckets: Dict[str, Dict[str, int]] = {}
    for r in recent_logs:
        prob = r.probability or {}
        score = float(prob.get("1", prob.get("fraud", prob.get("True", 0.0))))
        if score < 0.3:
            risk_counts["LOW"] += 1
        elif score < 0.5:
            risk_counts["MEDIUM"] += 1
        elif score < 0.75:
            risk_counts["HIGH"] += 1
        else:
            risk_counts["CRITICAL"] += 1

        hour_key = r.timestamp.strftime("%H:00") if r.timestamp else "recent"
        if hour_key not in hourly_buckets:
            hourly_buckets[hour_key] = {"transactions": 0, "fraud": 0}
        hourly_buckets[hour_key]["transactions"] += 1
        if str(r.prediction).lower() in ["1", "true", "fraud", "yes"]:
            hourly_buckets[hour_key]["fraud"] += 1

    trend_list = [
        {"hour": h, "transactions": b["transactions"], "fraud": b["fraud"]}
        for h, b in sorted(hourly_buckets.items())
    ]

    return {
        "model_id": model.id,
        "model_name": model.name,
        "total_predictions": total_predictions,
        "recent_predictions_analyzed": drift_res["recent_predictions_analyzed"],
        "mean_latency_ms": drift_res["mean_latency_ms"],
        "p95_latency_ms": drift_res["p95_latency_ms"],
        "error_count": error_count,
        "drifted_features_count": drift_res["drifted_features_count"],
        "total_features_monitored": drift_res["total_features_monitored"],
        "drift_detected": drift_res["drift_detected"],
        "concept_drift_status": concept_drift_info.get("status", "unknown"),
        "concept_drift_metrics": concept_drift_info.get("current_metrics"),
        "concept_drift_message": concept_drift_info.get("message"),
        "feature_drift": drift_res["feature_drift"],
        "recent_events": recent_events_data,
        "fraud_rate_trend": trend_list,
        "risk_distribution": risk_counts,
    }
