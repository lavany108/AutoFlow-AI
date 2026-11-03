import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Text
)
from sqlalchemy.orm import relationship
from backend.database import Base

def generate_uuid() -> str:
    return uuid.uuid4().hex

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    row_count = Column(Integer, nullable=False, default=0)
    column_count = Column(Integer, nullable=False, default=0)
    columns_info = Column(JSON, nullable=False, default=dict)
    summary_stats = Column(JSON, nullable=False, default=dict)
    target_column = Column(String(128), nullable=True)
    task_type = Column(String(32), default="classification", nullable=True)  # 'classification' (fraud detection)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    models = relationship("MLModelRecord", back_populates="dataset", cascade="all, delete-orphan")

    @property
    def resolved_file_path(self) -> str:
        """Resolves file path across host and Docker container environments."""
        from pathlib import Path
        from backend.config import settings
        p = Path(self.file_path)
        if p.is_file():
            return str(p)
        candidate = settings.DATASETS_DIR / p.name
        if candidate.is_file():
            return str(candidate)
        return self.file_path


class MLModelRecord(Base):
    __tablename__ = "ml_models"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(128), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    algorithm = Column(String(64), nullable=False)
    task_type = Column(String(32), default="classification", nullable=False)  # 'classification' (fraud detection)
    target_column = Column(String(128), nullable=False)
    feature_names = Column(JSON, nullable=False, default=list)
    target_classes = Column(JSON, nullable=True)  # List of classes for classification
    hyperparameters = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    artifact_path = Column(String(1024), nullable=False)
    reference_distribution_path = Column(String(1024), nullable=True)
    mlflow_run_id = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    is_tuned = Column(Boolean, default=False, nullable=False)
    training_duration_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship("DatasetModel", back_populates="models")
    predictions = relationship("PredictionLogRecord", back_populates="model", cascade="all, delete-orphan")
    monitoring_events = relationship("MonitoringEventRecord", back_populates="model", cascade="all, delete-orphan")
    retrain_jobs = relationship("RetrainJobRecord", back_populates="model", foreign_keys="RetrainJobRecord.model_id", cascade="all, delete-orphan")


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), nullable=False)
    job_type = Column(String(32), nullable=False)  # 'BENCHMARK', 'TUNE', 'RETRAIN'
    status = Column(String(32), nullable=False, default="PENDING")  # 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
    progress = Column(Float, nullable=False, default=0.0)
    current_step = Column(String(255), nullable=False, default="Initialized")
    leaderboard = Column(JSON, nullable=False, default=list)
    best_model_id = Column(String(36), nullable=True)
    result_summary = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class PredictionLogRecord(Base):
    __tablename__ = "prediction_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    features = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)
    probability = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=False, default=0.0)
    actual_label = Column(JSON, nullable=True)  # Ground truth if recorded later
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("MLModelRecord", back_populates="predictions")


class MonitoringEventRecord(Base):
    __tablename__ = "monitoring_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String(64), nullable=False)  # 'DRIFT_DETECTED', 'LATENCY_SPIKE', 'RETRAIN_TRIGGERED', 'CONCEPT_DRIFT'
    severity = Column(String(16), nullable=False, default="INFO")  # 'INFO', 'WARNING', 'CRITICAL'
    details = Column(JSON, nullable=False, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("MLModelRecord", back_populates="monitoring_events")


class RetrainJobRecord(Base):
    __tablename__ = "retrain_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(String(36), nullable=False)
    triggered_by = Column(String(64), nullable=False)  # 'DRIFT_THRESHOLD', 'MANUAL', 'EVALUATION_DEGRADATION'
    status = Column(String(32), nullable=False, default="PENDING")  # 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
    candidate_model_id = Column(String(36), nullable=True)
    old_model_id = Column(String(36), nullable=True)
    comparison_metrics = Column(JSON, nullable=True)
    promoted = Column(Boolean, default=False, nullable=False)
    decision_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    model = relationship("MLModelRecord", back_populates="retrain_jobs", foreign_keys=[model_id])
