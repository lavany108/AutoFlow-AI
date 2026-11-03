from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Dataset Schemas
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    inferred_type: str  # numerical, categorical, boolean, datetime, text
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: List[Any] = []

class DatasetUploadResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    row_count: int
    column_count: int
    file_size_bytes: int
    created_at: datetime

class DatasetConfigureTargetRequest(BaseModel):
    target_column: str
    task_type: Optional[str] = "classification"  # Fraud detection is classification

class DatasetDetailResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    row_count: int
    column_count: int
    file_size_bytes: int
    columns_info: Dict[str, Any]
    summary_stats: Dict[str, Any]
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    potential_targets: List[str] = []
    imbalance_stats: Optional[Dict[str, Any]] = None
    created_at: datetime

class DatasetPreviewResponse(BaseModel):
    id: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int

# ---------------------------------------------------------------------------
# Training & Benchmark Schemas
# ---------------------------------------------------------------------------

class StartBenchmarkRequest(BaseModel):
    dataset_id: str
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42
    cv_splits: int = Field(5, ge=2, le=10)
    use_smote: bool = True  # Fraud-specific: SMOTE oversampling

class LeaderboardItem(BaseModel):
    model_id: Optional[str] = None
    algorithm: str
    task_type: str
    primary_metric_name: str
    primary_metric_value: float
    cv_mean: float
    cv_std: float
    fit_time_seconds: float
    metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    status: str = "SUCCESS"
    error_message: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    dataset_id: str
    job_type: str
    status: str
    progress: float
    current_step: str
    leaderboard: List[LeaderboardItem] = []
    best_model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# ---------------------------------------------------------------------------
# Tuning Schemas
# ---------------------------------------------------------------------------

class StartTuningRequest(BaseModel):
    model_id: str
    n_trials: int = Field(10, ge=3, le=50)
    timeout_seconds: Optional[int] = Field(180, ge=10, le=600)

class TuningTrialInfo(BaseModel):
    trial_number: int
    value: float
    params: Dict[str, Any]

class TuningResultResponse(BaseModel):
    job_id: str
    status: str
    original_model_id: str
    tuned_model_id: Optional[str] = None
    best_params: Dict[str, Any] = {}
    best_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0
    trials_count: int = 0
    duration_seconds: float = 0.0
    trials: List[TuningTrialInfo] = []

# ---------------------------------------------------------------------------
# Model Registry Schemas
# ---------------------------------------------------------------------------

class MLModelSummary(BaseModel):
    id: str
    name: str
    version: int
    dataset_id: str
    algorithm: str
    task_type: str
    target_column: str
    metrics: Dict[str, Any]
    is_active: bool
    is_tuned: bool
    mlflow_run_id: Optional[str] = None
    created_at: datetime

class MLModelDetail(MLModelSummary):
    feature_names: List[str]
    target_classes: Optional[List[Any]] = None
    hyperparameters: Dict[str, Any]
    artifact_path: str
    training_duration_seconds: float

# ---------------------------------------------------------------------------
# Prediction / Scoring Schemas
# ---------------------------------------------------------------------------

class SinglePredictionRequest(BaseModel):
    model_id: Optional[str] = None  # uses active model if None
    features: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    model_id: Optional[str] = None
    records: List[Dict[str, Any]]

class PredictionOutput(BaseModel):
    prediction: Any
    probability: Optional[Dict[str, float]] = None
    fraud_score: Optional[float] = None          # probability of fraud (0–1)
    risk_level: Optional[str] = None             # LOW / MEDIUM / HIGH / CRITICAL
    latency_ms: float
    model_id: str
    model_version: int

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_records: int
    overall_latency_ms: float
    model_id: str
    model_version: int
    fraud_count: Optional[int] = None
    fraud_rate: Optional[float] = None

class ActualOutcomeRequest(BaseModel):
    prediction_log_id: str
    actual_label: Any

# ---------------------------------------------------------------------------
# Threshold Optimization Schemas
# ---------------------------------------------------------------------------

class ThresholdOptimizeRequest(BaseModel):
    model_id: Optional[str] = None    # uses active model if None
    strategy: str = Field("f2", pattern="^(f1|f2|cost|precision_at_recall)$")
    min_recall: float = Field(0.80, ge=0.5, le=0.99)
    cost_fp: float = Field(10.0, ge=0.0)    # cost per false alarm
    cost_fn: float = Field(500.0, ge=0.0)   # cost per missed fraud

class ThresholdOptimizeResponse(BaseModel):
    model_id: str
    strategy: str
    optimal_threshold: float
    optimal_precision: float
    optimal_recall: float
    optimal_f1: float
    optimal_f2: float
    tp: int
    tn: int
    fp: int
    fn: int
    threshold_scores: List[Dict[str, Any]] = []
    business_impact: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Fraud Scoring Schemas
# ---------------------------------------------------------------------------

class FraudScoreRequest(BaseModel):
    model_id: Optional[str] = None
    transaction: Dict[str, Any]
    threshold: Optional[float] = None   # if None, uses model default (0.5)

class FraudScoreResponse(BaseModel):
    fraud_score: float           # 0–1 probability
    risk_level: str              # LOW / MEDIUM / HIGH / CRITICAL
    predicted_fraud: bool
    threshold_used: float
    explanation_top_features: Optional[List[Dict[str, Any]]] = None
    latency_ms: float
    model_id: str
    prediction_log_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Explainability Schemas
# ---------------------------------------------------------------------------

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    mean_abs_shap: float
    rank: int

class ExplainabilityResponse(BaseModel):
    model_id: str
    algorithm: str
    task_type: str
    explainer_type: str
    sample_size: int
    feature_importance: List[FeatureImportanceItem]
    summary_plot_base64: Optional[str] = None
    limitations: Optional[str] = None

# ---------------------------------------------------------------------------
# Monitoring Schemas
# ---------------------------------------------------------------------------

class FeatureDriftItem(BaseModel):
    feature: str
    feature_type: str
    statistic_name: str
    statistic_value: float
    p_value: float
    threshold: float
    is_drifted: bool
    ref_mean_or_mode: Any
    curr_mean_or_mode: Any

class MonitoringSummaryResponse(BaseModel):
    model_id: str
    model_name: str
    total_predictions: int
    recent_predictions_analyzed: int
    mean_latency_ms: float
    p95_latency_ms: float
    error_count: int
    drifted_features_count: int
    total_features_monitored: int
    drift_detected: bool
    concept_drift_status: str
    concept_drift_metrics: Optional[Dict[str, Any]] = None
    feature_drift: List[FeatureDriftItem] = []
    recent_events: List[Dict[str, Any]] = []
    # Fraud-specific
    fraud_rate_trend: Optional[List[Dict[str, Any]]] = None
    false_positive_rate: Optional[float] = None
    risk_distribution: Optional[Dict[str, int]] = None

class RetrainTriggerRequest(BaseModel):
    model_id: Optional[str] = None
    reason: str = "Manual trigger from UI"
    performance_tolerance: float = 0.05
