import io
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from backend.automl.preprocessing import profile_dataset, detect_task_type, build_preprocessor, extract_reference_distribution
from backend.automl.model_selection import benchmark_models
from backend.automl.tuning import tune_model
from backend.automl.explainability import generate_shap_explainability
from backend.automl.monitoring import detect_numerical_drift, detect_categorical_drift, analyze_drift_and_latency
from backend.automl.model_registry import save_model_artifact, load_model_artifact
from backend.models_db import DatasetModel, MLModelRecord, RetrainJobRecord
from backend.automl.retraining import execute_retraining_workflow

def create_synthetic_classification_df():
    np.random.seed(42)
    n = 60
    age = np.random.randint(20, 65, size=n)
    income = np.random.normal(50000, 15000, size=n)
    category = np.random.choice(["Tier1", "Tier2", "Tier3"], size=n)
    churn = np.where((income > 55000) | (category == "Tier1"), "Yes", "No")
    return pd.DataFrame({"age": age, "income": income, "category": category, "churn": churn})

# 1. Health check test
def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "service" in data

# 2. Upload CSV and Profiling test
def test_csv_upload_and_profiling(client):
    df = create_synthetic_classification_df()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    
    files = {"file": ("test_churn.csv", io.BytesIO(csv_bytes), "text/csv")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert data["row_count"] == 60
    assert data["column_count"] == 4

    # Fetch dataset details
    det_res = client.get(f"/api/datasets/{data['id']}")
    assert det_res.status_code == 200
    det_data = det_res.json()
    assert "columns_info" in det_data
    assert "age" in det_data["columns_info"]
    assert det_data["columns_info"]["age"]["inferred_type"] == "numerical"
    assert det_data["columns_info"]["category"]["inferred_type"] == "categorical"
    assert det_data["target_column"] == "churn"
    assert det_data["task_type"] == "classification"

# 3. Invalid file rejection test
def test_invalid_file_upload(client):
    files = {"file": ("test.txt", io.BytesIO(b"random text"), "text/plain")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 400

# 4. Target and Task configuration test
def test_configure_target_column(client):
    df = create_synthetic_classification_df()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("test_target.csv", io.BytesIO(csv_bytes), "text/csv")}
    up_res = client.post("/api/datasets/upload", files=files)
    ds_id = up_res.json()["id"]

    # Configure target with classification
    cfg_res = client.post(
        f"/api/datasets/{ds_id}/target",
        json={"target_column": "churn", "task_type": "classification"}
    )
    assert cfg_res.status_code == 200
    assert cfg_res.json()["target_column"] == "churn"
    assert cfg_res.json()["task_type"] == "classification"

    # Verify rejection of non-classification task types (fraud platform strictly supports classification)
    bad_res = client.post(
        f"/api/datasets/{ds_id}/target",
        json={"target_column": "income", "task_type": "regression"}
    )
    assert bad_res.status_code == 400
    assert "classification" in bad_res.json()["detail"].lower()

# 5. Preprocessing pipeline test
def test_preprocessing_pipeline():
    df = create_synthetic_classification_df()
    X = df.drop(columns=["churn"])
    preprocessor = build_preprocessor(X, numerical_cols=["age", "income"], categorical_cols=["category"])
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape[0] == 60
    assert X_trans.shape[1] >= 5  # 2 numerical + 3 one-hot columns

# 6. Classification benchmark test
def test_classification_benchmark():
    df = create_synthetic_classification_df()
    results = benchmark_models(
        df=df,
        target_column="churn",
        task_type="classification",
        test_size=0.25,
        cv_splits=3,
        random_state=42
    )
    assert "leaderboard" in results
    assert len(results["leaderboard"]) == 5
    best_algo = results["best_algorithm"]
    assert best_algo in [item["algorithm"] for item in results["leaderboard"]]
    assert "best_pipeline" in results
    assert "f1" in results["best_metrics"]

# 7. Model persistence and loading test
def test_model_persistence():
    df = create_synthetic_classification_df()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    pre = build_preprocessor(X, ["age", "income"], ["category"])
    pipe = Pipeline([("preprocessor", pre), ("estimator", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X, y)

    art_path, ref_path = save_model_artifact("test_model_123", pipe, {"age": {"type": "numerical"}})
    loaded_pipe = load_model_artifact(art_path)
    pred = loaded_pipe.predict(X.head(2))
    assert len(pred) == 2

# 9. Prediction test
def test_single_and_batch_prediction(client, db_session):
    # Set up model in DB
    df = create_synthetic_classification_df()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    pre = build_preprocessor(X, ["age", "income"], ["category"])
    pipe = Pipeline([("preprocessor", pre), ("estimator", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X, y)

    art_path, ref_path = save_model_artifact("test_pred_model", pipe, {})
    model_rec = MLModelRecord(
        id="test_pred_model",
        name="Test RF",
        version=1,
        dataset_id="test_ds",
        algorithm="Random Forest Classifier",
        task_type="classification",
        target_column="churn",
        feature_names=["age", "income", "category"],
        target_classes=["No", "Yes"],
        hyperparameters={"n_estimators": 10},
        metrics={"f1": 0.90, "accuracy": 0.90},
        artifact_path=art_path,
        is_active=True
    )
    db_session.add(model_rec)
    db_session.commit()

    # Single prediction
    pred_res = client.post("/api/predictions/single", json={
        "features": {"age": 35, "income": 52000, "category": "Tier1"}
    })
    assert pred_res.status_code == 200
    data = pred_res.json()
    assert data["prediction"] in ["Yes", "No"]
    assert "latency_ms" in data
    assert "probability" in data

    # Batch prediction
    batch_res = client.post("/api/predictions/batch", json={
        "records": [
            {"age": 25, "income": 40000, "category": "Tier2"},
            {"age": 55, "income": 75000, "category": "Tier1"}
        ]
    })
    assert batch_res.status_code == 200
    b_data = batch_res.json()
    assert len(b_data["predictions"]) == 2

# 10. SHAP explainability test
def test_shap_explainability():
    df = create_synthetic_classification_df()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    pre = build_preprocessor(X, ["age", "income"], ["category"])
    pipe = Pipeline([("preprocessor", pre), ("estimator", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X, y)

    shap_res = generate_shap_explainability(pipe, X, "classification", "Random Forest Classifier", max_samples=20)
    assert "feature_importance" in shap_res
    assert len(shap_res["feature_importance"]) > 0
    assert shap_res["feature_importance"][0]["rank"] == 1

# 11. Monitoring and drift statistics test
def test_monitoring_drift():
    ref_vals = list(np.random.normal(50, 10, size=50))
    # Similar sample -> no drift
    curr_similar = list(np.random.normal(50, 10, size=30))
    res_stable = detect_numerical_drift(ref_vals, curr_similar, alpha=0.05)
    assert not res_stable["is_drifted"]

    # Shifted sample -> drift detected
    curr_drifted = list(np.random.normal(120, 10, size=30))
    res_drift = detect_numerical_drift(ref_vals, curr_drifted, alpha=0.05)
    assert res_drift["is_drifted"]


# 12. Fraud dataset profiling and auto-detected target with imbalance stats
def test_fraud_dataset_upload_and_imbalance_stats(client):
    # Create sample fraud dataset with 10% fraud
    np.random.seed(42)
    n = 100
    amounts = np.random.exponential(50, size=n)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:10] = 1  # 10 frauds

    df = pd.DataFrame({
        "amount": amounts,
        "hour": np.random.randint(0, 24, size=n),
        "is_foreign": np.random.choice([0, 1], size=n),
        "is_fraud": is_fraud
    })

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("fraud_sample.csv", io.BytesIO(csv_bytes), "text/csv")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 200
    ds_id = res.json()["id"]

    det_res = client.get(f"/api/datasets/{ds_id}")
    assert det_res.status_code == 200
    data = det_res.json()
    assert data["target_column"] == "is_fraud"
    assert data["task_type"] == "classification"
    assert data["imbalance_stats"] is not None
    assert data["imbalance_stats"]["fraud_count"] == 10
    assert data["imbalance_stats"]["legit_count"] == 90
    assert data["imbalance_stats"]["fraud_ratio"] == 0.1
    assert "severity" in data["imbalance_stats"]["imbalance_severity"].lower() or "moderate" in data["imbalance_stats"]["imbalance_severity"].lower()


# 13. Fraud-specific benchmarking with PR-AUC, F2, specificity, and PR-curve
def test_fraud_model_training_and_metrics():
    np.random.seed(42)
    n = 80
    amounts = np.random.exponential(50, size=n)
    velocities = np.random.poisson(2, size=n)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:10] = 1

    df = pd.DataFrame({
        "amount": amounts,
        "velocity": velocities,
        "is_fraud": is_fraud
    })

    results = benchmark_models(
        df=df,
        target_column="is_fraud",
        task_type="classification",
        test_size=0.25,
        cv_splits=3,
        random_state=42,
        use_smote=True
    )

    assert "leaderboard" in results
    assert len(results["leaderboard"]) == 5
    best_metrics = results["best_metrics"]
    assert "pr_auc" in best_metrics
    assert "f2" in best_metrics
    assert "specificity" in best_metrics
    assert "pr_curve" in best_metrics
    assert len(best_metrics["pr_curve"]["precision"]) > 0


def _create_test_fraud_model(db_session, model_id="test_fraud_model_thresh"):
    from sklearn.ensemble import ExtraTreesClassifier
    pr_curve = {
        "precision": [0.2, 0.4, 0.6, 0.8, 0.95],
        "recall": [1.0, 0.9, 0.7, 0.5, 0.2],
        "thresholds": [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    X_sample = pd.DataFrame({"amount": [10.0, 20.0, 1500.0, 50.0], "velocity": [1, 2, 12, 1]})
    y_sample = [0, 0, 1, 0]
    pre = build_preprocessor(X_sample, ["amount", "velocity"], [])
    pipe = Pipeline([
        ("preprocessor", pre),
        ("estimator", ExtraTreesClassifier(n_estimators=5, random_state=42))
    ])
    pipe.fit(X_sample, y_sample)
    art_path, _ = save_model_artifact(model_id, pipe, {})
    model_rec = MLModelRecord(
        id=model_id,
        name="Fraud Extra Trees",
        version=1,
        dataset_id="ds_fraud",
        algorithm="Extra Trees Classifier",
        task_type="classification",
        target_column="is_fraud",
        feature_names=["amount", "velocity"],
        target_classes=["0", "1"],
        hyperparameters={"n_estimators": 50},
        metrics={
            "pr_auc": 0.85,
            "f2": 0.80,
            "fraud_count": 20,
            "legit_count": 180,
            "pr_curve": pr_curve,
            "confusion_matrix": {"matrix": [[170, 10], [4, 16]], "labels": ["0", "1"]}
        },
        artifact_path=art_path,
        is_active=True
    )
    db_session.add(model_rec)
    db_session.commit()
    return model_rec


# 14. Threshold optimization endpoint and business impact calculations
def test_fraud_threshold_optimization(client, db_session):
    _create_test_fraud_model(db_session, "test_fraud_model_thresh")

    # Test F2 optimization
    res_f2 = client.post("/api/fraud/optimize-threshold", json={
        "model_id": "test_fraud_model_thresh",
        "strategy": "f2"
    })
    assert res_f2.status_code == 200
    data_f2 = res_f2.json()
    assert 0.0 < data_f2["optimal_threshold"] < 1.0
    assert "business_impact" in data_f2
    assert "estimated_savings" in data_f2["business_impact"]

    # Test business impact endpoint
    impact_res = client.get("/api/fraud/business-impact/test_fraud_model_thresh?cost_fp=15&cost_fn=600")
    assert impact_res.status_code == 200
    impact_data = impact_res.json()
    assert impact_data["cost_fp"] == 15
    assert impact_data["cost_fn"] == 600
    assert "threshold_sensitivity" in impact_data


# 15. Transaction scoring with risk level, SHAP explanation, and PredictionLog persistence
def test_fraud_transaction_scoring_and_logging(client, db_session):
    _create_test_fraud_model(db_session, "test_fraud_model_score")

    score_res = client.post("/api/fraud/score-transaction", json={
        "model_id": "test_fraud_model_score",
        "transaction": {"amount": 2500.0, "velocity": 15},
        "threshold": 0.4
    })
    assert score_res.status_code == 200
    score_data = score_res.json()
    assert "fraud_score" in score_data
    assert score_data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert "latency_ms" in score_data
    assert score_data["threshold_used"] == 0.4
    assert score_data["prediction_log_id"] is not None


# 16. Test route aliases and management endpoints
def test_route_aliases_and_management(client):
    # Test GET /api/training/jobs
    jobs_res = client.get("/api/training/jobs")
    assert jobs_res.status_code == 200
    assert isinstance(jobs_res.json(), list)

    # Test GET /api/retraining/jobs
    retrain_res = client.get("/api/retraining/jobs")
    assert retrain_res.status_code == 200
    assert isinstance(retrain_res.json(), list)

