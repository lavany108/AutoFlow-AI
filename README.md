# AutoFlow AI

An end-to-end **AutoML + MLOps** mini project. Upload a raw CSV and AutoFlow AI takes it
through the full ML lifecycle — preprocessing, feature engineering, model benchmarking,
hyperparameter tuning, explainability, deployment, and drift monitoring — with almost no
manual configuration.

```
Upload CSV → Profile & preprocess → Benchmark 5 algorithms → Optuna tuning
    → Track (MLflow) → Serve as REST API → SHAP explainability
    → Monitor drift & latency → Auto-trigger retraining
```

## Tech stack

| Layer            | Tech                                                              |
|-------------------|-------------------------------------------------------------------|
| Backend API       | FastAPI, Uvicorn                                                   |
| ML / AutoML       | scikit-learn, Optuna, SHAP, pandas, NumPy, SciPy                   |
| Experiment tracking | MLflow (local file store by default)                             |
| Metadata storage  | SQLAlchemy — SQLite by default, PostgreSQL via `DATABASE_URL`      |
| Frontend          | React + Vite                                                       |
| Deployment        | Docker, docker-compose                                             |

## Project layout

```
autoflow-ai/
├── backend/
│   ├── main.py                # FastAPI app: all API routes
│   ├── database.py            # SQLAlchemy engine/session (SQLite/Postgres)
│   ├── models_db.py           # ORM tables: Dataset, MLModel, MonitoringEvent, RetrainJob
│   ├── schemas.py             # Pydantic request/response models
│   ├── automl/
│   │   ├── preprocessing.py   # task detection, column typing, ColumnTransformer builder
│   │   ├── model_selection.py # cross-validated benchmarking of candidate algorithms
│   │   ├── tuning.py          # Optuna hyperparameter search per algorithm
│   │   ├── explainability.py  # SHAP global importance + summary plot
│   │   └── monitoring.py      # KS-test / chi-square drift detection, latency stats
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # pipeline-stage navigation shell
│   │   ├── api.js             # fetch wrapper for the backend API
│   │   └── components/        # UploadDataset, TrainModel, PredictPanel, ExplainPanel, MonitorPanel
│   └── Dockerfile
├── sample_data/
│   └── customer_churn.csv     # synthetic dataset to try the pipeline immediately
└── docker-compose.yml
```