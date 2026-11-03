import json
import os
from pathlib import Path
from typing import Union
from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "AutoFlow AI"
    API_V1_PREFIX: str = "/api"
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    STORAGE_DIR: Path = BASE_DIR / "storage"
    DATASETS_DIR: Path = BASE_DIR / "storage" / "datasets"
    MODELS_DIR: Path = BASE_DIR / "storage" / "models"
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR}/storage/mlflow.db")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/storage/autoflow.db")
    
    # CORS
    CORS_ORIGINS: Union[list[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, list[str]]) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if item]
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return [str(item).strip() for item in v if item]
        return []
    
    # Resource Limits
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    MAX_ROWS_FOR_IN_MEMORY_PREVIEW: int = 50
    DEFAULT_CV_SPLITS: int = 5
    DEFAULT_OPTUNA_TRIALS: int = 10
    MAX_OPTUNA_TRIALS: int = 50
    
    # Monitoring & Drift
    DEFAULT_DRIFT_THRESHOLD_PVALUE: float = 0.05
    LATENCY_WARNING_MS: float = 250.0

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
if settings.MLFLOW_TRACKING_URI.startswith("sqlite:///"):
    Path(settings.MLFLOW_TRACKING_URI.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
elif not settings.MLFLOW_TRACKING_URI.startswith(("http://", "https://", "file://")):
    Path(settings.MLFLOW_TRACKING_URI).mkdir(parents=True, exist_ok=True)
