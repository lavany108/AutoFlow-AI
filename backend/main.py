import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import init_db
from backend.routers import (
    datasets,
    training,
    models,
    predictions,
    explainability,
    monitoring,
    retraining,
    fraud,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("autoflow")

app = FastAPI(
    title="AutoFlow AI — Fraud Detection Platform",
    description="End-to-End Financial Transaction Fraud Detection & MLOps Platform API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    logger.info("Initializing AutoFlow Fraud Detection database schema...")
    init_db()
    # Reset any lingering RUNNING jobs from previous server interruptions
    try:
        from backend.database import SessionLocal
        from backend.models_db import TrainingJobRecord
        with SessionLocal() as db:
            db.query(TrainingJobRecord).filter(TrainingJobRecord.status == "RUNNING").update({
                "status": "FAILED",
                "error_message": "Previous job was interrupted by server reload or memory limit. Ready to restart."
            })
            db.commit()
    except Exception as e:
        logger.warning("Could not clean up dangling training jobs: %s", e)
    logger.info("AutoFlow AI Fraud Detection Platform ready!")

@app.get("/api/health", tags=["System"])
def health_check():
    """Health check endpoint providing system status and timestamp."""
    return {
        "status": "healthy",
        "service": "AutoFlow AI",
        "timestamp": datetime.utcnow().isoformat(),
        "storage_ready": settings.STORAGE_DIR.exists()
    }

# Register API Routers
app.include_router(datasets.router, prefix=settings.API_V1_PREFIX)
app.include_router(training.router, prefix=settings.API_V1_PREFIX)
app.include_router(models.router, prefix=settings.API_V1_PREFIX)
app.include_router(predictions.router, prefix=settings.API_V1_PREFIX)
app.include_router(explainability.router, prefix=settings.API_V1_PREFIX)
app.include_router(monitoring.router, prefix=settings.API_V1_PREFIX)
app.include_router(retraining.router, prefix=settings.API_V1_PREFIX)
app.include_router(fraud.router, prefix=settings.API_V1_PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
