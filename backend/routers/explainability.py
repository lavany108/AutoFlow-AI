from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import pandas as pd

from backend.database import get_db
from backend.models_db import MLModelRecord, DatasetModel
from backend.schemas import ExplainabilityResponse, FeatureImportanceItem
from backend.automl.explainability import generate_shap_explainability
from backend.automl.model_registry import load_model_artifact

router = APIRouter(prefix="/explainability", tags=["Explainability (SHAP)"])

@router.get("/{model_id}", response_model=ExplainabilityResponse)
def get_model_explainability(
    model_id: str,
    max_samples: Optional[int] = Query(None, ge=10, le=200),
    sample_size: Optional[int] = Query(None, ge=10, le=200),
    db: Session = Depends(get_db)
):
    """
    Computes global feature importance and SHAP explanations for a trained model.
    """
    effective_samples = sample_size or max_samples or 50
    model = db.query(MLModelRecord).filter(MLModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    dataset = db.query(DatasetModel).filter(DatasetModel.id == model.dataset_id).first()
    if not dataset or not dataset.file_path:
        raise HTTPException(status_code=404, detail="Associated dataset not found.")

    try:
        # Load sample data from original dataset
        df = pd.read_csv(getattr(dataset, "resolved_file_path", dataset.file_path))
        if model.target_column in df.columns:
            X_sample = df.drop(columns=[model.target_column])
        else:
            X_sample = df

        # Load pipeline
        pipeline = load_model_artifact(model.artifact_path)

        res = generate_shap_explainability(
            pipeline=pipeline,
            X_sample=X_sample,
            task_type=model.task_type,
            algorithm_name=model.algorithm,
            max_samples=effective_samples
        )

        importance_items = [FeatureImportanceItem(**item) for item in res["feature_importance"]]

        return ExplainabilityResponse(
            model_id=model.id,
            algorithm=model.algorithm,
            task_type=model.task_type,
            explainer_type=res["explainer_type"],
            sample_size=res["sample_size"],
            feature_importance=importance_items,
            summary_plot_base64=res.get("summary_plot_base64"),
            limitations=res.get("limitations")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability computation failed: {str(e)}")
