import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

def profile_dataset(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Optional[str], Optional[str]]:
    """
    Profiles raw pandas dataframe:
    - detects column types, missing rates, unique counts
    - generates summary statistics
    - suggests potential target columns
    - detects default target and task type
    """
    n_rows, n_cols = df.shape
    columns_info = {}
    potential_targets = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = float(round((missing_count / n_rows) * 100, 2)) if n_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)

        # Inferred type logic
        inferred_type = "categorical"
        if pd.api.types.is_bool_dtype(series):
            inferred_type = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            if unique_count <= 2:
                inferred_type = "boolean"
            elif unique_count <= 10 and not pd.api.types.is_float_dtype(series):
                inferred_type = "categorical"
            else:
                inferred_type = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred_type = "datetime"
        else:
            # Object or string
            # Check if parseable as datetime
            try:
                pd.to_datetime(series.dropna().head(20))
                inferred_type = "datetime"
            except (ValueError, TypeError):
                if unique_count > (n_rows * 0.8) and unique_count > 50:
                    inferred_type = "high_cardinality_text"
                else:
                    inferred_type = "categorical"

        # Check for constant column
        if unique_count <= 1 and missing_count == 0:
            inferred_type = "constant"

        sample_vals = series.dropna().head(5).tolist()
        # Convert non-serializable objects
        sample_vals = [val.isoformat() if hasattr(val, "isoformat") else (float(val) if isinstance(val, (np.floating, float)) else (int(val) if isinstance(val, (np.integer, int)) else str(val))) for val in sample_vals]

        columns_info[col] = {
            "name": col,
            "dtype": dtype_str,
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "sample_values": sample_vals
        }

        # Check if potential target
        # Targets typically have reasonable cardinality and are not pure IDs
        col_lower = col.lower()
        is_id_col = "id" in col_lower or "guid" in col_lower or "uuid" in col_lower
        if not is_id_col and unique_count > 1 and unique_count < (n_rows * 0.95):
            potential_targets.append(col)

    # Descriptive summary statistics
    summary_stats = {}
    for col, info in columns_info.items():
        series = df[col].dropna()
        if series.empty:
            continue
        if info["inferred_type"] in ["numerical"]:
            summary_stats[col] = {
                "type": "numerical",
                "count": int(len(series)),
                "mean": float(round(series.mean(), 4)),
                "std": float(round(series.std(), 4)) if len(series) > 1 else 0.0,
                "min": float(round(series.min(), 4)),
                "q25": float(round(series.quantile(0.25), 4)),
                "median": float(round(series.median(), 4)),
                "q75": float(round(series.quantile(0.75), 4)),
                "max": float(round(series.max(), 4))
            }
        else:
            value_counts = series.value_counts(dropna=True).head(10).to_dict()
            summary_stats[col] = {
                "type": "categorical",
                "count": int(len(series)),
                "unique": int(info["unique_count"]),
                "top": str(series.mode()[0]) if not series.empty else None,
                "top_freq": int(series.value_counts().iloc[0]) if not series.empty else 0,
                "frequency_distribution": {str(k): int(v) for k, v in value_counts.items()}
            }

    # Priority keywords for target: is_fraud, fraud, Class, churn, label, target, outcome, status, class, price, salary, y
    target_keywords = ["is_fraud", "fraud", "fraud_label", "fraud_flag", "is_fraudulent", "class", "churn", "target", "label", "outcome", "status", "is_churn", "price", "sale_price", "y"]
    detected_target = None
    for kw in target_keywords:
        for pt in potential_targets:
            if kw == pt.lower() or pt.lower().endswith(f"_{kw}"):
                detected_target = pt
                break
        if detected_target:
            break

    if not detected_target and potential_targets:
        # Fall back to the last non-id potential target
        detected_target = potential_targets[-1]

    detected_task = None
    if detected_target:
        detected_task = detect_task_type(df[detected_target])

    return columns_info, summary_stats, potential_targets, detected_target, detected_task


def detect_task_type(target_series: pd.Series) -> str:
    """
    AutoFlow AI is specifically an integrated platform for fraud detection,
    so all modeling workflows are classification tasks (fraud vs legitimate).
    """
    return "classification"


def build_preprocessor(
    X: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer:
    - Numerical: SimpleImputer(median) + StandardScaler
    - Categorical: SimpleImputer(most_frequent) + OneHotEncoder(ignore unknown)
    """
    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def extract_reference_distribution(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Extracts statistical baseline reference from training data for future drift monitoring.
    """
    reference = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            # Save summary metrics and a subsample of raw values for KS-test
            subsample = series.sample(min(len(series), 500), random_state=42).tolist()
            reference[col] = {
                "type": "numerical",
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "sample_values": subsample
            }
        else:
            # Frequency counts for Chi-Square test
            val_counts = series.astype(str).value_counts(normalize=True).to_dict()
            mode_val = str(series.mode()[0]) if not series.empty else ""
            reference[col] = {
                "type": "categorical",
                "count": int(len(series)),
                "mode": mode_val,
                "frequencies": {str(k): float(v) for k, v in val_counts.items()}
            }
    return reference
