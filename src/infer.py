# src/infer.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from .config import MLRUNS_DIR
from .train import EXPERIMENT_NAME

def load_latest_sklearn_model() -> Tuple[Any, str]:
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Train first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow runs found. Train first.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model, run_id

def predict_proba_one(features: Dict[str, Any]) -> Dict[str, Any]:
    model, run_id = load_latest_sklearn_model()
    X = pd.DataFrame([features])
    p = float(model.predict_proba(X)[:, 1][0])
    return {"churn_probability": p, "run_id": run_id}
