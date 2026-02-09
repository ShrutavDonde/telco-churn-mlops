# src/train.py
from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .config import MLRUNS_DIR
from .data import build_dataset
from .modeling import make_pipeline, TARGET_COL, ID_COL

EXPERIMENT_NAME = "telco-churn-xgb"

def main() -> None:
    df = build_dataset()

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = make_pipeline(X_train)

    # Local MLflow tracking (no server needed)
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("roc_auc", float(auc))

        # Log the whole pipeline (preprocess + model)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=None,  # keep local/simple for now
        )

        print(f"ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
