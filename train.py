"""
train.py - Standalone training script
======================================
Can be run independently or called from the Airflow DAG.
Logs all parameters, metrics, and artifacts to MLflow.

Usage:
    python train.py --n_estimators 100 --max_depth 10 --learning_rate 0.01
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "train_pipeline_experiment")
MODEL_NAME = os.getenv("MODEL_NAME", "sklearn_classifier")
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/ml_pipeline/data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/tmp/ml_pipeline/artifacts"))
RANDOM_SEED = 42


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for artifact lineage."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def preprocess(seed: int = RANDOM_SEED):
    """Load data, engineer features, split, and scale."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Feature engineering: interaction terms
    X_poly = np.c_[X, X[:, 0] * X[:, 1], X[:, 2] * X[:, 3]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=seed, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute data hash for lineage
    data_hash = hashlib.sha256(
        X_train_scaled.tobytes() + y_train.tobytes()
    ).hexdigest()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, data_hash


def train_and_log(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    learning_rate: float = 0.01,
    run_name: str | None = None,
):
    """Train model and log everything to MLflow. Returns (run_id, metrics)."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, scaler, data_hash = preprocess()

    if run_name is None:
        run_name = f"train_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # ----- Log parameters -----
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "learning_rate": learning_rate,
            "random_seed": RANDOM_SEED,
            "data_hash": data_hash,
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "model_type": "RandomForestClassifier",
            "scaler": "StandardScaler",
        })

        # ----- Train -----
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        # ----- Evaluate -----
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }
        mlflow.log_metrics(metrics)

        # ----- Save and hash model artifact -----
        model_dir = ARTIFACT_DIR / run.info.run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        model_hash = compute_file_hash(str(model_path))

        # ----- Tags for lineage -----
        mlflow.set_tag("model_hash", model_hash)
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("pipeline_version", "1.0.0")
        mlflow.set_tag("training_script", "train.py")

        # ----- Log artifacts -----
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(model_path))

        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(str(scaler_path))

        # Save metrics as JSON artifact
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
        print(f"Model hash: {model_hash[:12]}")

        return run.info.run_id, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML model with MLflow tracking")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_id, metrics = train_and_log(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
    )

    # Write metrics to file for CI validation
    with open("latest_metrics.json", "w") as f:
        json.dump({"run_id": run_id, **metrics}, f, indent=2)

    print(f"\nTraining complete. Metrics saved to latest_metrics.json")


if __name__ == "__main__":
    main()
