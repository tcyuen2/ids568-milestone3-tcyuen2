"""
Airflow DAG: train_pipeline
============================
Orchestrates the ML workflow: preprocess → train → register.

Idempotency guarantees:
- Each run uses a unique run_id (logical_date) to version outputs.
- Preprocessed data is written to a timestamped path; re-runs overwrite the same path.
- MLflow runs are tagged with the Airflow run_id so duplicates are identifiable.
- Model registration checks for existing versions before creating new ones.

Failure handling:
- Tasks retry up to 2 times with a 5-minute delay.
- on_failure_callback sends alert notifications and logs failure context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "train_pipeline_experiment")
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/ml_pipeline/data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/tmp/ml_pipeline/artifacts"))
MODEL_NAME = os.getenv("MODEL_NAME", "sklearn_classifier")
RANDOM_SEED = 42

logger = logging.getLogger("train_pipeline")

# ---------------------------------------------------------------------------
# Failure callback
# ---------------------------------------------------------------------------

def on_failure_callback(context):
    """
    Called when any task in the DAG fails.
    Logs detailed failure information and could be extended to send
    Slack/email/PagerDuty alerts.
    """
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    dag_run = context.get("dag_run")

    error_msg = (
        f"[ALERT] Task FAILED\n"
        f"  DAG:        {task_instance.dag_id}\n"
        f"  Task:       {task_instance.task_id}\n"
        f"  Run ID:     {dag_run.run_id}\n"
        f"  Logical Date: {dag_run.logical_date}\n"
        f"  Exception:  {exception}\n"
    )
    logger.error(error_msg)

    # Persist failure record for auditing
    failure_log = ARTIFACT_DIR / "failure_log.json"
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "dag_id": task_instance.dag_id,
        "task_id": task_instance.task_id,
        "run_id": dag_run.run_id,
        "logical_date": str(dag_run.logical_date),
        "exception": str(exception),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(failure_log, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def preprocess_data(**context):
    """
    Data cleaning and feature engineering.

    Idempotency: Outputs are written to a path keyed by the Airflow logical
    date, so re-running the same DAG run overwrites identical files.
    """
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logical_date = context["logical_date"].strftime("%Y%m%dT%H%M%S")
    output_dir = DATA_DIR / logical_date
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preprocessing data (run=%s)...", logical_date)

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Feature engineering: add polynomial features
    X_poly = np.c_[X, X[:, 0] * X[:, 1], X[:, 2] * X[:, 3]]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts (idempotent: same logical_date → same path)
    artifacts = {
        "X_train.pkl": X_train_scaled,
        "X_test.pkl": X_test_scaled,
        "y_train.pkl": y_train,
        "y_test.pkl": y_test,
        "scaler.pkl": scaler,
    }
    for name, obj in artifacts.items():
        with open(output_dir / name, "wb") as f:
            pickle.dump(obj, f)

    # Compute and log data hash for lineage
    data_hash = hashlib.sha256(
        X_train_scaled.tobytes() + y_train.tobytes()
    ).hexdigest()[:12]

    logger.info(
        "Preprocessing complete. Samples: train=%d test=%d  data_hash=%s",
        len(X_train_scaled), len(X_test_scaled), data_hash,
    )

    # Push metadata to XCom for downstream tasks
    context["ti"].xcom_push(key="data_dir", value=str(output_dir))
    context["ti"].xcom_push(key="data_hash", value=data_hash)
    context["ti"].xcom_push(key="n_features", value=X_train_scaled.shape[1])
    context["ti"].xcom_push(key="n_train_samples", value=len(X_train_scaled))


def train_model(**context):
    """
    Model training with hyperparameter logging to MLflow.

    Idempotency: Each Airflow run_id maps to a unique MLflow run.
    Re-running the same DAG run will overwrite metrics for the same tag.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    ti = context["ti"]
    data_dir = Path(ti.xcom_pull(task_ids="preprocess_data", key="data_dir"))
    data_hash = ti.xcom_pull(task_ids="preprocess_data", key="data_hash")
    logical_date = context["logical_date"].strftime("%Y%m%dT%H%M%S")

    # Load preprocessed data
    X_train = pickle.load(open(data_dir / "X_train.pkl", "rb"))
    X_test = pickle.load(open(data_dir / "X_test.pkl", "rb"))
    y_train = pickle.load(open(data_dir / "y_train.pkl", "rb"))
    y_test = pickle.load(open(data_dir / "y_test.pkl", "rb"))

    # Hyperparameters (can be overridden via Airflow Variables or DAG params)
    params = context["dag_run"].conf if context["dag_run"].conf else {}
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 10)
    min_samples_split = params.get("min_samples_split", 2)
    learning_rate = params.get("learning_rate", 0.1)  # logged for reference

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"airflow_{logical_date}") as run:
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "learning_rate": learning_rate,
            "random_seed": RANDOM_SEED,
            "data_hash": data_hash,
            "data_version": logical_date,
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "model_type": "RandomForestClassifier",
        })

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        })

        # Save and hash model artifact
        model_path = ARTIFACT_DIR / logical_date / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        model_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
        mlflow.set_tag("model_hash", model_hash)
        mlflow.set_tag("airflow_run_id", context["run_id"])
        mlflow.set_tag("pipeline_version", "1.0.0")

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(model_path))

        logger.info(
            "Training complete. accuracy=%.4f  f1=%.4f  model_hash=%s",
            accuracy, f1, model_hash[:12],
        )

        # Push to XCom
        ti.xcom_push(key="mlflow_run_id", value=run.info.run_id)
        ti.xcom_push(key="accuracy", value=accuracy)
        ti.xcom_push(key="f1_score", value=f1)
        ti.xcom_push(key="model_hash", value=model_hash)
        ti.xcom_push(key="model_path", value=str(model_path))


def register_model(**context):
    """
    Register the trained model to the MLflow Model Registry.

    Idempotency: Checks existing versions before registration.
    New versions are only created if the model hash differs.
    """
    ti = context["ti"]
    mlflow_run_id = ti.xcom_pull(task_ids="train_model", key="mlflow_run_id")
    accuracy = ti.xcom_pull(task_ids="train_model", key="accuracy")
    model_hash = ti.xcom_pull(task_ids="train_model", key="model_hash")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Check if model already registered with same hash (idempotency)
    try:
        latest_versions = client.get_latest_versions(MODEL_NAME)
        for v in latest_versions:
            run = client.get_run(v.run_id)
            existing_hash = run.data.tags.get("model_hash", "")
            if existing_hash == model_hash:
                logger.info(
                    "Model with hash %s already registered (version %s). Skipping.",
                    model_hash[:12], v.version,
                )
                ti.xcom_push(key="registered_version", value=v.version)
                ti.xcom_push(key="registration_status", value="skipped_duplicate")
                return
    except mlflow.exceptions.MlflowException:
        logger.info("No existing model '%s' found. Creating new entry.", MODEL_NAME)

    # Register new version
    model_uri = f"runs:/{mlflow_run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version

    # Add version description and tags
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"Accuracy: {accuracy:.4f} | Hash: {model_hash[:12]} | "
                    f"Registered by Airflow DAG at {datetime.utcnow().isoformat()}",
    )
    client.set_model_version_tag(MODEL_NAME, version, "accuracy", str(accuracy))
    client.set_model_version_tag(MODEL_NAME, version, "model_hash", model_hash)
    client.set_model_version_tag(MODEL_NAME, version, "airflow_run_id", context["run_id"])

    # Transition to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )

    logger.info(
        "Model registered: %s v%s (stage=Staging, accuracy=%.4f)",
        MODEL_NAME, version, accuracy,
    )

    ti.xcom_push(key="registered_version", value=version)
    ti.xcom_push(key="registration_status", value="registered")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="ML pipeline: preprocess → train → register with MLflow tracking",
    schedule=None,  # Trigger manually or via CI
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "mlflow"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    # Define task dependencies
    preprocess_task >> train_task >> register_task
