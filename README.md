# IDS 568 - Milestone 3: Workflow Automation & Experiment Tracking

Automated ML pipeline integrating Apache Airflow orchestration, MLflow experiment tracking, and GitHub Actions CI/CD with quality gates.

## Architecture Overview

The system uses three integrated components:

1. **Airflow DAG** (`dags/train_pipeline.py`): Orchestrates `preprocess_data >> train_model >> register_model` with retry logic and failure callbacks.
2. **CI/CD** (`.github/workflows/train_and_validate.yml`): GitHub Actions workflow that trains a model, logs to MLflow, and enforces quality gates via `model_validation.py`.
3. **MLflow Registry**: Tracks experiments (params, metrics, artifacts, hashes) with model staging (`None -> Staging -> Production`).

The pipeline uses the Iris dataset with engineered interaction features, trained via RandomForestClassifier. Every run logs parameters, metrics, artifacts, and SHA-256 hashes to MLflow for complete lineage.

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (optional, for Airflow with Docker Compose)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Tracking Server

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts
```

Open the MLflow UI at http://localhost:5000.

### 3. Set Up Airflow (Local)

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init

airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin \
  --email admin@example.com

mkdir -p $AIRFLOW_HOME/dags
cp dags/train_pipeline.py $AIRFLOW_HOME/dags/

airflow scheduler &
airflow webserver --port 8080 &
```

Open the Airflow UI at http://localhost:8080.

### 4. Run the Full Pipeline

**Option A: Via Airflow UI** - Enable `train_pipeline` DAG, click "Trigger DAG".

**Option B: Standalone Training**
```bash
python train.py --n_estimators 100 --max_depth 10 --learning_rate 0.01
```

**Option C: Run All 5 Experiments**
```bash
python run_experiments.py
```

## DAG Idempotency and Lineage Guarantees

**Idempotency** is enforced at every task level:

- **preprocess_data**: Outputs are written to a directory keyed by `logical_date`. Re-running the same DAG run overwrites the identical path.
- **train_model**: Each run is tagged with the Airflow `run_id`. MLflow deduplicates by checking run names and model hashes.
- **register_model**: Before creating a new registry version, the task checks whether a model with the same SHA-256 hash already exists. If so, it skips registration.

**Lineage** is guaranteed through:

- `data_hash`: SHA-256 of training data, logged as an MLflow parameter and tag.
- `model_hash`: SHA-256 of the serialized model artifact, logged to MLflow tags.
- `airflow_run_id`: Links every MLflow run back to the exact Airflow execution.
- `pipeline_version`: Semantic version tag for the pipeline code itself.

## CI-Based Model Governance

The GitHub Actions workflow enforces governance through automated quality gates:

1. **Environment parity**: CI installs exact pinned versions from `requirements.txt`.
2. **Automated training**: Model is trained with consistent hyperparameters on every push.
3. **Quality gates** (`model_validation.py`): Pipeline fails if any metric falls below threshold (accuracy >= 0.90, F1 >= 0.85, AUC >= 0.90).
4. **Artifact preservation**: Training metrics and MLflow data are uploaded as CI artifacts.

If a model fails validation, CI exits with code 1, blocking the PR from merging.

## Experiment Tracking Methodology

Experiments follow a systematic hyperparameter exploration strategy:

| Run | n_estimators | max_depth | min_samples_split | Rationale |
|-----|-------------|-----------|-------------------|-----------|
| baseline_run | 50 | 5 | 2 | Establish performance floor |
| deeper_trees | 100 | 15 | 2 | Test if deeper trees improve accuracy |
| large_forest | 200 | 10 | 2 | Reduce variance with more estimators |
| high_min_split | 100 | 10 | 5 | Regularize to prevent overfitting |
| production_candidate | 150 | 12 | 3 | Balanced config for deployment |

Every run logs: all hyperparameters, evaluation metrics (accuracy, F1, precision, recall, AUC), model artifacts, scaler artifacts, data hashes, and model hashes.

## Retry Strategies and Failure Handling

**Retry configuration** (in `default_args`):
- `retries: 2` - each task retries up to 2 times on failure.
- `retry_delay: 5 minutes` - waits between retries for transient issues.

**Failure callback** (`on_failure_callback`):
- Logs structured failure information (DAG ID, task ID, run ID, exception, timestamp).
- Writes a failure record to `failure_log.json` for auditing.
- Extensible for Slack/email/PagerDuty notifications.

## Monitoring and Alerting Recommendations

- **MLflow UI**: Review experiment runs, compare metrics, inspect artifacts.
- **Airflow UI**: Monitor DAG status, task durations, retry counts.
- **Metrics to track**: accuracy drift, training duration trends, failure rates.
- **Alerting**: Extend `on_failure_callback` for Slack webhooks or PagerDuty.

## Rollback Procedures

If a newly promoted model performs poorly:

```bash
# 1. Identify previous production version
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.get_latest_versions('sklearn_classifier', stages=['Production'])

# 2. Demote bad version back to Staging
client.transition_model_version_stage(name='sklearn_classifier', version=BAD_VERSION, stage='Staging')

# 3. Re-promote the previous good version
client.transition_model_version_stage(name='sklearn_classifier', version=GOOD_VERSION, stage='Production')
```

Since tasks are idempotent and outputs are versioned, you can re-trigger any previous DAG run safely.

## Repository Structure

```
ids568-milestone3-[netid]/
├── .github/workflows/
│   └── train_and_validate.yml   # CI/CD pipeline
├── dags/
│   └── train_pipeline.py        # Airflow DAG
├── train.py                     # Standalone training script
├── model_validation.py          # Quality gate validation
├── run_experiments.py           # Run 5+ experiments
├── requirements.txt             # Pinned dependencies
├── README.md                    # This file
└── lineage_report.md            # Experiment analysis report
```
