# Milestone 3 - Workflow Automation & Experiment Tracking

## What This Is

This project sets up an automated ML pipeline using Airflow, MLflow, and GitHub Actions. It trains a RandomForestClassifier on the Iris dataset, tracks experiments, and registers the best model.

## How to Set Up

1. Clone the repo and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start MLflow:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
```

4. Open a new terminal (activate venv again) and run experiments:

```bash
python run_experiments.py
```

This trains 5 models with different hyperparameters and registers the best one to MLflow.

## How to Run the Airflow DAG

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
mkdir -p $AIRFLOW_HOME/dags
cp dags/train_pipeline.py $AIRFLOW_HOME/dags/
airflow scheduler &
airflow webserver --port 8080 &
```

Then go to http://localhost:8080, enable the `train_pipeline` DAG, and trigger it. The DAG runs three tasks in order: `preprocess_data >> train_model >> register_model`.

## How the Pipeline Works

The Airflow DAG has three tasks:

- **preprocess_data** - Loads the Iris dataset, adds interaction features, scales everything with StandardScaler, and saves to disk
- **train_model** - Trains a RandomForest model and logs params/metrics/artifacts to MLflow
- **register_model** - Registers the trained model in the MLflow Model Registry

Tasks pass data between each other using XCom (file paths, hashes, run IDs).

## Idempotency

Each task is safe to re-run:

- Preprocessing writes to a folder named by the run date, so re-runs just overwrite the same files
- Training tags each MLflow run with the Airflow run ID so you can identify duplicates
- Registration checks if a model with the same SHA-256 hash already exists before creating a new version

## CI/CD

The GitHub Actions workflow (`.github/workflows/train_and_validate.yml`) does this on every push:

1. Installs dependencies from requirements.txt
2. Starts an MLflow server
3. Runs `train.py` to train a model
4. Runs `model_validation.py` to check if metrics meet thresholds

If accuracy < 0.90, F1 < 0.85, or AUC < 0.90, the pipeline fails and blocks the PR.

## Experiment Tracking

I ran 5 experiments varying n_estimators, max_depth, and min_samples_split. All runs are logged to MLflow with full parameters, metrics, model artifacts, and SHA-256 hashes for both the data and the model file.

See `lineage_report.md` for the full comparison and analysis.

## Retry and Failure Handling

- Tasks retry up to 2 times with a 5 minute delay between attempts
- The `on_failure_callback` logs the error details (task ID, exception, timestamp) to a JSON file
- Could be extended to send Slack or email alerts

## Monitoring

- Use the MLflow UI to compare runs and check model versions
- Use the Airflow UI to see task status, retries, and logs
- Watch for accuracy dropping below the 0.90 threshold over time

## Rollback

If a bad model gets promoted, you can roll back in MLflow:

1. Demote the bad version back to Staging
2. Re-promote the previous good version to Production

Since everything is versioned, old models are never deleted.

## Files

- `dags/train_pipeline.py` - Airflow DAG
- `train.py` - Training script
- `model_validation.py` - Quality gate checks
- `run_experiments.py` - Runs all 5 experiments
- `.github/workflows/train_and_validate.yml` - CI/CD workflow
- `requirements.txt` - Pinned dependencies
- `lineage_report.md` - Experiment analysis
- `experiment_results.json` - Exported run data
- `latest_metrics.json` - Best model metrics
- `mlartifacts/` - MLflow model artifacts from all runs
