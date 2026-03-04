"""
run_experiments.py - Run 5+ experiments with varying hyperparameters
=====================================================================
Systematically explores hyperparameter space and logs all runs to MLflow.
Produces a comparison table for the lineage report.

Usage:
    python run_experiments.py
"""

import json
import os
import sys
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from train import train_and_log, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME


# ---------------------------------------------------------------------------
# Experiment configurations - systematic hyperparameter exploration
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS = [
    {
        "run_name": "baseline_run",
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 2,
        "learning_rate": 0.01,
        "description": "Baseline: small forest, shallow trees",
    },
    {
        "run_name": "deeper_trees",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 2,
        "learning_rate": 0.01,
        "description": "Deeper trees to capture complex interactions",
    },
    {
        "run_name": "large_forest",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "learning_rate": 0.01,
        "description": "More estimators for variance reduction",
    },
    {
        "run_name": "high_min_split",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "description": "Higher min_samples_split for regularization",
    },
    {
        "run_name": "production_candidate",
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 3,
        "learning_rate": 0.005,
        "description": "Balanced config targeting production deployment",
    },
]


def run_all_experiments():
    """Run all experiment configurations and collect results."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    print("=" * 70)
    print("RUNNING {} EXPERIMENTS".format(len(EXPERIMENT_CONFIGS)))
    print("=" * 70)

    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print("\n--- Experiment {}/{}: {} ---".format(
            i, len(EXPERIMENT_CONFIGS), config["run_name"]
        ))
        print("  {}".format(config["description"]))

        run_id, metrics = train_and_log(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            learning_rate=config["learning_rate"],
            run_name=config["run_name"],
        )

        results.append({
            "run_name": config["run_name"],
            "run_id": run_id,
            "description": config["description"],
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_split": config["min_samples_split"],
            "learning_rate": config["learning_rate"],
            **metrics,
        })

    return results


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 90)

    header = "{:<22} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}".format(
        "Run Name", "Trees", "Depth", "Split", "Acc", "F1", "AUC", "Prec",
    )
    print(header)
    print("-" * 90)

    best_acc = max(r["accuracy"] for r in results)

    for r in results:
        marker = " *" if r["accuracy"] == best_acc else ""
        row = "{:<22} {:>6} {:>6} {:>6} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}{}".format(
            r["run_name"],
            r["n_estimators"],
            r["max_depth"],
            r["min_samples_split"],
            r["accuracy"],
            r["f1_score"],
            r["auc"],
            r["precision"],
            marker,
        )
        print(row)

    print("-" * 90)
    print("* = Best accuracy\n")


def register_best_model(results):
    """Register the best-performing model to the MLflow Model Registry."""
    client = MlflowClient()

    # Select production candidate (best accuracy)
    best = max(results, key=lambda r: r["accuracy"])
    print("PRODUCTION CANDIDATE: {} (accuracy={:.4f})".format(
        best["run_name"], best["accuracy"]
    ))

    # Register model
    model_uri = "runs:/{}/model".format(best["run_id"])
    try:
        result = mlflow.register_model(model_uri, MODEL_NAME)
        version = result.version

        # Add metadata
        client.update_model_version(
            name=MODEL_NAME,
            version=version,
            description="Production candidate: {} | Accuracy: {:.4f} | F1: {:.4f}".format(
                best["run_name"], best["accuracy"], best["f1_score"]
            ),
        )
        client.set_model_version_tag(MODEL_NAME, version, "accuracy", str(best["accuracy"]))
        client.set_model_version_tag(MODEL_NAME, version, "run_name", best["run_name"])
        client.set_model_version_tag(MODEL_NAME, version, "candidate", "true")

        # Stage transitions: None -> Staging -> Production
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage="Staging",
        )
        print("  -> Transitioned to Staging (v{})".format(version))

        client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage="Production",
        )
        print("  -> Promoted to Production (v{})".format(version))

    except Exception as e:
        print("WARNING: Model registration failed: {}".format(e))
        print("  (This is expected if MLflow server is not running)")


def main():
    results = run_all_experiments()
    print_comparison_table(results)

    # Save results for lineage report
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to experiment_results.json")

    # Register best model
    register_best_model(results)

    # Save best run metrics for CI validation
    best = max(results, key=lambda r: r["accuracy"])
    with open("latest_metrics.json", "w") as f:
        json.dump({"run_id": best["run_id"], **{
            k: v for k, v in best.items()
            if k in ("accuracy", "f1_score", "precision", "recall", "auc")
        }}, f, indent=2)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
