"""
model_validation.py - Threshold-based model acceptance logic
=============================================================
Used by CI/CD pipeline to enforce quality gates before model promotion.

Exit codes:
    0 = Model passes all quality gates
    1 = Model fails one or more quality gates

Usage:
    python model_validation.py --min-accuracy 0.90 --min-f1 0.85 --min-auc 0.90
    python model_validation.py --metrics-file latest_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "accuracy": 0.90,
    "f1_score": 0.85,
    "precision": 0.80,
    "recall": 0.80,
    "auc": 0.90,
}


def load_metrics(metrics_file):
    """Load metrics from a JSON file produced by train.py."""
    path = Path(metrics_file)
    if not path.exists():
        print("FATAL: Metrics file not found: {}".format(metrics_file))
        sys.exit(1)

    with open(path) as f:
        metrics = json.load(f)

    return metrics


def validate_metrics(metrics, thresholds):
    """
    Validate metrics against thresholds.
    Returns (passed, messages).
    """
    passed = True
    messages = []

    for metric_name, min_value in thresholds.items():
        actual = metrics.get(metric_name)

        if actual is None:
            messages.append(
                "  WARNING: Metric '{}' not found in results".format(metric_name)
            )
            continue

        if actual >= min_value:
            messages.append(
                "  PASS: {} = {:.4f} >= {:.4f}".format(metric_name, actual, min_value)
            )
        else:
            messages.append(
                "  FAIL: {} = {:.4f} < {:.4f}".format(metric_name, actual, min_value)
            )
            passed = False

    return passed, messages


def main():
    parser = argparse.ArgumentParser(
        description="Validate model metrics against quality thresholds"
    )
    parser.add_argument(
        "--metrics-file", type=str, default="latest_metrics.json",
        help="Path to JSON file with model metrics",
    )
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-f1", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-auc", type=float, default=None)
    args = parser.parse_args()

    # Build thresholds from defaults + CLI overrides
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.min_accuracy is not None:
        thresholds["accuracy"] = args.min_accuracy
    if args.min_f1 is not None:
        thresholds["f1_score"] = args.min_f1
    if args.min_precision is not None:
        thresholds["precision"] = args.min_precision
    if args.min_recall is not None:
        thresholds["recall"] = args.min_recall
    if args.min_auc is not None:
        thresholds["auc"] = args.min_auc

    # Load and validate
    print("=" * 60)
    print("MODEL VALIDATION - Quality Gate Check")
    print("=" * 60)

    metrics = load_metrics(args.metrics_file)
    run_id = metrics.get("run_id", "unknown")
    print("\nRun ID: {}".format(run_id))
    print("Metrics file: {}\n".format(args.metrics_file))

    print("Thresholds:")
    for name, value in thresholds.items():
        print("  {}: >= {:.4f}".format(name, value))

    print("\nResults:")
    passed, messages = validate_metrics(metrics, thresholds)
    for msg in messages:
        print(msg)

    print()
    if passed:
        print("RESULT: ALL QUALITY GATES PASSED")
        print("Model is approved for promotion to Staging.")
        sys.exit(0)
    else:
        print("RESULT: QUALITY GATE FAILED")
        print("Model does NOT meet minimum thresholds.")
        print("Pipeline will fail to prevent deployment of substandard model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
