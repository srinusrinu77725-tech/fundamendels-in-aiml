"""
report_generator.py
---------------------
FUNCTIONAL MODULE 3 (part B): REPORTING & ANALYTICS

Turns raw prediction history and model metrics into human-readable
summaries and charts:

    - Text summary of stored patient records (counts per disease/result).
    - Bar chart of positive vs negative predictions per module.
    - Model performance chart (accuracy/precision/recall/F1) read from
      the metrics JSON files written by model_trainer.py.

Charts are saved as PNG files under /reports so they can be embedded in
the project report or viewed directly.
"""

import os
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from src.patient_records import read_all_records
from src.logger_config import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def summarize_history() -> dict:
    """Return counts of predictions grouped by disease module and outcome."""
    records = read_all_records()
    by_module = Counter(r["disease_module"] for r in records)
    by_outcome = Counter((r["disease_module"], r["prediction"]) for r in records)

    summary = {
        "total_predictions": len(records),
        "by_module": dict(by_module),
        "by_outcome": {f"{m} - {o}": c for (m, o), c in by_outcome.items()},
    }
    return summary


def plot_prediction_distribution(save_path: str = None) -> str:
    """Bar chart: number of predictions made per disease module."""
    records = read_all_records()
    if not records:
        logger.warning("No records available to plot prediction distribution.")
        return None

    counts = Counter(r["disease_module"] for r in records)
    save_path = save_path or os.path.join(REPORTS_DIR, "prediction_distribution.png")

    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values(), color="#3b82c4")
    plt.title("Predictions Made per Module")
    plt.ylabel("Number of Predictions")
    plt.xlabel("Disease Module")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved prediction distribution chart -> %s", save_path)
    return save_path


def plot_model_performance(save_path: str = None) -> str:
    """Bar chart comparing accuracy/precision/recall/F1 across trained models."""
    metric_files = {
        "Diabetes": os.path.join(MODELS_DIR, "diabetes_metrics.json"),
        "Heart Disease": os.path.join(MODELS_DIR, "heart_metrics.json"),
        "Symptom Checker": os.path.join(MODELS_DIR, "symptom_metrics.json"),
    }

    available = {name: path for name, path in metric_files.items() if os.path.exists(path)}
    if not available:
        logger.warning("No trained model metrics found; train models first.")
        return None

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    data = {}
    for name, path in available.items():
        with open(path) as f:
            m = json.load(f)
        data[name] = [m[k] for k in metrics_to_plot]

    save_path = save_path or os.path.join(REPORTS_DIR, "model_performance.png")

    x = range(len(metrics_to_plot))
    width = 0.8 / max(len(data), 1)
    plt.figure(figsize=(7, 4.5))
    for i, (name, values) in enumerate(data.items()):
        offset = [xi + i * width for xi in x]
        plt.bar(offset, values, width=width, label=name)

    plt.xticks([xi + width * (len(data) - 1) / 2 for xi in x], metrics_to_plot)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved model performance chart -> %s", save_path)
    return save_path


def print_text_report():
    """Print a plain-text analytics summary to the console."""
    summary = summarize_history()
    print("\n=== PREDICTION HISTORY SUMMARY ===")
    print(f"Total predictions made : {summary['total_predictions']}")
    print("\nBy module:")
    for module, count in summary["by_module"].items():
        print(f"  - {module}: {count}")
    print("\nBy outcome:")
    for outcome, count in summary["by_outcome"].items():
        print(f"  - {outcome}: {count}")
    print("===================================\n")
