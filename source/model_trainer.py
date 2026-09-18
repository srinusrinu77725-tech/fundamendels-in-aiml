"""
model_trainer.py
------------------
FUNCTIONAL MODULE 2 (part A): MODEL TRAINING

Trains, evaluates, and persists machine-learning models for each disease
supported by the system:

    - Diabetes            -> RandomForestClassifier
    - Heart Disease       -> RandomForestClassifier
    - Symptom -> Illness  -> DecisionTreeClassifier

Trained artefacts (model + scaler/encoder + evaluation metrics) are saved
under /models so the predictor module can load them without retraining.
"""

import os
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from src.data_loader import (
    load_diabetes_data, load_heart_data, load_symptom_data, prepare_supervised_split
)
from src.logger_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _evaluate(model, X_test, y_test, average="binary") -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def _save_artifact(name: str, model, scaler, label_encoder, feature_names, metrics):
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, f"{name}_model.pkl"))
    with open(os.path.join(MODELS_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved model artefact '%s' -> %s", name, MODELS_DIR)


def train_diabetes_model() -> dict:
    df = load_diabetes_data()
    split = prepare_supervised_split(df, target_col="Outcome", scale=True)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(split["X_train"], split["y_train"])
    metrics = _evaluate(model, split["X_test"], split["y_test"])

    _save_artifact("diabetes", model, split["scaler"], split["label_encoder"],
                    split["feature_names"], metrics)
    logger.info("Diabetes model trained. Accuracy=%.3f", metrics["accuracy"])
    return metrics


def train_heart_model() -> dict:
    df = load_heart_data()
    split = prepare_supervised_split(df, target_col="target", scale=True)

    model = RandomForestClassifier(
        n_estimators=250, max_depth=7, random_state=42, class_weight="balanced"
    )
    model.fit(split["X_train"], split["y_train"])
    metrics = _evaluate(model, split["X_test"], split["y_test"])

    _save_artifact("heart", model, split["scaler"], split["label_encoder"],
                    split["feature_names"], metrics)
    logger.info("Heart disease model trained. Accuracy=%.3f", metrics["accuracy"])
    return metrics


def train_symptom_model() -> dict:
    df = load_symptom_data()
    split = prepare_supervised_split(df, target_col="disease", scale=False, encode_target=True)

    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(split["X_train"], split["y_train"])
    metrics = _evaluate(model, split["X_test"], split["y_test"], average="macro")

    _save_artifact("symptom", model, split["scaler"], split["label_encoder"],
                    split["feature_names"], metrics)
    logger.info("Symptom-checker model trained. Accuracy=%.3f", metrics["accuracy"])
    return metrics


def train_all_models() -> dict:
    """Train and persist all three models; return a summary of metrics."""
    logger.info("Starting training run for all disease models ...")
    results = {
        "diabetes": train_diabetes_model(),
        "heart": train_heart_model(),
        "symptom": train_symptom_model(),
    }
    logger.info("Training run complete.")
    return results


if __name__ == "__main__":
    summary = train_all_models()
    for disease, metrics in summary.items():
        print(f"\n{disease.upper()} MODEL")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v}")
