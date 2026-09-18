"""
predictor.py
-------------
FUNCTIONAL MODULE 2 (part B): PREDICTION ENGINE

Loads the persisted model artefacts produced by model_trainer.py and
exposes a single, simple `predict(disease, features)` entry point used by
the CLI (and reusable by any future web/API front end).

Validates all incoming feature values before they ever reach a model, and
raises a clear, actionable ValidationError if something is out of range.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.logger_config import get_logger
from src.validators import validate_range, validate_binary, ValidationError

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Field -> (min, max) plausible clinical range, used for input validation.
DIABETES_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 260),
    "BloodPressure": (20, 140),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (10, 70),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (1, 120),
}

HEART_RANGES = {
    "age": (1, 120),
    "trestbps": (70, 220),
    "chol": (100, 650),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
}
HEART_BINARY_FIELDS = ["sex", "fbs", "exang"]
HEART_CHOICE_FIELDS = {
    "cp": [0, 1, 2, 3],
    "restecg": [0, 1, 2],
    "slope": [0, 1, 2],
    "ca": [0, 1, 2, 3, 4],
    "thal": [0, 1, 2, 3],
}

SYMPTOM_FIELDS = [
    "fever", "cough", "fatigue", "headache", "sore_throat",
    "shortness_of_breath", "body_ache", "runny_nose",
    "loss_of_taste_smell", "nausea", "diarrhea", "chest_pain",
]


class ModelNotTrainedError(Exception):
    """Raised when a prediction is requested before the model has been trained."""
    pass


def _load_bundle(name: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    if not os.path.exists(path):
        raise ModelNotTrainedError(
            f"No trained model found for '{name}'. Please train the models first "
            f"(CLI option: 'Train / Retrain Models')."
        )
    return joblib.load(path)


def validate_diabetes_input(data: dict) -> dict:
    cleaned = {}
    for field, (low, high) in DIABETES_RANGES.items():
        cleaned[field] = validate_range(data.get(field), field, low, high)
    return cleaned


def validate_heart_input(data: dict) -> dict:
    cleaned = {}
    for field, (low, high) in HEART_RANGES.items():
        cleaned[field] = validate_range(data.get(field), field, low, high)
    for field in HEART_BINARY_FIELDS:
        cleaned[field] = validate_binary(data.get(field), field)
    for field, choices in HEART_CHOICE_FIELDS.items():
        val = data.get(field)
        try:
            val = int(val)
        except (TypeError, ValueError):
            raise ValidationError(f"'{field}' must be an integer in {choices} (got {val!r}).")
        if val not in choices:
            raise ValidationError(f"'{field}' must be one of {choices} (got {val!r}).")
        cleaned[field] = val
    return cleaned


def validate_symptom_input(data: dict) -> dict:
    cleaned = {}
    for field in SYMPTOM_FIELDS:
        cleaned[field] = validate_binary(data.get(field, 0), field)
    return cleaned


def predict_diabetes(raw_input: dict) -> dict:
    cleaned = validate_diabetes_input(raw_input)
    bundle = _load_bundle("diabetes")
    df = pd.DataFrame([cleaned])[bundle["feature_names"]]
    X = pd.DataFrame(bundle["scaler"].transform(df), columns=bundle["feature_names"])
    pred = int(bundle["model"].predict(X)[0])
    proba = float(bundle["model"].predict_proba(X)[0][pred])
    result = {
        "disease": "Diabetes",
        "prediction": "Positive" if pred == 1 else "Negative",
        "confidence": round(proba * 100, 2),
        "input": cleaned,
    }
    logger.info("Diabetes prediction: %s (%.1f%% confidence)", result["prediction"], result["confidence"])
    return result


def predict_heart_disease(raw_input: dict) -> dict:
    cleaned = validate_heart_input(raw_input)
    bundle = _load_bundle("heart")
    df = pd.DataFrame([cleaned])[bundle["feature_names"]]
    X = pd.DataFrame(bundle["scaler"].transform(df), columns=bundle["feature_names"])
    pred = int(bundle["model"].predict(X)[0])
    proba = float(bundle["model"].predict_proba(X)[0][pred])
    result = {
        "disease": "Heart Disease",
        "prediction": "Positive" if pred == 1 else "Negative",
        "confidence": round(proba * 100, 2),
        "input": cleaned,
    }
    logger.info("Heart disease prediction: %s (%.1f%% confidence)", result["prediction"], result["confidence"])
    return result


def predict_illness_from_symptoms(raw_input: dict) -> dict:
    cleaned = validate_symptom_input(raw_input)
    bundle = _load_bundle("symptom")
    df = pd.DataFrame([cleaned])[bundle["feature_names"]]
    pred_idx = int(bundle["model"].predict(df)[0])
    proba = float(np.max(bundle["model"].predict_proba(df)[0]))
    disease_name = bundle["label_encoder"].inverse_transform([pred_idx])[0]
    result = {
        "disease": "Symptom Checker",
        "prediction": disease_name,
        "confidence": round(proba * 100, 2),
        "input": cleaned,
    }
    logger.info("Symptom-based prediction: %s (%.1f%% confidence)", disease_name, result["confidence"])
    return result
