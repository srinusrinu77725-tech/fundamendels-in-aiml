"""
data_loader.py
----------------
FUNCTIONAL MODULE 1: DATA MANAGEMENT & PREPROCESSING

Responsibilities:
    - Load raw CSV datasets (generating them first if absent).
    - Clean and preprocess data (handle missing/zero-as-missing values,
      scale numeric features, encode categorical labels).
    - Provide a consistent train/test split for the model_trainer module.

This module isolates all data-handling logic so prediction and reporting
code never has to know where the data came from or how it was cleaned.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.logger_config import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DIABETES_CSV = os.path.join(DATA_DIR, "diabetes.csv")
HEART_CSV = os.path.join(DATA_DIR, "heart.csv")
SYMPTOMS_CSV = os.path.join(DATA_DIR, "symptoms.csv")

# Columns whose value of 0 is physiologically implausible and therefore
# treated as a missing reading in the diabetes dataset.
DIABETES_ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _ensure_datasets_exist():
    """Generate the synthetic datasets on first run if they are missing."""
    if all(os.path.exists(p) for p in (DIABETES_CSV, HEART_CSV, SYMPTOMS_CSV)):
        return
    logger.info("Datasets not found; generating synthetic datasets ...")
    script = os.path.join(DATA_DIR, "generate_datasets.py")
    subprocess.run([sys.executable, script], check=True)


def load_diabetes_data() -> pd.DataFrame:
    """Load and clean the diabetes dataset."""
    _ensure_datasets_exist()
    df = pd.read_csv(DIABETES_CSV)
    df[DIABETES_ZERO_AS_NAN_COLS] = df[DIABETES_ZERO_AS_NAN_COLS].replace(0, np.nan)
    df[DIABETES_ZERO_AS_NAN_COLS] = df[DIABETES_ZERO_AS_NAN_COLS].fillna(
        df[DIABETES_ZERO_AS_NAN_COLS].median()
    )
    logger.info("Loaded diabetes dataset: %d rows, %d columns", *df.shape)
    return df


def load_heart_data() -> pd.DataFrame:
    """Load and clean the heart disease dataset."""
    _ensure_datasets_exist()
    df = pd.read_csv(HEART_CSV)
    df = df.dropna()
    logger.info("Loaded heart disease dataset: %d rows, %d columns", *df.shape)
    return df


def load_symptom_data() -> pd.DataFrame:
    """Load the symptom-checklist dataset."""
    _ensure_datasets_exist()
    df = pd.read_csv(SYMPTOMS_CSV)
    logger.info("Loaded symptom dataset: %d rows, %d columns", *df.shape)
    return df


def prepare_supervised_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
                              random_state: int = 42, scale: bool = True, encode_target: bool = False):
    """
    Generic helper: splits a dataframe into scaled train/test feature and
    label arrays. Returns a dict bundling everything a trainer needs.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoder = None
    if encode_target:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": list(X.columns),
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
