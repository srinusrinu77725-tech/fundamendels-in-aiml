"""
generate_datasets.py
---------------------
Generates three realistic, statistically-grounded synthetic medical datasets
used by the Multi-Disease Prediction System:

    1. diabetes.csv   - Based on the feature schema of the Pima Indians
                         Diabetes dataset (8 clinical features).
    2. heart.csv       - Based on the UCI Heart Disease feature schema
                         (13 clinical features).
    3. symptoms.csv    - A symptom-checklist dataset mapping binary
                         symptom flags to one of five common illnesses.

Datasets are generated (not downloaded) so that the project runs fully
offline and reproducibly (fixed random seed = 42). Feature ranges and
correlations are modelled on publicly documented clinical statistics for
each condition so the data behaves realistically for teaching/demo
purposes.

NOTE: This data is SYNTHETIC and for academic demonstration only. It must
NEVER be used for real clinical decision-making.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_diabetes_dataset(n_samples: int = 768) -> pd.DataFrame:
    """Synthesizes a Pima-Indians-style diabetes dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    outcome = rng.binomial(1, 0.35, n_samples)

    pregnancies = np.clip(rng.poisson(3.5 + outcome * 1.2, n_samples), 0, 17)
    glucose = np.clip(rng.normal(110 + outcome * 40, 26, n_samples), 44, 199)
    blood_pressure = np.clip(rng.normal(70 + outcome * 4, 12, n_samples), 24, 122)
    skin_thickness = np.clip(rng.normal(20 + outcome * 6, 10, n_samples), 0, 99)
    insulin = np.clip(rng.normal(80 + outcome * 60, 90, n_samples), 0, 846)
    bmi = np.clip(rng.normal(30 + outcome * 4, 7, n_samples), 15, 67)
    dpf = np.clip(rng.normal(0.42 + outcome * 0.15, 0.28, n_samples), 0.05, 2.5)
    age = np.clip(rng.normal(31 + outcome * 8, 11, n_samples), 21, 81).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies.astype(int),
        "Glucose": glucose.round(1),
        "BloodPressure": blood_pressure.round(1),
        "SkinThickness": skin_thickness.round(1),
        "Insulin": insulin.round(1),
        "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": dpf.round(3),
        "Age": age,
        "Outcome": outcome,
    })
    return df


def generate_heart_dataset(n_samples: int = 700) -> pd.DataFrame:
    """Synthesizes a UCI-Heart-Disease-style dataset."""
    rng = np.random.default_rng(RANDOM_SEED + 1)

    target = rng.binomial(1, 0.45, n_samples)

    age = np.clip(rng.normal(54 + target * 4, 9, n_samples), 29, 77).astype(int)
    sex = rng.binomial(1, 0.68, n_samples)  # 1 = male, 0 = female
    cp = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.clip(rng.normal(131 + target * 6, 17, n_samples), 94, 200)
    chol = np.clip(rng.normal(246 + target * 15, 51, n_samples), 126, 564)
    fbs = rng.binomial(1, 0.15 + target * 0.05, n_samples)
    restecg = rng.choice([0, 1, 2], size=n_samples, p=[0.5, 0.48, 0.02])
    thalach = np.clip(rng.normal(150 - target * 15, 22, n_samples), 71, 202)
    exang = rng.binomial(1, 0.15 + target * 0.35, n_samples)
    oldpeak = np.clip(rng.exponential(0.8 + target * 0.8, n_samples), 0, 6.2)
    slope = rng.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47])
    ca = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.58, 0.22, 0.12, 0.06, 0.02])
    thal = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.02, 0.06, 0.55, 0.37])

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps.round(1),
        "chol": chol.round(1),
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach.round(1),
        "exang": exang,
        "oldpeak": oldpeak.round(2),
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target,
    })
    return df


def generate_symptom_dataset(n_samples: int = 900) -> pd.DataFrame:
    """
    Synthesizes a symptom-checklist dataset. Each row is a binary symptom
    vector and the label is one of five common illnesses. Symptom
    probabilities per illness are set from generally known clinical
    presentations (for teaching purposes only).
    """
    rng = np.random.default_rng(RANDOM_SEED + 2)

    symptoms = [
        "fever", "cough", "fatigue", "headache", "sore_throat",
        "shortness_of_breath", "body_ache", "runny_nose",
        "loss_of_taste_smell", "nausea", "diarrhea", "chest_pain",
    ]

    diseases = ["Common Cold", "Influenza (Flu)", "COVID-19",
                "Migraine", "Food Poisoning"]

    # Probability of each symptom occurring, given the disease.
    profile = {
        "Common Cold":      [0.3, 0.8, 0.4, 0.3, 0.7, 0.05, 0.2, 0.85, 0.05, 0.05, 0.05, 0.02],
        "Influenza (Flu)":  [0.85, 0.7, 0.9, 0.6, 0.4, 0.2, 0.85, 0.3, 0.1, 0.2, 0.1, 0.05],
        "COVID-19":         [0.75, 0.8, 0.8, 0.5, 0.35, 0.45, 0.55, 0.25, 0.6, 0.15, 0.1, 0.15],
        "Migraine":         [0.05, 0.02, 0.5, 0.95, 0.05, 0.02, 0.15, 0.02, 0.02, 0.4, 0.05, 0.05],
        "Food Poisoning":   [0.4, 0.02, 0.6, 0.3, 0.05, 0.02, 0.35, 0.02, 0.02, 0.85, 0.8, 0.05],
    }

    rows = []
    labels = rng.choice(diseases, size=n_samples,
                         p=[0.28, 0.22, 0.20, 0.15, 0.15])
    for label in labels:
        probs = profile[label]
        row = [rng.binomial(1, p) for p in probs]
        rows.append(row)

    df = pd.DataFrame(rows, columns=symptoms)
    df["disease"] = labels
    return df


def main():
    diabetes_path = os.path.join(OUTPUT_DIR, "diabetes.csv")
    heart_path = os.path.join(OUTPUT_DIR, "heart.csv")
    symptoms_path = os.path.join(OUTPUT_DIR, "symptoms.csv")

    generate_diabetes_dataset().to_csv(diabetes_path, index=False)
    generate_heart_dataset().to_csv(heart_path, index=False)
    generate_symptom_dataset().to_csv(symptoms_path, index=False)

    print(f"[OK] diabetes.csv  -> {diabetes_path}")
    print(f"[OK] heart.csv     -> {heart_path}")
    print(f"[OK] symptoms.csv  -> {symptoms_path}")


if __name__ == "__main__":
    main()
