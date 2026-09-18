"""
test_system.py
----------------
Unit / validation tests for the Multi-Disease Prediction System.

Run with:  pytest -v
"""

import os
import sys
import tempfile
import importlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import validators


# --------------------------------------------------------------------------
# Validator tests
# --------------------------------------------------------------------------

def test_validate_range_accepts_in_bounds_value():
    assert validators.validate_range(50, "Glucose", 40, 260) == 50.0


def test_validate_range_rejects_out_of_bounds_value():
    with pytest.raises(validators.ValidationError):
        validators.validate_range(500, "Glucose", 40, 260)


def test_validate_range_rejects_non_numeric():
    with pytest.raises(validators.ValidationError):
        validators.validate_range("abc", "Glucose", 40, 260)


def test_validate_binary_accepts_0_and_1():
    assert validators.validate_binary(1, "sex") == 1
    assert validators.validate_binary("0", "sex") == 0


def test_validate_binary_rejects_other_values():
    with pytest.raises(validators.ValidationError):
        validators.validate_binary(2, "sex")


def test_validate_choice_accepts_valid_choice():
    assert validators.validate_choice(2, "cp", [0, 1, 2, 3]) == 2


def test_validate_choice_rejects_invalid_choice():
    with pytest.raises(validators.ValidationError):
        validators.validate_choice(9, "cp", [0, 1, 2, 3])


def test_validate_non_empty_string_rejects_blank():
    with pytest.raises(validators.ValidationError):
        validators.validate_non_empty_string("   ", "patient_name")


def test_validate_non_empty_string_trims_whitespace():
    assert validators.validate_non_empty_string("  Alice ", "patient_name") == "Alice"


# --------------------------------------------------------------------------
# Data loader tests
# --------------------------------------------------------------------------

def test_diabetes_data_loads_and_has_expected_columns():
    from src import data_loader
    df = data_loader.load_diabetes_data()
    expected_cols = {
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 100


def test_heart_data_loads_and_has_expected_columns():
    from src import data_loader
    df = data_loader.load_heart_data()
    assert "target" in df.columns
    assert len(df) > 100


# --------------------------------------------------------------------------
# Model training + prediction integration tests
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_models():
    from src.model_trainer import train_all_models
    return train_all_models()


def test_all_models_train_with_reasonable_accuracy(trained_models):
    for disease, metrics in trained_models.items():
        assert metrics["accuracy"] >= 0.60, f"{disease} accuracy too low: {metrics['accuracy']}"


def test_diabetes_prediction_end_to_end(trained_models):
    from src.predictor import predict_diabetes
    sample = {
        "Pregnancies": 2, "Glucose": 190, "BloodPressure": 88,
        "SkinThickness": 32, "Insulin": 250, "BMI": 38.5,
        "DiabetesPedigreeFunction": 0.9, "Age": 55,
    }
    result = predict_diabetes(sample)
    assert result["prediction"] in ("Positive", "Negative")
    assert 0 <= result["confidence"] <= 100


def test_diabetes_prediction_rejects_bad_input(trained_models):
    from src.predictor import predict_diabetes
    from src.validators import ValidationError
    bad_sample = {
        "Pregnancies": 2, "Glucose": 9999, "BloodPressure": 88,
        "SkinThickness": 32, "Insulin": 250, "BMI": 38.5,
        "DiabetesPedigreeFunction": 0.9, "Age": 55,
    }
    with pytest.raises(ValidationError):
        predict_diabetes(bad_sample)


def test_heart_prediction_end_to_end(trained_models):
    from src.predictor import predict_heart_disease
    sample = {
        "age": 61, "sex": 1, "cp": 0, "trestbps": 140, "chol": 260,
        "fbs": 0, "restecg": 1, "thalach": 120, "exang": 1,
        "oldpeak": 2.4, "slope": 1, "ca": 1, "thal": 3,
    }
    result = predict_heart_disease(sample)
    assert result["prediction"] in ("Positive", "Negative")


def test_symptom_prediction_end_to_end(trained_models):
    from src.predictor import predict_illness_from_symptoms, SYMPTOM_FIELDS
    sample = {field: 0 for field in SYMPTOM_FIELDS}
    sample.update({"headache": 1, "nausea": 1, "fatigue": 1})
    result = predict_illness_from_symptoms(sample)
    assert isinstance(result["prediction"], str)
    assert 0 <= result["confidence"] <= 100


# --------------------------------------------------------------------------
# Patient record CRUD tests (isolated temp database)
# --------------------------------------------------------------------------

@pytest.fixture
def temp_records_db(monkeypatch, tmp_path):
    from src import patient_records
    temp_db = tmp_path / "test_records.db"
    monkeypatch.setattr(patient_records, "DB_PATH", str(temp_db))
    patient_records.init_db()
    return patient_records


def test_create_and_read_record(temp_records_db):
    pr = temp_records_db
    rid = pr.create_record("Test Patient", "Diabetes", "Positive", 87.5, {"Glucose": 190})
    records = pr.read_all_records()
    assert any(r["id"] == rid for r in records)


def test_read_records_by_patient_is_case_insensitive(temp_records_db):
    pr = temp_records_db
    pr.create_record("Alice", "Heart Disease", "Negative", 91.0, {"age": 40})
    results = pr.read_records_by_patient("ALICE")
    assert len(results) == 1


def test_update_record_changes_patient_name(temp_records_db):
    pr = temp_records_db
    rid = pr.create_record("Bob", "Diabetes", "Negative", 70.0, {"Glucose": 100})
    ok = pr.update_record(rid, patient_name="Robert")
    assert ok
    records = pr.read_all_records()
    updated = next(r for r in records if r["id"] == rid)
    assert updated["patient_name"] == "Robert"


def test_delete_record_removes_it(temp_records_db):
    pr = temp_records_db
    rid = pr.create_record("Carol", "Symptom Checker", "Flu", 65.0, {})
    ok = pr.delete_record(rid)
    assert ok
    records = pr.read_all_records()
    assert not any(r["id"] == rid for r in records)


def test_delete_nonexistent_record_returns_false(temp_records_db):
    pr = temp_records_db
    assert pr.delete_record(99999) is False
