"""
cli.py
-------
USER INTERFACE LAYER

A menu-driven console interface (in the same spirit as a typical
VITyarthi console project) that wires together all three functional
modules:

    1. Data Management       (src.data_loader, src.patient_records)
    2. Prediction Engine     (src.predictor, src.model_trainer)
    3. Reporting & Analytics (src.report_generator)

Every user input is validated; every action is logged; every unexpected
error is caught so the menu loop never crashes on bad input.
"""

import os
import sys

from src.logger_config import get_logger
from src.validators import ValidationError
from src.predictor import (
    predict_diabetes, predict_heart_disease, predict_illness_from_symptoms,
    ModelNotTrainedError, SYMPTOM_FIELDS,
)
from src.model_trainer import train_all_models
from src.patient_records import (
    create_record, read_all_records, read_records_by_patient,
    update_record, delete_record,
)
from src.report_generator import (
    print_text_report, plot_prediction_distribution, plot_model_performance,
)

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_REQUIRED_MODEL_FILES = ["diabetes_model.pkl", "heart_model.pkl", "symptom_model.pkl"]


def _models_are_trained() -> bool:
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in _REQUIRED_MODEL_FILES)


def _auto_train_if_needed():
    """On first run, train all models automatically so the app works out of the box."""
    if _models_are_trained():
        return
    print("\nNo trained models found — training all models automatically (first run only)...")
    summary = train_all_models()
    for disease, metrics in summary.items():
        print(f"  {disease.title():<10} accuracy: {metrics['accuracy']}")
    print("Done. You're ready to make predictions!\n")


def _prompt_float(label: str):
    return input(f"  {label}: ").strip()


def _prompt_int_choice(label: str, choices):
    return input(f"  {label} {choices}: ").strip()


def _pause():
    input("\nPress Enter to continue...")


# --------------------------------------------------------------------------
# Prediction flows
# --------------------------------------------------------------------------

def _run_diabetes_flow():
    print("\n--- Diabetes Risk Prediction ---")
    patient_name = input("  Patient name: ").strip()
    raw = {
        "Pregnancies": _prompt_float("Pregnancies (0-20)"),
        "Glucose": _prompt_float("Glucose level mg/dL (40-260)"),
        "BloodPressure": _prompt_float("Blood pressure mmHg (20-140)"),
        "SkinThickness": _prompt_float("Skin thickness mm (0-100)"),
        "Insulin": _prompt_float("Insulin level mu U/mL (0-900)"),
        "BMI": _prompt_float("BMI (10-70)"),
        "DiabetesPedigreeFunction": _prompt_float("Diabetes pedigree function (0.0-3.0)"),
        "Age": _prompt_float("Age (1-120)"),
    }
    try:
        result = predict_diabetes(raw)
    except ValidationError as e:
        print(f"  [Input Error] {e}")
        return
    except ModelNotTrainedError as e:
        print(f"  [Model Error] {e}")
        return

    print(f"\n  Result: {result['prediction']}  (confidence: {result['confidence']}%)")
    record_id = create_record(patient_name, "Diabetes", result["prediction"],
                               result["confidence"], result["input"])
    print(f"  Saved as record #{record_id}.")


def _run_heart_flow():
    print("\n--- Heart Disease Risk Prediction ---")
    patient_name = input("  Patient name: ").strip()
    raw = {
        "age": _prompt_float("Age (1-120)"),
        "sex": _prompt_int_choice("Sex", "1=Male, 0=Female"),
        "cp": _prompt_int_choice("Chest pain type", [0, 1, 2, 3]),
        "trestbps": _prompt_float("Resting blood pressure mmHg (70-220)"),
        "chol": _prompt_float("Serum cholesterol mg/dL (100-650)"),
        "fbs": _prompt_int_choice("Fasting blood sugar > 120 mg/dL", "1=Yes, 0=No"),
        "restecg": _prompt_int_choice("Resting ECG result", [0, 1, 2]),
        "thalach": _prompt_float("Max heart rate achieved (60-220)"),
        "exang": _prompt_int_choice("Exercise-induced angina", "1=Yes, 0=No"),
        "oldpeak": _prompt_float("ST depression induced by exercise (0.0-7.0)"),
        "slope": _prompt_int_choice("Slope of peak exercise ST segment", [0, 1, 2]),
        "ca": _prompt_int_choice("Number of major vessels colored", [0, 1, 2, 3, 4]),
        "thal": _prompt_int_choice("Thalassemia type", [0, 1, 2, 3]),
    }
    try:
        result = predict_heart_disease(raw)
    except ValidationError as e:
        print(f"  [Input Error] {e}")
        return
    except ModelNotTrainedError as e:
        print(f"  [Model Error] {e}")
        return

    print(f"\n  Result: {result['prediction']}  (confidence: {result['confidence']}%)")
    record_id = create_record(patient_name, "Heart Disease", result["prediction"],
                               result["confidence"], result["input"])
    print(f"  Saved as record #{record_id}.")


def _run_symptom_flow():
    print("\n--- Symptom-Based Illness Checker ---")
    patient_name = input("  Patient name: ").strip()
    print("  Answer 1 for Yes, 0 for No, for each symptom:")
    raw = {}
    for field in SYMPTOM_FIELDS:
        raw[field] = input(f"    {field.replace('_', ' ').title()}: ").strip() or "0"

    try:
        result = predict_illness_from_symptoms(raw)
    except ValidationError as e:
        print(f"  [Input Error] {e}")
        return
    except ModelNotTrainedError as e:
        print(f"  [Model Error] {e}")
        return

    print(f"\n  Most likely condition: {result['prediction']}  (confidence: {result['confidence']}%)")
    print("  Note: This is an educational demo, not a medical diagnosis.")
    record_id = create_record(patient_name, "Symptom Checker", result["prediction"],
                               result["confidence"], result["input"])
    print(f"  Saved as record #{record_id}.")


# --------------------------------------------------------------------------
# Record management (CRUD) flows
# --------------------------------------------------------------------------

def _records_menu():
    while True:
        print("\n--- Patient Record Management ---")
        print("  1. View all records")
        print("  2. Search records by patient name")
        print("  3. Update a record's patient name")
        print("  4. Delete a record")
        print("  5. Back to main menu")
        choice = input("  Choose an option: ").strip()

        if choice == "1":
            records = read_all_records()
            _print_records(records)
        elif choice == "2":
            name = input("  Patient name to search: ").strip()
            records = read_records_by_patient(name)
            _print_records(records)
        elif choice == "3":
            try:
                rid = int(input("  Record ID to update: ").strip())
                new_name = input("  New patient name: ").strip()
                ok = update_record(rid, patient_name=new_name)
                print("  Updated." if ok else "  Record not found.")
            except ValueError:
                print("  [Input Error] Record ID must be a number.")
            except ValidationError as e:
                print(f"  [Input Error] {e}")
        elif choice == "4":
            try:
                rid = int(input("  Record ID to delete: ").strip())
                ok = delete_record(rid)
                print("  Deleted." if ok else "  Record not found.")
            except ValueError:
                print("  [Input Error] Record ID must be a number.")
        elif choice == "5":
            return
        else:
            print("  Invalid option, please try again.")


def _print_records(records):
    if not records:
        print("  No records found.")
        return
    print(f"\n  {'ID':<4}{'Patient':<18}{'Module':<18}{'Prediction':<20}{'Conf%':<8}{'Created At'}")
    print("  " + "-" * 80)
    for r in records:
        print(f"  {r['id']:<4}{r['patient_name']:<18}{r['disease_module']:<18}"
              f"{r['prediction']:<20}{r['confidence']:<8}{r['created_at']}")


# --------------------------------------------------------------------------
# Reporting flows
# --------------------------------------------------------------------------

def _reports_menu():
    while True:
        print("\n--- Reporting & Analytics ---")
        print("  1. Print text summary of prediction history")
        print("  2. Generate prediction-distribution chart")
        print("  3. Generate model performance chart")
        print("  4. Back to main menu")
        choice = input("  Choose an option: ").strip()

        if choice == "1":
            print_text_report()
        elif choice == "2":
            path = plot_prediction_distribution()
            print(f"  Chart saved to: {path}" if path else "  No data to chart yet.")
        elif choice == "3":
            path = plot_model_performance()
            print(f"  Chart saved to: {path}" if path else "  No trained models found. Train models first.")
        elif choice == "4":
            return
        else:
            print("  Invalid option, please try again.")


# --------------------------------------------------------------------------
# Main menu
# --------------------------------------------------------------------------

def main_menu():
    print("=" * 60)
    print("   MULTI-DISEASE PREDICTION SYSTEM (VITyarthi Project)")
    print("=" * 60)
    print("  Disclaimer: Educational demo only. Not medical advice.")

    _auto_train_if_needed()

    while True:
        print("\n=== MAIN MENU ===")
        print("  1. Train / Retrain Prediction Models")
        print("  2. Predict Diabetes Risk")
        print("  3. Predict Heart Disease Risk")
        print("  4. Symptom-Based Illness Checker")
        print("  5. Patient Record Management (CRUD)")
        print("  6. Reporting & Analytics")
        print("  7. Exit")
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                print("\nTraining all models, please wait...")
                summary = train_all_models()
                for disease, metrics in summary.items():
                    print(f"  {disease.title():<10} accuracy: {metrics['accuracy']}")
            elif choice == "2":
                _run_diabetes_flow()
            elif choice == "3":
                _run_heart_flow()
            elif choice == "4":
                _run_symptom_flow()
            elif choice == "5":
                _records_menu()
            elif choice == "6":
                _reports_menu()
            elif choice == "7":
                print("Goodbye!")
                sys.exit(0)
            else:
                print("Invalid option, please try again.")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting safely.")
            sys.exit(0)
        except Exception as e:
            # Catch-all so the menu loop never crashes on an unexpected error.
            logger.exception("Unhandled error in main menu")
            print(f"  [Unexpected Error] {e}")

        _pause()
