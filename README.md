# Disease-prediction-system
# 🩺 Multi-Disease Prediction System

A menu-driven Python application that uses machine learning to estimate
the risk of **Diabetes**, **Heart Disease**, and to suggest a likely
**illness from a symptom checklist** (Common Cold, Flu, COVID-19,
Migraine, Food Poisoning). Built as a VITyarthi "Build Your Own Project"
submission.


---

## Overview

Instead of one narrow script, the project is built as a small, modular
system with three functional modules that talk to each other:

1. **Data Management** — generates, cleans, and preprocesses datasets.
2. **Prediction Engine** — trains and serves ML models for three disease
   modules, with full input validation.
3. **Reporting & Analytics** — stores every prediction (CRUD), and turns
   history + model metrics into text summaries and charts.

All of this is wrapped in a simple console menu (`main.py`) so it can be
demoed without any extra setup.

---

## ✨ Features

- **Diabetes Risk Prediction** — Random Forest classifier over 8 clinical
  features (glucose, BMI, blood pressure, etc.).
- **Heart Disease Risk Prediction** — Random Forest classifier over 13
  clinical features (chest pain type, cholesterol, ECG results, etc.).
- **Symptom-Based Illness Checker** — Decision Tree classifier that maps
  a 12-symptom checklist to the most likely common illness.
- **Patient Record Management (CRUD)** — every prediction is saved to a
  local SQLite database; records can be viewed, searched, updated, and
  deleted from the menu.
- **Reporting & Analytics** — text summaries plus auto-generated bar
  charts for prediction history and model performance (accuracy,
  precision, recall, F1-score).
- **Input validation & error handling** — every field is range/type
  checked before it reaches a model; invalid input never crashes the app.
- **Logging** — every action and error is logged to `logs/app.log`
  (rotating file handler) in addition to the console.
- **Automated tests** — 21 `pytest` unit/integration tests covering
  validation, data loading, model training, prediction, and CRUD.

---

## Project Structure

```
disease-prediction-system/
├── main.py                     # Entry point
├── requirements.txt
├── statement.md                 # Problem statement, scope, target users
├── data/
│   ├── generate_datasets.py    # Synthetic dataset generator (fixed seed)
│   ├── diabetes.csv            # Generated on first run
│   ├── heart.csv                # Generated on first run
│   └── symptoms.csv             # Generated on first run
├── src/
│   ├── data_loader.py          # Module 1: data loading & preprocessing
│   ├── model_trainer.py        # Module 2a: train/evaluate/save models
│   ├── predictor.py             # Module 2b: validated runtime prediction
│   ├── patient_records.py      # Module 3a: SQLite CRUD for history
│   ├── report_generator.py     # Module 3b: analytics & charts
│   ├── validators.py            # Shared input validation + exceptions
│   ├── logger_config.py         # Centralized logging setup
│   └── cli.py                   # Menu-driven user interface
├── tests/
│   └── test_system.py           # 21 automated tests (pytest)
├── models/                      # Trained model artifacts (generated)
├── reports/                      # Generated charts (PNG)
└── logs/                         # Application logs (generated)
```

## 🛠️ Technologies / Tools Used

- **Language:** Python 3.12
- **ML:** scikit-learn (RandomForestClassifier, DecisionTreeClassifier)
- **Data:** pandas, numpy
- **Persistence:** SQLite3 (built-in), joblib (model serialization)
- **Visualization:** matplotlib
- **Testing:** pytest
- **Version Control:** Git

---

## Steps to Install & Run the Project

**Prerequisites:** Python 3.10+ installed.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd disease-prediction-system

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

 **no manual setup step is needed.** The first time you run
`main.py`, it automatically:
- generates the three synthetic datasets in `data/` (if not already present),
- trains all three ML models,
- saves the trained models to `models/`.

You'll see a short "training..." message once, then the main menu
appears and you can immediately use options 2–4 to run predictions,
option 5 to manage patient records, and option 6 to view
analytics/reports. (Option 1 lets you retrain the models any time you
want, e.g. after changing the data.)

---

## 🧪 Instructions for Testing

Run the full automated test suite:

```bash
pip install pytest
pytest -v
```

This runs 21 tests covering:
- Input validation (range, binary, choice, string checks)
- Dataset loading and column integrity
- End-to-end model training (accuracy threshold check per disease)
- End-to-end predictions (valid input → valid output; invalid input → rejected)
- Full CRUD lifecycle on patient records (create, read, update, delete)

**Manual testing checklist:**
1. Launch `python main.py` → train models (option 1).
2. Run a diabetes prediction (option 2) with a high-glucose, high-BMI
   sample → expect a "Positive" risk result.
3. Run a heart disease prediction (option 3) with typical low-risk
   values → expect a "Negative" result.
4. Run the symptom checker (option 4) answering "yes" to headache,
   nausea, and fatigue → expect "Migraine" or a plausible match.
5. Open Patient Record Management (option 5) → confirm the three
   predictions above were saved; try updating and deleting a record.
6. Open Reporting & Analytics (option 6) → print the text summary and
   generate both charts; confirm PNG files appear under `reports/`.
7. Try an out-of-range input (e.g. Glucose = 9999) → confirm the app
   shows a validation error instead of crashing.

---

## 📊 Screenshots  
<img width="1920" height="1200" alt="Screenshot 2026-09-16 090311" src="https://github.com/user-attachments/assets/7a46f96b-5377-4eda-b3d1-6c8c34dc2bd7" />

<img width="1920" height="1200" alt="Screenshot 2026-09-16 090343" src="https://github.com/user-attachments/assets/19ba9d61-993b-4f8b-8700-ccf913f470da" />

<img width="1920" height="1200" alt="Screenshot 2026-09-16 090515" src="https://github.com/user-attachments/assets/15e77fd2-56a7-4a37-b763-94e4585959dc" />

<img width="840" height="540" alt="model_performance" src="https://github.com/user-attachments/assets/e942673a-f967-4763-be02-cf1363861e41" />

<img width="720" height="480" alt="prediction_distribution" src="https://github.com/user-attachments/assets/c6e6aecb-cdee-49d5-9380-88cb60794804" />

---

## 👤 Author
- NAME:-sai kumar
- REGISTRATION NUMBER:-25MIB10027
