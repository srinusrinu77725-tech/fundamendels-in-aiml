# Problem Statement — Multi-Disease Prediction System

## Problem Statement

Early risk-awareness of common conditions like diabetes and heart
disease can encourage people to seek timely medical checkups, but most
people have no easy way to explore "what does my risk look like based on
these numbers?" outside of a doctor's office. Separately, everyday minor
illnesses (cold, flu, food poisoning, migraine, etc.) share overlapping
symptoms and are often self-diagnosed incorrectly.

This project builds a small, educational **Multi-Disease Prediction
System** that applies supervised machine-learning classification to
three related problems: estimating diabetes risk, estimating heart
disease risk, and suggesting a likely everyday illness from a symptom
checklist — all wrapped in a single, testable, well-structured
application rather than a single throwaway script.

## Scope of the Project

**In scope:**
- Training and serving three independent classification models
  (Diabetes, Heart Disease, Symptom → Illness).
- Full input validation and error handling around every prediction.
- Local persistence (CRUD) of every prediction made, for later review.
- Text and chart-based analytics over prediction history and model
  performance.
- A console-based menu interface tying the above together.
- Automated tests covering validation, training, prediction, and CRUD.

**Out of scope:**
- Real patient data or integration with any hospital/clinical system.
- Any claim of diagnostic accuracy suitable for real medical use.
- A graphical (web/desktop) front end — the interface is console-based
  for this submission, though the modules are designed so a web UI
  could be layered on top later without changing the core logic.

## Target Users

- **Students / instructors** evaluating the project as a demonstration
  of applied machine learning, software modularity, and testing
  practices.
- **Hobbyist learners** who want a runnable example of how to structure
  a small ML-backed console application (data → model → validated
  prediction → persistence → reporting).

*(This project is not intended for use by patients or clinicians for
real health decisions.)*

## High-Level Features

1. **Data Management** — synthetic dataset generation (fixed random
   seed for reproducibility), cleaning of implausible/missing values,
   and a shared preprocessing pipeline (scaling, train/test split).
2. **Prediction Engine** — Random Forest models for Diabetes and Heart
   Disease, and a Decision Tree model for the Symptom Checker; each
   prediction returns a class label plus a confidence percentage.
3. **Patient Record Management (CRUD)** — every prediction is saved
   with the patient's name, module, result, confidence, and timestamp
   in a local SQLite database, with full create/read/update/delete
   support from the menu.
4. **Reporting & Analytics** — a text summary of prediction history and
   two auto-generated charts: predictions-per-module, and model
   performance (accuracy/precision/recall/F1) comparison.
5. **Robustness** — centralized input validation, custom exceptions,
   rotating file logging, and a menu loop that catches unexpected
   errors instead of crashing.
