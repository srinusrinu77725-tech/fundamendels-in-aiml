"""
patient_records.py
--------------------
FUNCTIONAL MODULE 3 (part A): PATIENT RECORD MANAGEMENT (CRUD)

Stores every prediction made through the system as a patient record in a
local SQLite database (data/records.db), and provides Create, Read,
Update, and Delete operations over that history. This underpins the
Reporting & Analytics module and satisfies the "CRUD operations"
functional requirement.
"""

import os
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

from src.logger_config import get_logger
from src.validators import validate_non_empty_string, ValidationError

logger = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "records.db")


@contextmanager
def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create the records table if it does not already exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                disease_module TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                input_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
    logger.info("Database initialised at %s", DB_PATH)


def create_record(patient_name: str, disease_module: str, prediction: str,
                   confidence: float, input_data: dict) -> int:
    """CREATE: insert a new prediction record. Returns the new record's id."""
    patient_name = validate_non_empty_string(patient_name, "patient_name")
    init_db()
    with _get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO records (patient_name, disease_module, prediction, "
            "confidence, input_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (patient_name, disease_module, prediction, confidence,
             json.dumps(input_data), datetime.now().isoformat(timespec="seconds"))
        )
        record_id = cur.lastrowid
    logger.info("Created record #%d for patient '%s' (%s)", record_id, patient_name, disease_module)
    return record_id


def read_all_records() -> list:
    """READ: return every stored record, most recent first."""
    init_db()
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def read_records_by_patient(patient_name: str) -> list:
    """READ: return all records for a given patient name (case-insensitive)."""
    init_db()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM records WHERE LOWER(patient_name) = LOWER(?) ORDER BY id DESC",
            (patient_name,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_record(record_id: int, **fields) -> bool:
    """UPDATE: modify one or more fields of an existing record."""
    if not fields:
        raise ValidationError("update_record requires at least one field to update.")
    init_db()
    allowed = {"patient_name", "disease_module", "prediction", "confidence"}
    set_clause = ", ".join(f"{k} = ?" for k in fields if k in allowed)
    values = [v for k, v in fields.items() if k in allowed]
    if not set_clause:
        raise ValidationError("No valid fields supplied to update_record.")
    values.append(record_id)
    with _get_connection() as conn:
        cur = conn.execute(f"UPDATE records SET {set_clause} WHERE id = ?", values)
        updated = cur.rowcount > 0
    logger.info("Update record #%d -> %s", record_id, "success" if updated else "not found")
    return updated


def delete_record(record_id: int) -> bool:
    """DELETE: remove a record by id."""
    init_db()
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        deleted = cur.rowcount > 0
    logger.info("Delete record #%d -> %s", record_id, "success" if deleted else "not found")
    return deleted
