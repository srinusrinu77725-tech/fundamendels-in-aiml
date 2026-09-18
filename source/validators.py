"""
validators.py
--------------
Input validation utilities and custom exceptions (Non-Functional
Requirement: Error Handling Strategy). Keeping validation in one module
means every entry point (CLI, tests, future API) enforces identical rules.
"""

from typing import Union

Number = Union[int, float]


class ValidationError(Exception):
    """Raised when user-supplied input fails a validation rule."""
    pass


def validate_range(value: Number, field_name: str, low: Number, high: Number) -> float:
    """Ensure `value` lies within [low, high]; raise ValidationError otherwise."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{field_name}' must be numeric, got: {value!r}")

    if not (low <= value <= high):
        raise ValidationError(
            f"'{field_name}' must be between {low} and {high} (got {value})."
        )
    return value


def validate_choice(value, field_name: str, choices) -> object:
    """Ensure `value` is one of `choices`."""
    if value not in choices:
        raise ValidationError(
            f"'{field_name}' must be one of {choices} (got {value!r})."
        )
    return value


def validate_binary(value, field_name: str) -> int:
    """Ensure `value` is 0 or 1 (accepts '0'/'1'/0/1/True/False)."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{field_name}' must be 0 or 1 (got {value!r}).")
    if ivalue not in (0, 1):
        raise ValidationError(f"'{field_name}' must be 0 or 1 (got {value!r}).")
    return ivalue


def validate_non_empty_string(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"'{field_name}' must be a non-empty string.")
    return value.strip()
