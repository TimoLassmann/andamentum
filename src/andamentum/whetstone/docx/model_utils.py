"""Utility functions for working with Pydantic models and dicts.

Inlined from utilities.model_utils for standalone package use.
"""

from typing import Any, Union

from .constants import (
    SKIP_FIELDS,
    SCORE_SUFFIX,
    JUSTIFICATION_SUFFIX,
    STRUCTURED_FIELDS,
)


def normalize_to_dict(obj: Any) -> dict[str, Any]:
    """Convert Pydantic model, dict, or object to dict format."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, dict):
        return obj
    elif hasattr(obj, "__dict__"):
        return vars(obj)
    else:
        return dict(obj) if hasattr(obj, "__iter__") else {}


def get_field(obj: Any, *field_names: str, default: Any = None) -> Any:
    """Get field value from Pydantic model, dict, or object."""
    if isinstance(obj, dict):
        for name in field_names:
            if name in obj:
                return obj[name]
    else:
        for name in field_names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def extract_fields(obj: Any, field_mapping: dict[str, Union[str, list[str]]]) -> dict[str, Any]:
    """Extract multiple fields from object using a mapping."""
    result = {}
    for output_key, field_names in field_mapping.items():
        if isinstance(field_names, str):
            field_names = [field_names]
        result[output_key] = get_field(obj, *field_names, default=None)
    return result


def categorize_review_fields(
    review_data: dict[str, Any],
    skip_fields: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Categorize review fields by type (scores, justifications, structured)."""
    if skip_fields is None:
        skip_fields = SKIP_FIELDS

    categories: dict[str, dict[str, Any]] = {"scores": {}, "justifications": {}, "structured": {}, "other": {}}

    for key, value in review_data.items():
        if key in skip_fields:
            continue
        if SCORE_SUFFIX in key and isinstance(value, (int, float)):
            label = key.replace(SCORE_SUFFIX, "").replace("_", " ").title()
            categories["scores"][label] = value
        elif JUSTIFICATION_SUFFIX in key and isinstance(value, str):
            label = key.replace(JUSTIFICATION_SUFFIX, "").replace("_", " ").title()
            categories["justifications"][label] = value
        elif key in STRUCTURED_FIELDS:
            categories["structured"][key] = value
        elif not key.endswith(JUSTIFICATION_SUFFIX):
            categories["other"][key] = value

    return categories
