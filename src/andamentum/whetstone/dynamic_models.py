"""Dynamic Pydantic model creation from AnalysisField specifications.

Simplified port of ``src/mosaic/core/dynamic_models.py`` for the standalone
document-review package.  Only the 4 scalar types used by AnalysisField are
supported: ``str``, ``int``, ``float``, ``bool``.

Usage::

    from andamentum.whetstone.dynamic_models import convert_fields_to_schema, create_output_model
    from andamentum.whetstone.agents.output_models import AnalysisField

    fields = [
        AnalysisField(name="clarity", description="Clarity 1-5", field_type="int", min_value=1, max_value=5),
        AnalysisField(name="summary", description="Brief summary", field_type="str"),
    ]
    spec = convert_fields_to_schema(fields)
    Model = create_output_model("custom_review", spec)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model

from .agents.output_models import AnalysisField

# Scalar types supported by AnalysisField
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def convert_fields_to_schema(fields: list[AnalysisField]) -> dict[str, Any]:
    """Convert a list of AnalysisField objects into the dict format expected by :func:`create_output_model`.

    Returns:
        ``{"fields": {"field_name": {"type": "str", ...}, ...}}``
    """
    schema_fields: dict[str, Any] = {}

    for field in fields:
        field_spec: dict[str, Any] = {
            "type": field.field_type,
            "description": field.description,
        }

        if field.field_type in ("int", "float"):
            if field.min_value is not None:
                field_spec["ge"] = field.min_value
                range_info = f" (minimum: {field.min_value}"
                if field.max_value is not None:
                    range_info += f", maximum: {field.max_value}"
                range_info += ")"
                field_spec["description"] = field.description + range_info

            if field.max_value is not None:
                field_spec["le"] = field.max_value
                if field.min_value is None:
                    field_spec["description"] = field.description + f" (maximum: {field.max_value})"

        schema_fields[field.name] = field_spec

    return {"fields": schema_fields}


def create_output_model(agent_name: str, model_spec: dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model from a field specification dict.

    Args:
        agent_name: Used to derive the model class name.
        model_spec: ``{"fields": {"name": {"type": "str", "description": "...", ...}, ...}}``

    Returns:
        A dynamically-created Pydantic BaseModel subclass.
    """
    model_name = model_spec.get("name", f"{agent_name.title().replace('_', '')}Output")
    fields_spec = model_spec.get("fields", {})

    if not fields_spec:
        raise ValueError(f"output_model for {agent_name} must define 'fields'")

    field_definitions: dict[str, Any] = {}

    for field_name, field_spec in fields_spec.items():
        if not isinstance(field_spec, dict):
            raise ValueError(f"Field '{field_name}' must be a dictionary")

        field_type_str = field_spec.get("type", "str")
        description = field_spec.get("description", "")
        default = field_spec.get("default", ...)

        field_type = _TYPE_MAP.get(field_type_str, str)

        field_kwargs: dict[str, Any] = {"description": description}

        ge = field_spec.get("ge")
        le = field_spec.get("le")
        if ge is not None:
            field_kwargs["ge"] = ge
        if le is not None:
            field_kwargs["le"] = le

        if default is not ...:
            field_kwargs["default"] = default

        field_definitions[field_name] = (field_type, Field(**field_kwargs))

    return create_model(model_name, **field_definitions)
