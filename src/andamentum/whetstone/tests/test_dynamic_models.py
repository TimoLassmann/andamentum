"""Tests for dynamic model creation from AnalysisField specs."""

import pytest

from andamentum.whetstone.agents.output_models import AnalysisField
from andamentum.whetstone.dynamic_models import (
    convert_fields_to_schema,
    create_output_model,
)


def test_schema_for_str_field():
    fields = [AnalysisField(name="summary", description="Brief summary", field_type="str")]
    spec = convert_fields_to_schema(fields)
    assert "summary" in spec["fields"]
    assert spec["fields"]["summary"]["type"] == "str"


def test_schema_for_int_with_range():
    fields = [
        AnalysisField(
            name="clarity",
            description="Clarity 1-5",
            field_type="int",
            min_value=1,
            max_value=5,
        )
    ]
    spec = convert_fields_to_schema(fields)
    assert spec["fields"]["clarity"]["ge"] == 1
    assert spec["fields"]["clarity"]["le"] == 5


def test_create_output_model_roundtrip():
    fields = [
        AnalysisField(name="doc_id", description="ID", field_type="str"),
        AnalysisField(name="score", description="Score", field_type="int", min_value=1, max_value=10),
    ]
    spec = convert_fields_to_schema(fields)
    Model = create_output_model("custom_test", spec)

    instance = Model(doc_id="abc", score=7)
    dumped = instance.model_dump()
    assert dumped["doc_id"] == "abc"
    assert dumped["score"] == 7


def test_create_output_model_requires_fields():
    with pytest.raises(ValueError, match="must define 'fields'"):
        create_output_model("empty", {"fields": {}})
