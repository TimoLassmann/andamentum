"""The schema-envelope unwrap — the deterministic repair for small models' one
structured-output failure mode.

A small model on the PromptedOutput path sometimes returns the JSON *schema envelope*
(``{"properties": {...}, "required": [...], "type": "object", "title": ...}``) with the
real instance intact under ``properties``. ``EnvelopeTolerantModel`` unwraps that
losslessly before validation; anything still invalid fails exactly as loudly as before.
The guarantee must hold on BOTH sides of forge's boundary: forge's own agent-output
schemas (drift-tested here against ``agents.py``) and the agent-output models rendered
into every generated system."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

import andamentum.forge.agents as forge_agents
from andamentum.core import AgentDefinition
from andamentum.forge import compile_spec, render
from andamentum.forge.runtime import EnvelopeTolerantModel
from andamentum.forge.schemas import (
    DesignPlan,
    ForgeWhy,
    NodeDraft,
    NodeKind,
    PieceOut,
)

# --- the unwrap itself ----------------------------------------------------------


def test_unwraps_the_exact_envelope_from_the_field_failure() -> None:
    # The verbatim shape a gemma4:12b run produced (the crash this feature fixes):
    # schema vocabulary at the top level, the real answer under `properties`.
    envelope = {
        "properties": {"body": "ctx.state.x = 1\nreturn Done()"},
        "required": ["body"],
        "type": "object",
        "title": "PieceOut",
        "description": "A draft/repair head's output.",
    }
    out = PieceOut.model_validate(envelope)
    assert out.body == "ctx.state.x = 1\nreturn Done()"


def test_a_plain_instance_passes_through_untouched() -> None:
    out = PieceOut.model_validate({"body": "return Done()"})
    assert out.body == "return Done()"


def test_an_envelope_of_field_schemas_still_fails_loud() -> None:
    # `properties` holding field *definitions* (a genuine schema, no values) must not
    # validate — the unwrap is lossless normalisation, never invention.
    schema_only = {
        "properties": {"body": {"type": "string", "title": "Body"}},
        "required": ["body"],
        "type": "object",
    }
    with pytest.raises(ValidationError):
        PieceOut.model_validate(schema_only)


def test_a_real_field_at_top_level_blocks_the_unwrap() -> None:
    # If any top-level key is NOT JSON-Schema vocabulary, the payload is treated as an
    # instance as-is (the precision guard: never unwrap a legitimate instance).
    payload = {"body": "return Done()", "properties": {"body": "WRONG"}}
    out = PieceOut.model_validate(payload)
    assert out.body == "return Done()"


def test_properties_alone_without_corroboration_is_not_unwrapped() -> None:
    # `{"properties": {...}}` with no second envelope marker (type/required/title) is
    # too ambiguous to touch; it fails validation like any other wrong payload.
    with pytest.raises(ValidationError):
        PieceOut.model_validate({"properties": {"body": "x"}})


# --- forge side: every agent head is covered (drift test) ------------------------


def test_every_forge_agent_output_model_is_envelope_tolerant() -> None:
    defs = [
        v for v in vars(forge_agents).values() if isinstance(v, AgentDefinition)
    ]
    assert defs, "expected forge to declare its heads as AgentDefinition constants"
    for d in defs:
        assert d.output_model is not None
        assert issubclass(d.output_model, EnvelopeTolerantModel), (
            f"agent {d.name!r}: output model {d.output_model.__name__} must inherit "
            "EnvelopeTolerantModel — a new head must not reintroduce the envelope crash"
        )


# --- generated side: rendered systems inherit the same guarantee -----------------


def _plan() -> DesignPlan:
    return DesignPlan(
        why=ForgeWhy(
            purpose="Help manage a reading list.",
            boundary_in="a request",
            boundary_out="an answer",
        ),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Parse the request.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["parsed_request"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Answer the request.",
                kind=NodeKind.HEAD,
                consumes=["parsed_request"],
                produces=["answer"],
            ),
        ],
    )


def test_rendered_agent_outputs_inherit_the_envelope_base(tmp_path: Path) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    models_src = (tmp_path / spec.name / "models.py").read_text()

    tree = ast.parse(models_src)
    bases_by_class = {
        node.name: {b.id for b in node.bases if isinstance(b, ast.Name)}
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    agent_outputs = {a.output.name for a in spec.agents}
    assert agent_outputs, "the reading-list spec has one head, so one agent output"
    for name in agent_outputs:
        assert "EnvelopeTolerantModel" in bases_by_class[name], (
            f"generated agent-output model {name} must inherit EnvelopeTolerantModel"
        )
    # Input / entities / State are filled by code, not an LLM — they stay plain.
    input_name = spec.input.model.name
    assert bases_by_class[input_name] == {"BaseModel"}
