"""A hard LLM-output failure must degrade legibly, never crash the pipeline.

A small model that can't produce schema-valid output (e.g. it returns the JSON *schema*
envelope instead of an instance) makes pydantic-ai raise ``UnexpectedModelBehavior`` after
its retries. forge must catch that at every ``sink.run`` boundary and degrade: a build node
becomes unfillable, an advisory audit head is skipped, and a load-bearing design stage fails
loud with a clean, actionable error — never an uncaught 200-line traceback that discards the
whole run."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import UnexpectedModelBehavior

from andamentum.core import AgentDefinition
from andamentum.forge import build_system, compile_spec, render, run_forge
from andamentum.forge.schemas import (
    DataKind,
    DesignPlan,
    ForgeWhy,
    NodeDraft,
    NodeKind,
    NodeTyping,
)

from .conftest import FakeSandbox, ScriptedSink


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


def _build_kwargs() -> dict[str, object]:
    """Minimal sink config for build_system-level tests (design heads unused there)."""
    return dict(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Answer the request."]},
    )


def _run_forge_kwargs() -> dict[str, object]:
    """Full sink config so the whole design pipeline produces a valid spec."""
    return dict(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
        typings={
            "n1": NodeTyping(
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["parsed_request"],
                produces_kind=DataKind.SIGNAL,
            ),
            "n2": NodeTyping(
                kind=NodeKind.HEAD,
                consumes=["parsed_request"],
                produces=["answer"],
                produces_kind=DataKind.SIGNAL,
            ),
        },
    )


_BOOM = UnexpectedModelBehavior("Exceeded maximum output retries (5)")


class _DraftFailsSink(ScriptedSink):
    """The body-authoring head always fails hard (like a small model returning garbage)."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name in ("build_draft", "build_repair"):
            raise _BOOM
        return await super().run(defn, **kwargs)


class _ManagerFailsSink(ScriptedSink):
    """The advisory component manager fails hard; the gate-valid body must still stand."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "component_manager":
            raise _BOOM
        return await super().run(defn, **kwargs)


class _AuditHeadsFailSink(ScriptedSink):
    """Both advisory audit heads fail hard; the audit must still complete."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name in ("requirements", "critic"):
            raise _BOOM
        return await super().run(defn, **kwargs)


class _UnderstandFailsSink(ScriptedSink):
    """A load-bearing design head fails hard — terminal, but must be legible."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "understand":
            raise _BOOM
        return await super().run(defn, **kwargs)


# --- Build: a per-node authoring failure degrades to UnfillableNode ------------------


async def test_build_degrades_to_unfillable_when_model_cannot_author(
    tmp_path: Path,
) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    sink = _DraftFailsSink(**_build_kwargs())  # type: ignore[arg-type]

    report = await build_system(spec, pkg, sink=sink, attempt_cap=3)  # must NOT raise

    assert not report.all_filled
    assert report.unfillable, "a body the model cannot author must settle unfillable"
    assert any(
        "failed to produce a valid body" in u.last_error for u in report.unfillable
    ), "the unfillable node records the model failure as its last error"


# --- Build: the advisory component manager failing does not block a gate-valid body --


async def test_component_manager_failure_keeps_the_gate_valid_body(
    tmp_path: Path,
) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    sink = _ManagerFailsSink(**_build_kwargs())  # type: ignore[arg-type]

    report = await build_system(spec, pkg, sink=sink, attempt_cap=3)  # must NOT raise

    assert report.all_filled, report.unfillable  # advisory failure never blocks fillability


# --- Audit: advisory heads failing are skipped, the run still completes --------------


async def test_audit_skips_advisory_heads_on_model_failure(tmp_path: Path) -> None:
    result = await run_forge(
        "Help the user manage a personal reading list.",
        model="stub",
        dest=tmp_path,
        sink=_AuditHeadsFailSink(**_run_forge_kwargs()),  # type: ignore[arg-type]
        sandbox=FakeSandbox(),
    )
    # The run reaches audit and completes — no crash — with the advisory heads skipped.
    assert result.audit is not None
    assert result.audit.requirements is None
    assert result.audit.critic is None


# --- Design: a load-bearing head failing is terminal but legible (clean ValueError) --


async def test_design_stage_model_failure_raises_a_clean_valueerror(
    tmp_path: Path,
) -> None:
    sink = _UnderstandFailsSink(**_run_forge_kwargs())  # type: ignore[arg-type]

    with pytest.raises(ValueError) as excinfo:
        await run_forge(
            "manage my reading list",
            model="stub",
            dest=tmp_path,
            sink=sink,
            sandbox=FakeSandbox(),
        )

    # Translated to forge's fail-loud ValueError (which the CLI presents) — NOT a raw
    # pydantic-ai UnexpectedModelBehavior traceback.
    assert not isinstance(excinfo.value, UnexpectedModelBehavior)
    msg = str(excinfo.value)
    assert "Understand" in msg
    assert "more capable model" in msg
