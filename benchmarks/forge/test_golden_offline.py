"""Offline self-tests for the Tier-3 golden harness — no live model, no network.

``score_output`` is proven directly (full / partial / case-insensitive coverage). The
``run_golden`` plumbing is proven end-to-end with the forge stub sink (the real graph
runs, zero model calls) plus a ``FakeSandbox`` for the audit, and a monkeypatched
``subprocess.run`` standing in for the live execution of the built package: a canned
stdout carrying every marker scores ``correct``; one missing a group scores
``wrong_output``; a non-zero exit scores ``run_failed``. The real execution path is
live-only — these tests prove the wiring and the scoring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from andamentum.forge.schemas import ForgeWhy
from andamentum.forge.spec import NodeKind
from andamentum.forge.tests.conftest import FakeSandbox, NodeScript, ScriptedSink

from .golden import GoldenCase, run_golden, score_output

# The model is unused when a sink is injected, but run_forge requires the keyword.
_STUB_MODEL = "stub:offline"

_CASE = GoldenCase(
    key="wiring",
    brief="Summarise the document into three bullet points.",
    input_text="Alpha note.\nBravo note.",
    marker_groups=[("alpha", "a1"), ("bravo",)],
)


# ── score_output ─────────────────────────────────────────────────────────────


def test_score_output_full_coverage() -> None:
    covered, ok = score_output(_CASE, "The alpha item and the bravo item.")
    assert covered == [True, True]
    assert ok is True


def test_score_output_partial_coverage() -> None:
    covered, ok = score_output(_CASE, "Only the alpha item appears.")
    assert covered == [True, False]
    assert ok is False


def test_score_output_any_marker_in_group_counts() -> None:
    # "a1" is an alternative marker for the first group.
    covered, ok = score_output(_CASE, "Ticket A1 resolved; bravo confirmed.")
    assert covered == [True, True]
    assert ok is True


def test_score_output_case_insensitive() -> None:
    covered, ok = score_output(_CASE, "ALPHA and BrAvO.")
    assert covered == [True, True]
    assert ok is True


def test_score_output_empty_output_covers_nothing() -> None:
    covered, ok = score_output(_CASE, "")
    assert covered == [False, False]
    assert ok is False


# ── run_golden wiring (stub build + canned execution) ────────────────────────


def _coherent_sink() -> ScriptedSink:
    """A small coherent rung-1 script: parse the request (spine) → answer it (head)."""
    return ScriptedSink(
        why=ForgeWhy(
            purpose="Summarise the input.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
        typings={
            "n1": NodeScript(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeScript(
                kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
            ),
        },
    )


def _canned_run(stdout: str, returncode: int = 0):
    """A stand-in for ``subprocess.run`` that returns a scripted completed process."""

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=returncode, stdout=stdout, stderr=""
        )

    return fake_run


async def _run_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stdout: str, returncode: int = 0
):
    """Drive run_golden with the stub build and a canned execution result."""
    monkeypatch.setattr(subprocess, "run", _canned_run(stdout, returncode))
    return await run_golden(
        _CASE,
        model=_STUB_MODEL,
        dest_root=tmp_path,
        sink=_coherent_sink(),
        sandbox=FakeSandbox(exit_code=0, stdout="2 passed in 0.10s"),
    )


async def test_run_golden_all_markers_scores_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = await _run_wired(
        tmp_path, monkeypatch, stdout="Summary: the alpha note and the bravo note."
    )
    assert outcome.kind == "correct"
    assert outcome.works is True
    assert outcome.covered == [True, True]
    assert outcome.seconds > 0


async def test_run_golden_missing_group_scores_wrong_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = await _run_wired(
        tmp_path, monkeypatch, stdout="Summary: only the alpha note."
    )
    assert outcome.kind == "wrong_output"
    assert outcome.works is True
    assert outcome.covered == [True, False]
    assert "alpha" in outcome.output_tail


async def test_run_golden_nonzero_exit_scores_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = await _run_wired(
        tmp_path, monkeypatch, stdout="Traceback: boom", returncode=1
    )
    assert outcome.kind == "run_failed"
    assert outcome.error == "exit code 1"
    assert "boom" in outcome.output_tail
