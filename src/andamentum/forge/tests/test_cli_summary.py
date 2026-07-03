"""The CLI's self-correction trace formatter — a pure function over the audit history.

Kept separate from the ForgeResult so the loop's human trace is testable without
constructing a whole spec/result."""

from __future__ import annotations

from andamentum.forge.cli import _self_correction_lines
from andamentum.forge.schemas import AuditRound


def test_no_history_is_silent() -> None:
    assert _self_correction_lines([]) == []


def test_single_pass_does_not_emit_a_trace() -> None:
    # A clean first build produces one audit pass and no loop; nothing extra to show.
    history = [AuditRound(index=1, failing_checks="")]
    assert _self_correction_lines(history) == []


def test_converged_after_one_rebuild_traces_each_pass() -> None:
    history = [
        AuditRound(
            index=1,
            failing_checks="tests: 1 failed",
            rebuild_targets=["NormaliseTheRequest"],
        ),
        AuditRound(index=2, failing_checks=""),
    ]
    lines = _self_correction_lines(history)
    text = "\n".join(lines)
    # header names the pass + rebuild count
    assert "self-correction: 2 audit passes (1 rebuild)" in text
    # pass 1 shows what failed and what was re-authored
    assert "pass 1: failed — tests: 1 failed" in text
    assert "re-authored: NormaliseTheRequest" in text
    # pass 2 is clean
    assert "pass 2: clean" in text


def test_plural_rebuilds() -> None:
    history = [
        AuditRound(index=1, failing_checks="x", rebuild_targets=["A"]),
        AuditRound(index=2, failing_checks="y", rebuild_targets=["B"]),
        AuditRound(index=3, failing_checks="z"),
    ]
    header = _self_correction_lines(history)[0]
    assert "3 audit passes (2 rebuilds)" in header
