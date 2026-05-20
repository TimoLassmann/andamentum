"""Live-Ollama smoke test for the Strunk sub-graph.

Marked ``@pytest.mark.ollama`` so it is deselected by the default
``addopts = "-m 'not ollama and not benchmark'"``. Run explicitly with::

    uv run pytest -m ollama src/andamentum/whetstone/lenses/strunk/

This test does NOT assert anything about the LLM's judgement — that
is the calibration phase, intentionally out of Phase A scope. It only
verifies the wiring: the full sub-graph runs end-to-end with a real
``AgentRunner``, completes without crashing, and the deterministic
R2 finding (which is content-independent and must always fire) is
present in the output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.core.agents import AgentRunner
from andamentum.whetstone.lenses.strunk import run_strunk_lens
from andamentum.whetstone.lenses.strunk.state import StrunkLensDeps
from andamentum.whetstone.structural.types import SectionRef


pytestmark = pytest.mark.ollama


async def test_smoke_runs_with_real_model_without_crash():
    fixture = Path(__file__).parent / "fixtures" / "sample_draft.md"
    text = fixture.read_text()
    section = SectionRef(
        id="sec_001",
        title="Sample Draft",
        text=text,
        char_start=0,
        char_end=len(text),
    )
    runner = AgentRunner(model="ollama:gemma3:4b-it-q4_K_M")
    deps = StrunkLensDeps(executor=runner)

    findings = await run_strunk_lens(section, deps=deps)

    # Shape: returns a list of public Finding objects.
    assert isinstance(findings, list)
    # The R2 deterministic screen is purely text-driven, so the planted
    # Oxford-comma omission must always be flagged regardless of the
    # LLM verdict on R11 / R13.
    assert any(f.category == "r2-series-comma" for f in findings), (
        f"deterministic R2 finding missing — got categories: "
        f"{[f.category for f in findings]}"
    )
    # Every finding should have the lens tag.
    for f in findings:
        assert f.perspective == "strunk"
