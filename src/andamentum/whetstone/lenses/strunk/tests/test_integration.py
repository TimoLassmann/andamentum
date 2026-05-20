"""End-to-end mock-LLM integration test for the Strunk sub-graph.

Feeds a hand-crafted draft with planted R2, R11, R13 violations
through the full sub-graph (DeterministicScreen → R11 → R13 →
ResolveDemands → Aggregate). The ``StubAgentExecutor`` returns
canned ``ActiveVoiceReport`` / ``OmitNeedlessWordsReport`` lists,
so no LLM is involved and the test runs deterministically.

This pins the load-bearing perf property: **one LLM call per rule
per section** — not per sentence.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.whetstone.lenses.strunk import run_strunk_lens
from andamentum.whetstone.lenses.strunk.models import (
    ActiveVoiceReport,
    ActiveVoiceViolation,
    OmitNeedlessWordsReport,
    OmitNeedlessWordsViolation,
)
from andamentum.whetstone.lenses.strunk.state import StrunkLensDeps
from andamentum.whetstone.structural.types import SectionRef


_FIXTURE = Path(__file__).parent / "fixtures" / "sample_draft.md"


def _r11_report() -> ActiveVoiceReport:
    return ActiveVoiceReport(
        violations=[
            ActiveVoiceViolation(
                span="were made by the team",
                suggested_active_rewrite=(
                    "The team made errors during the initial run."
                ),
                confidence="high",
            ),
        ]
    )


def _r13_report() -> OmitNeedlessWordsReport:
    return OmitNeedlessWordsReport(
        violations=[
            OmitNeedlessWordsViolation(
                span="The reason that we ran the second experiment is that",
                category="throat-clearing",
                suggested_deletion=(
                    "We ran the second experiment to confirm the first finding."
                ),
                confidence="high",
            ),
        ]
    )


def _responder(defn, kwargs):
    if defn.name == "strunk.r13_omit_needless_words":
        return _r13_report()
    if defn.name == "strunk.r11_active_voice":
        return _r11_report()
    raise AssertionError(f"Unexpected agent: {defn.name!r}")


def _section_from_fixture() -> SectionRef:
    text = _FIXTURE.read_text()
    return SectionRef(
        id="sec_001",
        title="Sample Draft",
        text=text,
        char_start=0,
        char_end=len(text),
    )


async def test_end_to_end_finds_r2_r11_r13(stub_executor):
    section = _section_from_fixture()
    deps = StrunkLensDeps(executor=stub_executor(_responder))
    findings = await run_strunk_lens(section, deps=deps)

    by_category = {f.category: f for f in findings}
    assert set(by_category.keys()) == {
        "r2-series-comma",
        "r11-active-voice",
        "r13-throat-clearing",
    }, f"unexpected categories: {[f.category for f in findings]}"

    for f in findings:
        assert f.perspective == "strunk"
        assert f.source == "investigate"
        assert f.sections_involved == ["sec_001"]
        assert len(f.quotes) == 1
        assert f.quotes[0].section_id == "sec_001"


async def test_end_to_end_r2_anchors_to_oxford_violation(stub_executor):
    section = _section_from_fixture()
    deps = StrunkLensDeps(executor=stub_executor(_responder))
    findings = await run_strunk_lens(section, deps=deps)

    [r2] = [f for f in findings if f.category == "r2-series-comma"]
    quote = r2.quotes[0]
    assert quote.text == "red, white and blue"
    assert section.text[quote.char_start : quote.char_end] == quote.text
    assert "red, white, and blue" in r2.rationale  # suggested rewrite


async def test_one_llm_call_per_rule_per_section(stub_executor):
    """Per-section design: each rule node makes ONE call per section.
    No per-sentence fan-out, no pre-screens."""
    section = _section_from_fixture()
    executor = stub_executor(_responder)
    deps = StrunkLensDeps(executor=executor)
    await run_strunk_lens(section, deps=deps)

    call_counts: dict[str, int] = {}
    for name, _ in executor.calls:
        call_counts[name] = call_counts.get(name, 0) + 1
    assert call_counts == {
        "strunk.r11_active_voice": 1,
        "strunk.r13_omit_needless_words": 1,
    }


async def test_each_call_receives_whole_section(stub_executor):
    section = _section_from_fixture()
    executor = stub_executor(_responder)
    deps = StrunkLensDeps(executor=executor)
    await run_strunk_lens(section, deps=deps)
    for _name, kwargs in executor.calls:
        assert kwargs == {"section_text": section.text}


async def test_end_to_end_executor_none_yields_only_deterministic(stub_executor):
    """When deps.executor is None, agent nodes pass straight through
    and only deterministic findings appear."""
    section = _section_from_fixture()
    deps = StrunkLensDeps(executor=None)
    findings = await run_strunk_lens(section, deps=deps)
    assert {f.category for f in findings} == {"r2-series-comma"}


async def test_end_to_end_sorts_findings_by_char_offset(stub_executor):
    section = _section_from_fixture()
    deps = StrunkLensDeps(executor=stub_executor(_responder))
    findings = await run_strunk_lens(section, deps=deps)
    starts = [f.quotes[0].char_start for f in findings]
    assert starts == sorted(starts)
