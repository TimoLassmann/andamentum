"""Tests for the R11 (active voice) agent node — per-section design.

Two layers:

* Pure-function tests on ``_violation_to_finding``: no LLM, plain
  assertions on the anchor → finding mapping.
* Mocked-LLM tests on ``R11ActiveVoice.run`` indirectly via the
  whole sub-graph (``run_strunk_lens``) with a ``StubAgentExecutor``
  returning canned ``ActiveVoiceReport``s.
"""

from __future__ import annotations

from andamentum.whetstone.lenses.strunk import run_strunk_lens
from andamentum.whetstone.lenses.strunk.models import (
    ActiveVoiceReport,
    ActiveVoiceViolation,
)
from andamentum.whetstone.lenses.strunk.nodes.r11_active_voice import (
    ACTIVE_VOICE_AGENT,
    _violation_to_finding,
)
from andamentum.whetstone.lenses.strunk.state import StrunkLensDeps
from andamentum.whetstone.structural.types import SectionRef


# ── Anchor → Finding ────────────────────────────────────────────────────


def test_violation_to_finding_anchors_to_section_text():
    section_text = "Errors were made by the team during the initial run."
    v = ActiveVoiceViolation(
        span="were made by the team",
        suggested_active_rewrite="The team made errors during the initial run.",
        confidence="high",
    )
    f = _violation_to_finding(v, section_text)
    assert f is not None
    assert f.rule_number == 11
    assert f.rule_name == "active-voice"
    assert f.span_text == "were made by the team"
    # Char offsets re-slice to the same verbatim text.
    assert section_text[f.char_start : f.char_end] == "were made by the team"
    assert f.suggested_replacement == "The team made errors during the initial run."


def test_violation_to_finding_returns_none_for_fabricated_span():
    """If the LLM hallucinates a span that doesn't appear in the
    section, the finding is dropped — we never anchor against text
    that isn't really there."""
    section_text = "The cat sat on the mat."
    v = ActiveVoiceViolation(
        span="were dispatched by the courier",
        confidence="high",
    )
    assert _violation_to_finding(v, section_text) is None


def test_violation_to_finding_drops_empty_span():
    section_text = "any text"
    v = ActiveVoiceViolation(span="", confidence="high")
    assert _violation_to_finding(v, section_text) is None


def test_violation_to_finding_propagates_confidence():
    section_text = "The X was Y by Z."
    for conf in ("low", "medium", "high"):
        v = ActiveVoiceViolation(
            span="was Y by Z", confidence=conf  # type: ignore[arg-type]
        )
        f = _violation_to_finding(v, section_text)
        assert f is not None
        assert f.confidence == conf


# ── End-to-end (sub-graph with stubbed executor) ────────────────────────


def _make_section(text: str) -> SectionRef:
    return SectionRef(
        id="sec_test",
        title="Test",
        text=text,
        char_start=0,
        char_end=len(text),
    )


async def test_node_emits_finding_when_report_lists_violation(stub_executor):
    section = _make_section("Errors were made by the team during the run.")

    def respond(defn, kwargs):
        assert defn is ACTIVE_VOICE_AGENT
        assert kwargs == {"section_text": section.text}
        return ActiveVoiceReport(
            violations=[
                ActiveVoiceViolation(
                    span="were made by the team",
                    suggested_active_rewrite="The team made errors during the run.",
                    confidence="high",
                )
            ]
        )

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)

    r11 = [f for f in findings if f.category == "r11-active-voice"]
    assert len(r11) == 1
    assert r11[0].quotes[0].text == "were made by the team"


async def test_empty_report_yields_no_r11_findings(stub_executor):
    section = _make_section("The cat sat on the mat. The dog ran fast.")

    def respond(defn, kwargs):
        if defn is ACTIVE_VOICE_AGENT:
            return ActiveVoiceReport(violations=[])
        # Any other call (e.g. R13) — return empty.
        from andamentum.whetstone.lenses.strunk.models import (
            OmitNeedlessWordsReport,
        )

        return OmitNeedlessWordsReport(violations=[])

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    assert not [f for f in findings if f.category == "r11-active-voice"]


async def test_executor_exception_records_demand_and_continues(stub_executor):
    """If the R11 agent call raises, the run does NOT crash. A demand
    is appended internally; R13 still runs (and we don't see it here
    because the executor responder raises for it too — but the test
    proves the chain doesn't bail)."""
    section = _make_section("Some prose.")

    def respond(defn, kwargs):
        raise RuntimeError("model unavailable")

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    # No findings (everything raised) but no exception either.
    assert findings == []


async def test_wrong_output_type_records_demand(stub_executor):
    section = _make_section("Some prose.")

    def respond(defn, kwargs):
        return {"not": "a report"}

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    assert findings == []


async def test_fabricated_span_dropped_during_anchoring(stub_executor):
    """LLM returns a violation whose ``span`` isn't in the section.
    The finding is silently dropped — no fabricated quotes ever
    leak into the output."""
    section = _make_section("The cat sat on the mat.")

    def respond(defn, kwargs):
        if defn is ACTIVE_VOICE_AGENT:
            return ActiveVoiceReport(
                violations=[
                    ActiveVoiceViolation(
                        span="was dispatched by the courier",  # not in section
                        suggested_active_rewrite="The courier dispatched it.",
                        confidence="high",
                    )
                ]
            )
        from andamentum.whetstone.lenses.strunk.models import (
            OmitNeedlessWordsReport,
        )

        return OmitNeedlessWordsReport(violations=[])

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    assert not [f for f in findings if f.category == "r11-active-voice"]
