"""Tests for the R13 (omit needless words) agent node — per-section design.

Two layers:

* Pure-function tests on ``_violation_to_finding``.
* Mocked-LLM tests via the full sub-graph: ``StubAgentExecutor``
  returns canned ``OmitNeedlessWordsReport``s and we check the
  findings come out the other end.
"""

from __future__ import annotations

from andamentum.whetstone.lenses.strunk import run_strunk_lens
from andamentum.whetstone.lenses.strunk.models import (
    ActiveVoiceReport,
    OmitNeedlessWordsReport,
    OmitNeedlessWordsViolation,
)
from andamentum.whetstone.lenses.strunk.nodes.r13_omit_needless_words import (
    OMIT_NEEDLESS_WORDS_AGENT,
    _violation_to_finding,
)
from andamentum.whetstone.lenses.strunk.state import StrunkLensDeps
from andamentum.whetstone.structural.types import SectionRef


# ── Anchor → Finding ────────────────────────────────────────────────────


def test_violation_to_finding_anchors_and_categorises():
    section_text = "The reason that I came is that I wanted to see you."
    v = OmitNeedlessWordsViolation(
        span="The reason that I came is that",
        category="throat-clearing",
        suggested_deletion="I came because I wanted to see you.",
        confidence="high",
    )
    f = _violation_to_finding(v, section_text)
    assert f is not None
    assert f.rule_number == 13
    assert f.rule_name == "omit-needless-words"
    assert f.category == "r13-throat-clearing"
    assert f.span_text == "The reason that I came is that"
    assert section_text[f.char_start : f.char_end] == f.span_text


def test_violation_to_finding_each_category():
    section_text = (
        "The reaction was rather slow. "
        "The advance planning paid off. "
        "It was of a fragile nature. "
        "There was throat-clearing here. "
        "Other things, too."
    )
    for category, span in [
        ("weak-qualifier", "rather"),
        ("redundancy", "advance planning"),
        ("filler-prepositional", "of a fragile nature"),
        ("other", "Other things"),
    ]:
        v = OmitNeedlessWordsViolation(
            span=span,
            category=category,  # type: ignore[arg-type]
            confidence="high",
        )
        f = _violation_to_finding(v, section_text)
        assert f is not None
        assert f.category == f"r13-{category}"


def test_violation_to_finding_drops_fabricated_span():
    section_text = "The cat sat on the mat."
    v = OmitNeedlessWordsViolation(
        span="advance planning",  # not in section
        category="redundancy",
        confidence="high",
    )
    assert _violation_to_finding(v, section_text) is None


def test_violation_to_finding_drops_empty_span():
    v = OmitNeedlessWordsViolation(span="", category="other", confidence="low")
    assert _violation_to_finding(v, "any text") is None


# ── End-to-end (sub-graph with stubbed executor) ────────────────────────


def _section(text: str) -> SectionRef:
    return SectionRef(
        id="sec_test",
        title="Test",
        text=text,
        char_start=0,
        char_end=len(text),
    )


async def test_node_emits_finding_for_each_violation(stub_executor):
    section = _section(
        "The reason that we ran the experiment is that we wanted to confirm. "
        "The reaction was rather slow."
    )

    def respond(defn, kwargs):
        if defn is OMIT_NEEDLESS_WORDS_AGENT:
            return OmitNeedlessWordsReport(
                violations=[
                    OmitNeedlessWordsViolation(
                        span="The reason that we ran the experiment is that",
                        category="throat-clearing",
                        suggested_deletion="We ran the experiment to confirm.",
                        confidence="high",
                    ),
                    OmitNeedlessWordsViolation(
                        span="rather",
                        category="weak-qualifier",
                        suggested_deletion="The reaction was slow.",
                        confidence="high",
                    ),
                ]
            )
        return ActiveVoiceReport(violations=[])

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    r13 = [f for f in findings if f.category.startswith("r13-")]
    assert len(r13) == 2
    categories = {f.category for f in r13}
    assert categories == {"r13-throat-clearing", "r13-weak-qualifier"}


async def test_one_llm_call_per_rule_per_section(stub_executor):
    """The hot path: one R11 call + one R13 call per section, never
    per sentence. This is the load-bearing perf assertion."""
    section = _section(
        "Sentence one. Sentence two. Sentence three. "
        "Sentence four. Sentence five."
    )

    def respond(defn, kwargs):
        if defn is OMIT_NEEDLESS_WORDS_AGENT:
            return OmitNeedlessWordsReport(violations=[])
        return ActiveVoiceReport(violations=[])

    executor = stub_executor(respond)
    deps = StrunkLensDeps(executor=executor)
    await run_strunk_lens(section, deps=deps)

    call_names = [name for name, _ in executor.calls]
    assert call_names.count("strunk.r13_omit_needless_words") == 1
    assert call_names.count("strunk.r11_active_voice") == 1
    # Each call was passed the whole section, not a sentence.
    for _name, kwargs in executor.calls:
        assert kwargs == {"section_text": section.text}


async def test_executor_none_yields_only_deterministic_findings(stub_executor):
    """With no executor, agent nodes pass through and only the
    deterministic R2 hits should appear."""
    section = _section("We tested red, white and blue.")
    deps = StrunkLensDeps(executor=None)
    findings = await run_strunk_lens(section, deps=deps)
    categories = {f.category for f in findings}
    assert categories == {"r2-series-comma"}


async def test_executor_exception_does_not_crash(stub_executor):
    section = _section("Some prose here.")

    def respond(defn, kwargs):
        raise ValueError("boom")

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    assert findings == []


async def test_wrong_output_type_recorded_as_demand(stub_executor):
    section = _section("Some prose here.")

    def respond(defn, kwargs):
        return "not a report"

    deps = StrunkLensDeps(executor=stub_executor(respond))
    findings = await run_strunk_lens(section, deps=deps)
    assert findings == []
