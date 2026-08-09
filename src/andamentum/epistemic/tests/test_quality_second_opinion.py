"""A single sampled point value must not be able to destroy evidence.

``ExtractEvidenceOperation._score_evidence`` Path 2 blends four agent-produced
floats into one quality score and, below 0.10, invalidated the evidence
outright. That is a categorical cliff driven by one sample: an unlucky draw
erased real evidence, and nothing downstream could tell erasure from absence.

The invalidation is now gated by an ``llm_judge`` second opinion. These tests
pin the gate's five behaviours — destroy when confidently confirmed, retain on
disagreement, retain on doubt, never fire on the happy path, and never
silently degrade when the wiring is missing.

Every test runs offline through the shared ``stub_llm_judge`` fixture, which
patches ``llm_judge.elicit.elicit_criterion_score``.
"""

from __future__ import annotations

import pytest

from andamentum.epistemic.entities import Evidence
from andamentum.epistemic.operations.evidence import ExtractEvidenceOperation
from andamentum.llm_judge import elicit as _elicit
from andamentum.epistemic.tests.conftest import FakeAgentRunner

_LOW_QUALITY_AGENT = {
    "epistemic_assess_evidence_quality": {
        "source_credibility": 0.02,
        "relevance": 0.02,
        "specificity": 0.02,
        "recency_appropriate": 0.02,
        "justification": "Page appears to be navigation chrome.",
    }
}

_NORMAL_QUALITY_AGENT = {
    "epistemic_assess_evidence_quality": {
        "source_credibility": 0.8,
        "relevance": 0.7,
        "specificity": 0.6,
        "recency_appropriate": 0.7,
        "justification": "A peer-reviewed cohort study.",
    }
}


def _op(repo, *, overrides, judge_model: str | None = "test:judge"):
    return ExtractEvidenceOperation(
        repo,
        FakeAgentRunner(overrides=overrides),
        judge_model=judge_model,
    )


def _evidence(content: str = "A randomised trial of 4200 adults reported ...") -> Evidence:
    return Evidence(
        objective_id="obj-1",
        source_type="web_search",
        source_ref="https://example.org/paper",
        extracted_content=content,
    )


async def test_low_quality_evidence_is_invalidated_when_the_judge_confidently_fails(
    repo, stub_llm_judge
) -> None:
    """The destroy path still works when it should.

    PINS: this change gates TMS invalidation, it does not disable it.
    """
    recorder = stub_llm_judge((2, 3, 95), reasoning="Empty navigation chrome.")
    ev = _evidence()
    await _op(repo, overrides=_LOW_QUALITY_AGENT)._score_evidence(ev)

    assert ev.invalidated is True
    assert ev.quality_distribution is not None
    assert "independent judge" in (ev.invalidation_reason or "")
    assert len(recorder.calls) == 1


async def test_low_quality_evidence_survives_when_the_judge_disagrees(
    repo, stub_llm_judge
) -> None:
    """THE CORE BEHAVIOUR THIS CHANGE EXISTS FOR.

    The agent scored near zero; an independent judge says the text meets the
    bar. The evidence survives, KEEPING its low ``quality_score`` so it is
    still heavily down-weighted in ``scrutiny_weight._item_masses``, the
    supporting-source count and the counting posterior.

    PINS: attenuation, not destruction. Reverting to the unconditional
    ``quality_score < 0.10 -> invalidate`` fails here.
    """
    stub_llm_judge((92, 5, 3), reasoning="Contains a full methods section.")
    ev = _evidence()
    await _op(repo, overrides=_LOW_QUALITY_AGENT)._score_evidence(ev)

    assert ev.invalidated is False
    assert ev.quality_score is not None and ev.quality_score < 0.10, (
        "the low score must SURVIVE — the evidence is attenuated, not "
        "rehabilitated"
    )
    assert ev.quality_distribution is not None


async def test_low_quality_evidence_survives_when_the_judge_is_doubtful(
    repo, stub_llm_judge
) -> None:
    """Doubt → retain.

    A 34/33/33 judge cannot commit. ``needs_review`` folds in
    ``doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD``, so the destroy is withheld
    even though the argmax may read 'fails'.

    PINS: a future "simplify — just use the overall label" change would
    destroy the evidence here and fail.
    """
    stub_llm_judge((33, 33, 34))
    ev = _evidence()
    await _op(repo, overrides=_LOW_QUALITY_AGENT)._score_evidence(ev)

    assert ev.quality_doubt is not None and ev.quality_doubt >= 0.5
    assert ev.invalidated is False


async def test_second_opinion_is_exactly_one_call_at_fast_temperature(
    repo, stub_llm_judge
) -> None:
    """One criterion, one call, FAST path — no cost blowup, no panel.

    PINS: mapping the four quality axes onto four ``Criterion`` objects
    would 4x the cost of this tail path AND silently replace the agent
    scorer's deliberate 0.35/0.25/0.25/0.15 blend with an equal-weight
    mean. Passing a model SEQUENCE would switch llm_judge to panel mode:
    K sequential calls and an agreement-poll contract, when what this
    destroy/retain gate needs is reproducibility.
    """
    recorder = stub_llm_judge((2, 3, 95))
    await _op(repo, overrides=_LOW_QUALITY_AGENT)._score_evidence(_evidence())

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["temperature"] == _elicit.FAST_TEMPERATURE
    assert isinstance(call["model"], str)
    assert call["criterion"] == "substantive_content"


async def test_no_second_opinion_on_the_happy_path(repo, stub_llm_judge) -> None:
    """Laziness (P7): the judge fires at the cliff, never on every item.

    PINS: a normal quality score must cost zero extra model calls.
    """
    recorder = stub_llm_judge((90, 5, 5))
    ev = _evidence()
    await _op(repo, overrides=_NORMAL_QUALITY_AGENT)._score_evidence(ev)

    assert ev.invalidated is False
    assert ev.quality_distribution is None, (
        "presence of quality_distribution IS the marker that a second "
        "opinion ran — it must be absent on the happy path"
    )
    assert recorder.calls == []


async def test_empty_content_is_invalidated_without_a_second_opinion(
    repo, stub_llm_judge
) -> None:
    """We do not pay a model call to judge nothing.

    PINS: the judge cannot rescue an empty body, so asking it is pure cost.
    """
    recorder = stub_llm_judge((90, 5, 5))
    ev = _evidence(content="")
    await _op(repo, overrides=_LOW_QUALITY_AGENT)._score_evidence(ev)

    assert ev.invalidated is True
    assert recorder.calls == []
    assert ev.quality_distribution is None


async def test_missing_judge_model_raises_instead_of_silently_invalidating(
    repo, stub_llm_judge
) -> None:
    """No silent fallback: a broken construction crashes.

    An ``agent_runner`` present with no ``judge_model`` is a wiring bug.
    Quietly resuming the old unconditional destroy would hide it behind
    lost evidence — the exact failure this change exists to stop.

    PINS: the raise, and that the evidence is NOT invalidated on the way out.
    """
    stub_llm_judge((2, 3, 95))
    ev = _evidence()
    op = _op(repo, overrides=_LOW_QUALITY_AGENT, judge_model=None)

    with pytest.raises(RuntimeError, match="judge_model"):
        await op._score_evidence(ev)
    assert ev.invalidated is False


def _executable_source(text: str) -> str:
    """Source with comments and string literals stripped.

    A module that merely NAMES the quality signal in a docstring — as three
    modules now do, explaining why they are shaped the way they are — is not
    a module that READS it. Scanning raw text conflates the two, and the
    conflation is not harmless in either direction: it fails on prose, and
    it would pass a module that reads the signal inside an f-string.
    """
    import io
    import tokenize

    kept: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(tok.string)
    return "\n".join(kept)


#: Modules allowed to name the quality second-opinion signal in executable
#: code, and what each is allowed to do with it.
_QUALITY_SIGNAL_PRODUCERS_AND_GATE = {
    "entities/evidence.py",  # the field + its derived properties
    "operations/evidence.py",  # the one gate
    "quality_doubt.py",  # the module that produces it
}
#: Modules that CARRY the signal onto the run's verdict object so a consumer
#: can read it. They may copy it; they may not branch on it — which the
#: companion test below enforces separately.
_QUALITY_SIGNAL_CARRIERS = {
    "entities/verdict.py",  # EvidenceRef.quality_doubt — the field
    "confidence.py",  # the pure copy that fills it
}


def test_quality_doubt_is_read_by_nothing_but_the_invalidation_gate() -> None:
    """PARTIAL-INSTRUMENTATION GUARD, asserted rather than documented.

    Only agent-scored (Path 2) evidence that hit the cliff ever carries
    ``quality_distribution``. Path 1 (bibliometric) and Path 3
    (gatherer-supplied) never do. Any GLOBAL gate reading this signal would
    therefore penalise exactly the instrumented subset and nothing else — a
    systematic bias that looks like a quality improvement.

    PINS: the only non-entity, non-test reader stays
    ``operations/evidence.py``, and the only other modules that may name it
    at all are the two that carry it onto the verdict object for a reader.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent
    allowed = _QUALITY_SIGNAL_PRODUCERS_AND_GATE | _QUALITY_SIGNAL_CARRIERS
    offenders: list[str] = []
    for py in sorted(src.rglob("*.py")):
        rel = py.relative_to(src).as_posix()
        if rel.startswith("tests/") or rel in allowed:
            continue
        if "quality_doubt" in _executable_source(py.read_text()):
            offenders.append(rel)
        elif "quality_distribution" in _executable_source(py.read_text()):
            offenders.append(rel)
    assert not offenders, (
        "These modules read the quality second-opinion signal, which only "
        "Path-2 evidence ever carries. Wiring it into a broader gate "
        "silently penalises the instrumented subset: " + ", ".join(offenders)
    )


def test_the_verdict_object_carries_quality_doubt_but_never_branches_on_it() -> None:
    """THE NEW EXEMPTION, BOUNDED.

    ``EvidenceRef.quality_doubt`` exists so a consumer of the run's verdict
    object can SEE how sure the quality judge was. That is a data-visibility
    change, and it must not become a decision: the moment ``confidence`` or
    ``entities/verdict`` compares this partially-instrumented signal against
    anything, the systematic bias the test above exists to prevent is back —
    just one indirection further away, on the object every renderer reads.

    PINS: the carriers may only COPY the signal. A comparison, a boolean
    condition, or a threshold read on it fails here.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent
    # Every AST node kind that turns a value into a DECISION.
    deciding = (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.If, ast.IfExp, ast.While)
    offenders: list[str] = []
    for rel in sorted(_QUALITY_SIGNAL_CARRIERS):
        source = (src / rel).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, deciding):
                continue
            # ``test`` / operand subtrees only — an ``if`` BODY that merely
            # assigns the field is fine; it is the CONDITION that must be
            # free of it.
            subtrees = (
                [node.test]
                if isinstance(node, (ast.If, ast.IfExp, ast.While))
                else [node]
            )
            for sub in subtrees:
                names = {
                    n.attr if isinstance(n, ast.Attribute) else n.id
                    for n in ast.walk(sub)
                    if isinstance(n, (ast.Attribute, ast.Name))
                }
                if "quality_doubt" in names or "quality_distribution" in names:
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "A carrier module BRANCHES on the quality second-opinion signal, "
        "which only Path-2 evidence carries — that is the systematic bias "
        "the companion test exists to prevent, one indirection further "
        "away: " + ", ".join(sorted(set(offenders)))
    )
