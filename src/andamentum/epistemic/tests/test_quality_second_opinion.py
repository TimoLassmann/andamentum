"""A single sampled point value must not be able to destroy evidence.

``ExtractEvidenceOperation._score_evidence`` Path 2 blends four agent-produced
floats into one quality score and, below 0.10, invalidated the evidence
outright. That is a categorical cliff driven by one sample: an unlucky draw
erased real evidence, and nothing downstream could tell erasure from absence.

The invalidation is now gated by an ``llm_judge`` second opinion. These tests
pin the gate's five behaviours — destroy when confidently confirmed, retain on
disagreement, retain on doubt, never fire on the happy path, and never
silently degrade when the wiring is missing.

The structural tests at the bottom cover BOTH destroy gates: this one and the
``evidence_corrupted`` gate in ``operations/scrutiny.py`` (whose behavioural
tests live in ``test_scrutiny_corruption_gate.py``). They pin the two to one
decision rule, and pin the scrutiny gate to calling the shared driver without
reading the partially-instrumented ``quality_distribution`` field.

Every test runs offline through the shared ``stub_llm_judge`` fixture, which
patches ``llm_judge.elicit.elicit_criterion_score``.
"""

from __future__ import annotations

import ast

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


def _evidence(
    content: str = "A randomised trial of 4200 adults reported ...",
) -> Evidence:
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
        "the low score must SURVIVE — the evidence is attenuated, not rehabilitated"
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
    "operations/evidence.py",  # destroy gate 1 — near-zero quality score
    # Destroy gate 2 — the ``evidence_corrupted`` sentinel. It IMPORTS the
    # driver from ``quality_doubt`` and nothing else: it never reads
    # ``Evidence.quality_distribution`` or ``Evidence.quality_doubt``, which
    # only Path-2 evidence carries. Enforced, not asserted in prose, by
    # ``test_the_second_destroy_gate_calls_the_driver_but_never_reads_the_field``.
    "operations/scrutiny.py",
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

    PINS: the only non-entity, non-test readers stay the two DESTROY GATES,
    and the only other modules that may name the signal at all are the two
    that carry it onto the verdict object for a reader.
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


def test_the_second_destroy_gate_calls_the_driver_but_never_reads_the_field() -> None:
    """THE EXEMPTION THAT LETS SCRUTINY ONTO THE ALLOWLIST, BOUNDED.

    ``operations/scrutiny.py`` names ``quality_doubt`` for exactly one
    reason: it imports ``confirm_corruption``, the shared destroy-gate
    driver. It must NOT grow a read of ``Evidence.quality_distribution`` or
    ``Evidence.quality_doubt`` — those only Path-2 evidence ever carries, and
    a scrutiny-time gate reading them would penalise exactly the instrumented
    subset, which is the systematic bias the companion test above exists to
    prevent.

    PINS both halves: the import must be there (a re-implemented,
    parallel second opinion fails), and no read of the partially-instrumented
    signal may appear (a "while we're here, also check the quality doubt"
    change fails).
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent
    tree = ast.parse((src / "operations" / "scrutiny.py").read_text())

    imports_the_driver = any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("quality_doubt")
        and any(alias.name == "confirm_corruption" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_the_driver, (
        "operations/scrutiny.py must gate its evidence_corrupted destroy "
        "through quality_doubt.confirm_corruption — one destroy-gate "
        "mechanism, not two."
    )

    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    leaked = names & {"quality_doubt", "quality_distribution"}
    assert not leaked, (
        "operations/scrutiny.py reads the partially-instrumented quality "
        "second-opinion signal: " + ", ".join(sorted(leaked))
    )


def _decision_rule(expr: ast.expr, receiver: str) -> str:
    """``expr`` unparsed with ``receiver.`` blanked, for cross-file comparison."""
    return ast.unparse(expr).replace(f"{receiver}.", "<opinion>.")


def test_both_destroy_gates_use_one_decision_rule() -> None:
    """ONE RULE FOR "MAY I DESTROY THIS EVIDENCE", pinned across two files.

    ``QualitySecondOpinion.confidently_fails`` is the rule: a confident
    ``fails``, with ``needs_review`` folding in llm_judge's own doubt
    threshold so neither gate invents a numeric cutoff. Gate 2 (scrutiny)
    calls the property; gate 1 (``operations/evidence.py``) still spells the
    same predicate inline.

    PINS them equal. Loosening either — dropping ``needs_review`` on one
    side, adding a bespoke ``doubt < 0.3`` on the other — makes the two
    unparsed expressions differ and fails here. A reader may then unify them
    on purpose; they cannot drift by accident.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent

    prop_tree = ast.parse((src / "quality_doubt.py").read_text())
    prop_rules = [
        _decision_rule(node.value, "self")
        for fn in ast.walk(prop_tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "confidently_fails"
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert len(prop_rules) == 1, prop_rules

    gate1_tree = ast.parse((src / "operations" / "evidence.py").read_text())
    gate1_rules = [
        _decision_rule(node.test, "opinion")
        for fn in ast.walk(gate1_tree)
        if isinstance(fn, ast.AsyncFunctionDef)
        and fn.name == "_confirm_or_retain_low_quality"
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and "overall" in ast.unparse(node.test)
    ]
    assert len(gate1_rules) == 1, gate1_rules

    assert prop_rules[0] == gate1_rules[0], (
        "the two TMS destroy gates no longer apply the same rule:\n"
        f"  confidently_fails: {prop_rules[0]}\n"
        f"  operations/evidence.py: {gate1_rules[0]}"
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
