"""SITE B — an independent judge on a BORDERLINE deterministic reading.

``compute_evidence_weight`` is pure and free, and its label is a step
function of three cutoffs. A reading sitting a hair from one of them has a
label that is an artefact of where the cutoff sits, not of the evidence.
This is the one call that asks an independent judge about that case — and
NEVER changes the verdict it comments on.

Every test here is offline: the elicitation seam is monkeypatched via the
shared ``stub_llm_judge`` fixture, so the suite can never reach a live model.
"""

from __future__ import annotations

import pytest

from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.scrutiny import ScrutiniseClaimOperation
from andamentum.epistemic.scrutiny_opinion import (
    SCRUTINY_CRITERIA,
    reading_dissents,
)
from andamentum.epistemic.scrutiny_weight import (
    compute_evidence_weight,
    is_borderline,
)
from andamentum.epistemic.thresholds import (
    SCRUTINY_STRONG_MASS_THRESHOLD,
    SCRUTINY_WEAK_MASS_THRESHOLD,
)
from andamentum.llm_judge import elicit as _elicit

from .conftest import FakeAgentRunner

# ── Fixtures ─────────────────────────────────────────────────────────
#
# Masses are chosen against the named cutoffs, never against a literal, so
# a threshold change moves the fixtures with it.

#: quality × p_supports = the item's supporting mass (corroboration 1).
#: 1.44 against a strong cut of 1.50 → inside the ±20 % band.
_BORDERLINE_QUALITY = 0.8
#: 0.30 against a strong cut of 1.50 and a weak cut of 0.40 → outside both.
_CLEARLY_MODERATE_QUALITY = 1.0


async def _seed(
    repo,
    *,
    quality: float,
    n_items: int = 2,
    dist: list[float] | None = None,
) -> Claim:
    obj = Objective(description="q", phase="planned")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(objective_id=obj.entity_id, statement="X causes Y", scope="adults")
    ids: list[str] = []
    for i in range(n_items):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="openalex",
            source_ref=f"doi:10.1234/{i}",
            extracted_content=f"A finding numbered {i}.",
            extracted=True,
            quality_score=quality,
            cluster_status="representative",
            # Graded, so SITE A's one-hot selector never fires here and
            # these tests isolate SITE B.
            judgment_distribution=dist or [0.90, 0.05, 0.05],
            support_judgment="supports",
        )
        await repo.save(ev)
        ids.append(ev.entity_id)
    claim.evidence_ids = ids
    claim.evidence_count = len(ids)
    await repo.save(claim)
    return claim


def _op(repo, *, runner=None, judge_model: str | None = "test:judge"):
    return ScrutiniseClaimOperation(
        repo,
        runner,
        embedding_model="test-model",
        judge_model=judge_model,
    )


def _work(claim: Claim) -> OperationInput:
    return OperationInput(
        entity_id=claim.entity_id,
        entity_type="claim",
        operation="scrutinise_claim",
    )


# ── The trigger ──────────────────────────────────────────────────────


async def test_a_clearly_non_borderline_reading_makes_zero_calls(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS THE COST BOUND. ``is_borderline`` is pure and free and runs
    BEFORE any model call, so the common case pays nothing.

    A future change that ungates the judge — fires it every pass, or per
    evidence item — fails here.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, quality=_CLEARLY_MODERATE_QUALITY, n_items=1)

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.scrutiny_opinion is None
    assert recorder.calls == []


async def test_a_borderline_reading_takes_exactly_one_fast_call(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS: FAST not PANEL, and exactly ONE criterion.

    A regression to ``model=[m]`` would silently be PANEL — ``_is_panel``
    keys on TYPE, not length — so temperature would become 0.7 and the gate
    would stop being reproducible. A four-criterion list would be a 4x cost
    blowup. Both fail here.
    """
    recorder = stub_llm_judge((60, 30, 10))
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["temperature"] == _elicit.FAST_TEMPERATURE == 0.0
    assert call["model"] == "test:judge"
    assert call["criterion"] == SCRUTINY_CRITERIA[0].name
    assert len(SCRUTINY_CRITERIA) == 1


async def test_the_fixture_really_is_borderline_and_the_other_really_is_not(
    repo,
) -> None:
    """Fixture sanity, asserted rather than assumed — otherwise the two
    tests above could both be passing for the wrong reason."""
    from andamentum.epistemic.gates import eligible_representative_evidence

    borderline = await _seed(repo, quality=_BORDERLINE_QUALITY)
    w = compute_evidence_weight(
        await eligible_representative_evidence(borderline, repo)
    )
    assert is_borderline(w)
    assert w.supporting_mass < SCRUTINY_STRONG_MASS_THRESHOLD

    clear = await _seed(repo, quality=_CLEARLY_MODERATE_QUALITY, n_items=1)
    w2 = compute_evidence_weight(await eligible_representative_evidence(clear, repo))
    assert not is_borderline(w2)
    assert w2.supporting_mass > SCRUTINY_WEAK_MASS_THRESHOLD


def test_is_borderline_is_false_when_there_is_no_directional_mass() -> None:
    """A reading with nothing in it is not a coin flip — there is no body
    of evidence to put to a judge."""
    from andamentum.epistemic.scrutiny_weight import EvidenceWeight

    assert not is_borderline(
        EvidenceWeight(
            label="weak", supporting_mass=0.0, contradicting_mass=0.0, doubt=None
        )
    )


# ── The verdict never moves ──────────────────────────────────────────


@pytest.mark.parametrize(
    "triple, expected_overall",
    [((95, 3, 2), "meets"), ((2, 3, 95), "fails"), ((10, 80, 10), "partial")],
)
async def test_the_opinion_never_changes_the_verdict(
    repo, fake_runner, stub_llm_judge, triple, expected_overall
) -> None:
    """PINS the doubt-is-a-suspicion-raiser doctrine at the claim level.

    The verdict is identical whichever way the judge reads it, and equals
    what the deterministic weight alone produces. A future change that lets
    a disagreeing judge downgrade or upgrade a label fails here — and would
    also silently break ``ConcludeAtCycleCapOperation``'s documented "no
    second, divergent reading" contract, which nothing else would catch.
    """
    from andamentum.epistemic.gates import eligible_representative_evidence

    stub_llm_judge(triple)
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    weight = compute_evidence_weight(
        await eligible_representative_evidence(reloaded, repo)
    )
    assert reloaded.scrutiny_verdict == ScrutiniseClaimOperation._verdict_from_weight(
        weight.label
    )
    assert reloaded.scrutiny_opinion is not None
    assert reloaded.scrutiny_opinion.overall == expected_overall


async def test_scrutiny_and_the_cap_read_the_same_label_after_a_disagreement(
    repo, fake_runner, stub_llm_judge
) -> None:
    """THE HAZARD A VERDICT-CHANGING DESIGN WOULD OPEN.

    ``ConcludeAtCycleCapOperation`` re-runs ``compute_evidence_weight`` "so
    the cap does not introduce a second, divergent reading". If the judge
    could move scrutiny's label, scrutiny and the cap would read the same
    claim differently — a divergence visible only at the cycle cap.
    """
    from andamentum.epistemic.gates import eligible_representative_evidence

    stub_llm_judge((2, 3, 95))  # the judge flatly disagrees
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    cap_reading = compute_evidence_weight(
        await eligible_representative_evidence(reloaded, repo)
    )
    assert reloaded.scrutiny_opinion is not None
    assert reloaded.scrutiny_opinion.dissents is True
    assert (
        ScrutiniseClaimOperation._verdict_from_weight(cap_reading.label)
        == reloaded.scrutiny_verdict
    )


async def test_scrutiny_doubt_still_equals_the_one_aggregator_exactly(
    repo, fake_runner, stub_llm_judge
) -> None:
    """SINGLE PROVENANCE. The judge's own doubt lands on
    ``scrutiny_opinion`` and nowhere else; ``scrutiny_doubt`` remains
    literally ``compute_evidence_weight(eligible).doubt``.

    PINS: a "helpful" max-pool of the two into ``scrutiny_doubt`` would
    silently widen that field's meaning — and, because
    ``ConcludeAtCycleCapOperation`` calls the same function, the cap's too.
    """
    from andamentum.epistemic.gates import eligible_representative_evidence

    stub_llm_judge((34, 33, 33))  # near-uniform: the judge's doubt is ~1.0
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    expected = compute_evidence_weight(
        await eligible_representative_evidence(reloaded, repo)
    ).doubt
    assert expected is not None
    assert reloaded.scrutiny_doubt == pytest.approx(expected)
    assert reloaded.scrutiny_opinion is not None
    assert reloaded.scrutiny_opinion.doubt > reloaded.scrutiny_doubt


async def test_the_weight_is_still_computed_exactly_once_with_a_judge_wired(
    repo, fake_runner, stub_llm_judge
) -> None:
    """The pre-existing exactly-once pin, re-run on the live-judge path."""
    from unittest.mock import patch

    from andamentum.epistemic.operations import scrutiny as scrutiny_module

    stub_llm_judge((60, 30, 10))
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    real = scrutiny_module.compute_evidence_weight
    calls: list[int] = []

    def counting(items):
        calls.append(1)
        return real(items)

    with patch.object(scrutiny_module, "compute_evidence_weight", counting):
        await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(calls) == 1


# ── The dissent rule ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "weight_label, overall, expected",
    [
        ("strong", "fails", True),
        ("moderate", "fails", True),
        ("strong", "meets", False),
        ("strong", "partial", False),
        ("weak", "meets", True),
        ("conflicting", "meets", True),
        ("weak", "fails", False),
        ("weak", "partial", False),
        ("conflicting", "partial", False),
    ],
)
def test_the_dissent_rule_is_pure_and_never_fires_on_a_hedge(
    weight_label: str, overall: str, expected: bool
) -> None:
    """PINS: 'partial' NEVER dissents. Treating a hedge as a contradiction
    would fire Gate 0 on every merely-cautious judge, which is the fastest
    way to turn a suspicion-raiser into a cost multiplier."""
    assert reading_dissents(weight_label=weight_label, overall=overall) is expected


# ── Wiring, failure and concurrency ──────────────────────────────────


async def test_missing_judge_model_on_a_borderline_reading_raises(
    repo, fake_runner, stub_llm_judge
) -> None:
    """FAIL LOUD. An agent_runner with no judge_model is a wiring bug, and
    silently skipping the second opinion would hide it behind a verdict that
    looks fully checked.

    PINS: a future 'just skip it quietly' fails here.
    """
    stub_llm_judge((60, 30, 10))
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    with pytest.raises(RuntimeError, match="judge_model"):
        await _op(repo, runner=fake_runner, judge_model=None).execute(_work(claim))


async def test_a_judge_failure_propagates_out_of_scrutiny(
    repo, fake_runner, stub_llm_judge
) -> None:
    """No try/except around the second opinion — a swallowed failure would
    become a verdict that silently never got its second look."""
    stub_llm_judge((60, 30, 10), raises=RuntimeError("judge down"))
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    with pytest.raises(RuntimeError, match="judge down"):
        await _op(repo, runner=fake_runner).execute(_work(claim))


async def test_offline_scrutiny_never_reaches_the_judge(
    repo, stub_llm_judge
) -> None:
    """PINS: the whole existing offline suite stays model-free. A mandatory
    second opinion would break dozens of tests; this one names why it is
    not mandatory."""
    recorder = stub_llm_judge((60, 30, 10))
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    result = await _op(repo, runner=None, judge_model=None).execute(_work(claim))

    assert result.success
    assert recorder.calls == []
    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.scrutiny_verdict is not None
    assert reloaded.scrutiny_opinion is None


async def test_no_elicitation_runs_while_an_issue_call_is_open(
    repo, stub_llm_judge, monkeypatch
) -> None:
    """NEVER TWO CONCURRENT LOCAL-MODEL CALLS.

    ``llm_judge`` does NOT share ``core.AgentRunner``'s semaphore (sized 1
    for Ollama), so an elicitation dropped inside ``_execute_split``'s
    ``asyncio.gather`` would fire concurrent local inferences. A shared
    call-order log makes that structural rather than a comment.

    PINS: moving ``_second_opinion_if_borderline`` inside the gather — or
    ahead of the ``await`` on it — fails here.
    """
    open_issue_calls = 0
    overlaps: list[str] = []

    class _CountingRunner(FakeAgentRunner):
        async def run(self, agent_name: str, **kwargs):
            nonlocal open_issue_calls
            if agent_name == "epistemic_identify_single_issue":
                open_issue_calls += 1
                try:
                    import asyncio

                    await asyncio.sleep(0)  # force a real interleaving point
                    return await super().run(agent_name, **kwargs)
                finally:
                    open_issue_calls -= 1
            return await super().run(agent_name, **kwargs)

    real_stub = stub_llm_judge((60, 30, 10))
    real_fake = _elicit.elicit_criterion_score

    async def _watching(*args, **kwargs):
        if open_issue_calls:
            overlaps.append("elicitation opened while an issue call was open")
        return await real_fake(*args, **kwargs)

    monkeypatch.setattr(_elicit, "elicit_criterion_score", _watching)

    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)
    await _op(repo, runner=_CountingRunner()).execute(_work(claim))

    assert len(real_stub.calls) == 1
    assert overlaps == []


# ── Persistence ──────────────────────────────────────────────────────


async def test_the_opinion_round_trips_through_the_claim_document(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS the to_document/from_document pair — the edit that is easiest to
    half-do, and whose omission is invisible to every in-memory test while
    silently blanking the signal on reload."""
    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, quality=_BORDERLINE_QUALITY)

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.scrutiny_opinion is not None
    assert reloaded.scrutiny_opinion.overall == "fails"
    assert reloaded.scrutiny_opinion.reference == "moderate"
    assert reloaded.scrutiny_opinion.dissents is True
    assert reloaded.scrutiny_opinion.doubt < 0.5


async def test_an_absent_opinion_round_trips_as_none(repo) -> None:
    """None must survive the round trip as None — never as a default-
    constructed opinion, which would read as 'a judge looked and agreed'."""
    claim = await _seed(repo, quality=_CLEARLY_MODERATE_QUALITY, n_items=1)
    await _op(repo, runner=None, judge_model=None).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.scrutiny_opinion is None
    assert reloaded.judgment_dissent_count == 0
