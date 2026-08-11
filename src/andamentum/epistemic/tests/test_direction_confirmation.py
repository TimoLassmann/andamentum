"""SITE A — an independent judge confirms a ONE-HOT evidence direction.

THE HOLE. ``Claim.scrutiny_doubt`` is a mean of judgment ENTROPIES, and
``thresholds.JUDGMENT_ONE_HOT_THRESHOLD`` records that small models emit a
one-hot histogram 53-82 % of the time. A claim whose judgments are all
one-hot therefore has ``scrutiny_doubt ≈ 0`` and Gate 0 is structurally dead
on it. Re-reading the same distribution cannot help — it is degenerate. Only
a second, independent elicitation can.

Every test here is offline via the shared ``stub_llm_judge`` fixture.
"""

from __future__ import annotations

import pytest

from andamentum.epistemic import direction_opinion as direction_module
from andamentum.epistemic import reproducibility
from andamentum.epistemic.direction_opinion import (
    CONTRADICTS_CRITERION,
    SUPPORTS_CRITERION,
    confirm_direction,
    select_confirmation_candidates,
)
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.scrutiny import ScrutiniseClaimOperation
from andamentum.epistemic.scrutiny_weight import (
    compute_evidence_weight,
    is_borderline,
)
from andamentum.epistemic.thresholds import (
    JUDGMENT_CONFIRMATION_MAX_PER_CLAIM,
    JUDGMENT_CONFIRMATION_MAX_PER_PASS,
)
from andamentum.llm_judge import elicit as _elicit

#: One-hot (top mass ≥ JUDGMENT_ONE_HOT_THRESHOLD) and directional — the
#: exact blind spot. ``quality=1.0`` keeps the resulting weight well clear of
#: every borderline band, so SITE B stays silent and these tests isolate
#: SITE A. ``test_the_fixture_never_triggers_the_claim_level_opinion`` pins
#: that.
_ONE_HOT = [0.98, 0.01, 0.01]
_GRADED = [0.55, 0.30, 0.15]


async def _seed(repo, specs: list[dict]) -> Claim:
    """One claim; one Evidence per spec dict (overrides on the defaults)."""
    obj = Objective(description="q", phase="planned")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(objective_id=obj.entity_id, statement="X causes Y", scope="adults")
    ids: list[str] = []
    for i, spec in enumerate(specs):
        fields: dict = {
            "objective_id": obj.entity_id,
            "source_type": "openalex",
            "source_ref": f"doi:10.1234/{i:03d}",
            "extracted_content": f"A finding numbered {i}.",
            "extracted": True,
            "quality_score": 1.0,
            "cluster_status": "representative",
            "judgment_distribution": list(_ONE_HOT),
            "support_judgment": "supports",
        }
        fields.update(spec)
        ev = Evidence(**fields)
        await repo.save(ev)
        ids.append(ev.entity_id)
    claim.evidence_ids = ids
    claim.evidence_count = len(ids)
    await repo.save(claim)
    return claim


def _op(repo, *, runner=None, judge_model: str | None = "test:judge"):
    return ScrutiniseClaimOperation(
        repo, runner, embedding_model="test-model", judge_model=judge_model
    )


def _work(claim: Claim) -> OperationInput:
    return OperationInput(
        entity_id=claim.entity_id,
        entity_type="claim",
        operation="scrutinise_claim",
    )


async def _rescrutinise(repo, claim: Claim) -> Claim:
    """Force a genuinely fresh pass, the way the Scrutinize node does."""
    reloaded = await repo.get("claim", claim.entity_id)
    reloaded.scrutiny_verdict = None
    reloaded.scrutiny_fingerprint = None
    await repo.save(reloaded)
    return reloaded


# ── The call shape ───────────────────────────────────────────────────


async def test_one_fast_call_per_candidate_with_one_criterion(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS: FAST not PANEL, and exactly ONE criterion per confirmation.

    ``model=[m]`` — a one-element LIST — would silently be PANEL, because
    ``_is_panel`` keys on TYPE not length, so temperature would become 0.7
    and the confirmation would stop being reproducible. A multi-criterion
    list would multiply the cost of every confirmation. Both fail here.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{}, {}])

    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(recorder.calls) == 2
    for call in recorder.calls:
        assert call["temperature"] == _elicit.FAST_TEMPERATURE == 0.0
        assert call["model"] == "test:judge"
        assert call["criterion"] in {
            SUPPORTS_CRITERION.name,
            CONTRADICTS_CRITERION.name,
        }


async def test_the_criterion_follows_the_primary_stance(
    repo, fake_runner, stub_llm_judge
) -> None:
    """The proposition put to the judge is the one the primary judge
    committed to — asking "does it support?" about an item judged
    ``contradicts`` would make a confirming answer read as a dissent."""
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(
        repo,
        [
            {"support_judgment": "supports"},
            {
                "support_judgment": "contradicts",
                "judgment_distribution": [0.01, 0.98, 0.01],
            },
        ],
    )

    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert {c["criterion"] for c in recorder.calls} == {
        SUPPORTS_CRITERION.name,
        CONTRADICTS_CRITERION.name,
    }


async def test_a_non_directional_stance_is_a_loud_caller_bug() -> None:
    """The selector only ever yields directional candidates, so reaching
    this is a bypass — and confirming the wrong proposition silently would
    be worse than crashing."""
    with pytest.raises(ValueError, match="directional stance"):
        await confirm_direction(
            content="text",
            source_ref="ref",
            claim_statement="c",
            claim_scope="s",
            stance="no_bearing",
            model="test:judge",
        )


# ── The bounds (the cost argument, made structural) ──────────────────


async def test_bounded_at_the_per_pass_cap(repo, fake_runner, stub_llm_judge) -> None:
    """PINS THE PER-PASS BOUND with 20 eligible candidates seeded.

    A change that drops the truncation, or that moves the confirmation into
    one of the per-item judging loops, fails here. This is the test that
    makes the unbounded per-item design impossible to land.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{} for _ in range(20)])

    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_PASS == 3


async def test_bounded_at_the_per_claim_cap_across_passes(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS THE LIFETIME BOUND. Four scrutiny passes over 20 fresh
    candidates must still spend at most ``JUDGMENT_CONFIRMATION_MAX_PER_CLAIM``
    calls in total — not ``MAX_PER_PASS × passes``.

    This is the constant that turns "3 per pass × 4 passes = 12" into a flat
    4, and it is the only thing standing between a bounded second opinion
    and a per-claim cost that scales with the number of re-scrutinies.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{} for _ in range(20)])

    for _ in range(4):
        await _op(repo, runner=fake_runner).execute(_work(claim))
        claim = await _rescrutinise(repo, claim)

    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_CLAIM == 4


async def test_an_already_confirmed_item_is_never_re_confirmed(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS the once-per-Evidence lifetime guard.

    With exactly ``MAX_PER_PASS`` candidates, a second pass must cost ZERO —
    every item already carries an opinion. Removing the
    ``direction_opinion is None`` filter is a silent 4x cost regression on
    the arm the paper measures that no other test would catch.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{} for _ in range(JUDGMENT_CONFIRMATION_MAX_PER_PASS)])

    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_PASS

    claim = await _rescrutinise(repo, claim)
    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_PASS


async def test_the_per_claim_budget_counts_items_the_eligible_set_dropped(
    repo, fake_runner, stub_llm_judge
) -> None:
    """The ceiling is counted over ``claim.evidence_ids``, not over the
    currently-eligible subset, so evidence churn cannot reopen the budget.

    Here three confirmed items are then invalidated — dropping them out of
    ``eligible`` entirely. A budget counted over ``eligible`` would read
    ``0 already confirmed`` and hand the claim a full fresh allowance.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{} for _ in range(20)])

    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert len(recorder.calls) == 3

    for eid in claim.evidence_ids:
        ev = await repo.get("evidence", eid)
        if ev.direction_opinion is not None:
            ev.invalidated = True
            await repo.save(ev)

    claim = await _rescrutinise(repo, claim)
    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_CLAIM


# ── The selector (cost bound AND information argument) ───────────────


async def test_fires_only_on_eligible_one_hot_directional_items(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS the selectivity that is BOTH the cost bound and the information
    argument.

    Five items are seeded; exactly ONE is eligible ∧ one-hot ∧ directional.
    The graded item already reports its own doubt (which already reaches
    Gate 0), so paying a call to hear "yes, unsure" buys nothing; the
    one-hot ``no_bearing`` item never moved the posterior; the invalidated
    and corroborative items are not part of the reading at all.

    The naive "confirm everything" design fails here and at the bounds.

    This mixed pool happens to land the claim-level weight inside the
    borderline band, so SITE B fires too; the assertion filters on the
    per-item criteria rather than pretending it does not.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(
        repo,
        [
            {  # graded — its entropy is already informative
                "judgment_distribution": list(_GRADED),
                "extracted_content": "GRADED",
            },
            {  # one-hot but no_bearing — moved no direction
                "judgment_distribution": [0.01, 0.01, 0.98],
                "support_judgment": "no_bearing",
                "extracted_content": "NO_BEARING",
            },
            {"invalidated": True, "extracted_content": "INVALIDATED"},
            {"cluster_status": "corroborative", "extracted_content": "CORROBORATIVE"},
            {"extracted_content": "THE_ELIGIBLE_ONE"},
        ],
    )

    await _op(repo, runner=fake_runner).execute(_work(claim))

    per_item = [
        c
        for c in recorder.calls
        if c["criterion"] in {SUPPORTS_CRITERION.name, CONTRADICTS_CRITERION.name}
    ]
    assert len(per_item) == 1
    assert per_item[0]["output"] == "THE_ELIGIBLE_ONE"


def test_the_selector_shares_reproducibilitys_one_hot_predicate() -> None:
    """ONE selectivity definition, two consumers.

    A copy-pasted near-duplicate that later drifts from reproducibility's is
    the failure mode; identity is the strongest available assertion against
    it.
    """
    assert (
        direction_module.warrants_reproducibility_check
        is reproducibility.warrants_reproducibility_check
    )


async def test_candidates_are_ranked_by_directional_mass_with_a_stable_tiebreak(
    repo,
) -> None:
    """We spend the budget on the items that most moved the verdict, and the
    same evidence set always yields the same spend across re-runs.

    Pure selector test — no runner, no model call.
    """
    claim = await _seed(
        repo,
        [
            {"quality_score": 0.1, "source_ref": "doi:low"},
            {"quality_score": 0.9, "source_ref": "doi:high"},
            {"quality_score": 0.5, "source_ref": "doi:b-tie"},
            {"quality_score": 0.5, "source_ref": "doi:a-tie"},
        ],
    )
    evidence = [await repo.get("evidence", eid) for eid in claim.evidence_ids]

    picked = select_confirmation_candidates(
        evidence, per_pass_limit=3, remaining_budget=99
    )
    assert [e.source_ref for e in picked] == ["doi:high", "doi:a-tie", "doi:b-tie"]

    again = select_confirmation_candidates(
        list(reversed(evidence)), per_pass_limit=3, remaining_budget=99
    )
    assert [e.source_ref for e in again] == [e.source_ref for e in picked]


async def test_the_selector_respects_the_smaller_of_the_two_bounds(repo) -> None:
    claim = await _seed(repo, [{} for _ in range(10)])
    evidence = [await repo.get("evidence", eid) for eid in claim.evidence_ids]

    assert len(select_confirmation_candidates(
        evidence, per_pass_limit=3, remaining_budget=1
    )) == 1
    assert len(select_confirmation_candidates(
        evidence, per_pass_limit=1, remaining_budget=3
    )) == 1
    assert select_confirmation_candidates(
        evidence, per_pass_limit=3, remaining_budget=0
    ) == []
    assert select_confirmation_candidates(
        evidence, per_pass_limit=3, remaining_budget=-1
    ) == []


# ── It never changes a verdict ───────────────────────────────────────


async def test_a_confident_dissent_changes_no_judgment_and_no_label(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS: doubt is a suspicion-raiser only.

    The judge flatly disagrees with every direction, and the primary
    judgment, its distribution and the weight label are all untouched. A
    future change that lets a dissent flip a verdict fails here.
    """
    from andamentum.epistemic.gates import eligible_representative_evidence

    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, [{}, {}])

    before = compute_evidence_weight(
        await eligible_representative_evidence(claim, repo)
    )
    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    after_items = await eligible_representative_evidence(reloaded, repo)
    for ev in after_items:
        assert ev.support_judgment == "supports"
        assert ev.judgment_distribution == pytest.approx(_ONE_HOT)
        assert ev.direction_opinion is not None
        assert ev.direction_opinion.dissents is True
    assert compute_evidence_weight(after_items).label == before.label
    assert reloaded.judgment_dissent_count == 2


async def test_a_confirming_opinion_records_no_dissent(
    repo, fake_runner, stub_llm_judge
) -> None:
    """An agreeing second opinion must not manufacture an escalation."""
    stub_llm_judge((95, 3, 2))
    claim = await _seed(repo, [{}, {}])

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.judgment_dissent_count == 0
    for eid in reloaded.evidence_ids:
        ev = await repo.get("evidence", eid)
        assert ev.direction_opinion is not None
        assert ev.direction_opinion.dissents is False


async def test_a_hedged_opinion_is_not_a_dissent(
    repo, fake_runner, stub_llm_judge
) -> None:
    """'partial' is a hedge, not a contradiction — counting it would fire
    Gate 0 on every merely-cautious judge."""
    stub_llm_judge((10, 80, 10))
    claim = await _seed(repo, [{}])

    await _op(repo, runner=fake_runner).execute(_work(claim))

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.judgment_dissent_count == 0


async def test_a_dissent_survives_a_later_pass_that_makes_no_new_calls(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS THAT SITE A'S SIGNAL REACHES GATE 0 AT ALL.

    An item is confirmed at most once ever, so a second pass makes no new
    calls for it. If the count described only that pass's call log it would
    write 0 over a real dissent — and because Gate 0 reads the field at
    synthesis time, AFTER the last pass, any rescrutiny would silently erase
    SITE A's entire contribution. The count therefore describes the claim's
    standing evidence, not this pass's calls."""
    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, [{}])
    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert (await repo.get("claim", claim.entity_id)).judgment_dissent_count == 1

    # A fresh pass over items that are all already confirmed: no new
    # confirmations, and the dissenting item is still linked and still
    # load-bearing, so the claim still carries its dissent.
    claim = await _rescrutinise(repo, claim)
    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert (await repo.get("claim", claim.entity_id)).judgment_dissent_count == 1


async def test_the_dissent_count_never_goes_stale_when_evidence_is_unlinked(
    repo, fake_runner, stub_llm_judge
) -> None:
    """The other half of the same rule: the count is rewritten unconditionally
    from the claim's CURRENT evidence, so a dissent whose item has been
    unlinked leaves the count on the very next pass. A count that only ever
    incremented would keep escalating a claim whose evidence was replaced."""
    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, [{}])
    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert (await repo.get("claim", claim.entity_id)).judgment_dissent_count == 1

    # The dissenting evidence is unlinked from the claim.
    claim = await _rescrutinise(repo, claim)
    claim.evidence_ids = []
    await repo.save(claim)
    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert (await repo.get("claim", claim.entity_id)).judgment_dissent_count == 0


async def test_an_offline_pass_does_not_erase_a_standing_dissent(
    repo, fake_runner, stub_llm_judge
) -> None:
    """THE SAME ERASURE, ONE BRANCH OVER.

    A pass with no ``agent_runner`` can make no confirmations — but the
    dissents already on the claim's books still stand, on evidence that is
    still linked and still load-bearing. Returning 0 from the offline branch
    had ``execute`` write over them, and because Gate 0 reads the field at
    synthesis time, that silently deleted SITE A's whole contribution on any
    run that mixed an offline pass in. The tally is a repository read with
    no model call, so there is nothing the offline branch is entitled to
    skip.
    """
    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, [{}])
    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert (await repo.get("claim", claim.entity_id)).judgment_dissent_count == 1

    claim = await _rescrutinise(repo, claim)
    result = await _op(repo, runner=None, judge_model=None).execute(_work(claim))

    assert result.success
    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.judgment_dissent_count == 1
    # And the evidence really does still carry the dissent it was counted
    # from — so this is preservation, not a stale field nobody rewrote.
    ev = await repo.get("evidence", reloaded.evidence_ids[0])
    assert ev.direction_opinion is not None
    assert ev.direction_opinion.dissents is True


# ── The exact shape of the per-claim bound ───────────────────────────


async def test_evidence_added_between_passes_cannot_reopen_the_budget(
    repo, fake_runner, stub_llm_judge
) -> None:
    """The reachable churn case, pinned.

    Investigation appends and extends ``claim.evidence_ids`` between passes.
    A budget counted over the pass's own candidate list would hand every
    growth spurt a fresh allowance; counting over the claim's linked ids
    means the ceiling holds however much evidence arrives.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{} for _ in range(3)])
    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_PASS

    # Three more passes, each bringing 5 brand-new eligible one-hot items.
    for _ in range(3):
        grown = await _seed(repo, [{} for _ in range(5)])
        reloaded = await repo.get("claim", claim.entity_id)
        reloaded.evidence_ids = [*reloaded.evidence_ids, *grown.evidence_ids]
        reloaded.evidence_count = len(reloaded.evidence_ids)
        await repo.save(reloaded)
        claim = await _rescrutinise(repo, reloaded)
        await _op(repo, runner=fake_runner).execute(_work(claim))

    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_CLAIM == 4


async def test_unlinking_a_confirmed_item_releases_its_budget_slot(
    repo, fake_runner, stub_llm_judge
) -> None:
    """THE STATED LIMIT OF THE BOUND, asserted rather than assumed.

    The budget is a read of the claim's PRESENT books, so an id removed from
    ``claim.evidence_ids`` takes its confirmation with it and frees a slot.
    That is unreachable today — nothing outside tests removes ids — but it
    is the difference between "a hard bound" and "a bound under the
    pipeline's current write patterns", and the docstring on
    ``JUDGMENT_CONFIRMATION_MAX_PER_CLAIM`` says so because of this test.
    A future caller that removes ids must revisit the ceiling.
    """
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{} for _ in range(6)])
    await _op(repo, runner=fake_runner).execute(_work(claim))
    assert len(recorder.calls) == JUDGMENT_CONFIRMATION_MAX_PER_PASS

    reloaded = await repo.get("claim", claim.entity_id)
    keep: list[str] = []
    for eid in reloaded.evidence_ids:
        ev = await repo.get("evidence", eid)
        if ev.direction_opinion is None:
            keep.append(eid)
    reloaded.evidence_ids = keep
    reloaded.evidence_count = len(keep)
    await repo.save(reloaded)

    claim = await _rescrutinise(repo, reloaded)
    await _op(repo, runner=fake_runner).execute(_work(claim))

    # The three unlinked confirmations no longer occupy budget, so a full
    # per-pass allowance is spent again — 6, not the 4 a lifetime ledger
    # would have enforced.
    assert len(recorder.calls) == 2 * JUDGMENT_CONFIRMATION_MAX_PER_PASS


# ── Isolation, wiring, failure, concurrency, persistence ─────────────


async def test_the_fixture_never_triggers_the_claim_level_opinion(repo) -> None:
    """Fixture sanity: SITE B must stay silent here, or the call counts
    above would be measuring both mechanisms at once."""
    from andamentum.epistemic.gates import eligible_representative_evidence

    for n in (1, 2, 3, 5, 20):
        claim = await _seed(repo, [{} for _ in range(n)])
        weight = compute_evidence_weight(
            await eligible_representative_evidence(claim, repo)
        )
        assert not is_borderline(weight), f"n={n} is borderline: {weight}"


async def test_missing_judge_model_raises_instead_of_silently_skipping(
    repo, fake_runner, stub_llm_judge
) -> None:
    """FAIL LOUD on a wiring bug. Silently not confirming would leave a
    verdict that looks fully checked and is not."""
    stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{}])

    with pytest.raises(RuntimeError, match="judge_model"):
        await _op(repo, runner=fake_runner, judge_model=None).execute(_work(claim))


async def test_a_judge_failure_propagates(repo, fake_runner, stub_llm_judge) -> None:
    stub_llm_judge((90, 5, 5), raises=RuntimeError("judge down"))
    claim = await _seed(repo, [{}])

    with pytest.raises(RuntimeError, match="judge down"):
        await _op(repo, runner=fake_runner).execute(_work(claim))


async def test_offline_scrutiny_makes_no_confirmation_call(
    repo, stub_llm_judge
) -> None:
    """The whole existing offline suite stays model-free."""
    recorder = stub_llm_judge((90, 5, 5))
    claim = await _seed(repo, [{}, {}])

    result = await _op(repo, runner=None, judge_model=None).execute(_work(claim))

    assert result.success
    assert recorder.calls == []
    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.scrutiny_verdict is not None
    assert reloaded.judgment_dissent_count == 0


async def test_confirmations_are_strictly_sequential(
    repo, fake_runner, stub_llm_judge, monkeypatch
) -> None:
    """NEVER TWO CONCURRENT LOCAL-MODEL CALLS.

    A refactor of the confirmation loop into ``asyncio.gather`` — the shape
    three neighbouring judging loops already use — fails here rather than at
    2am on a local Ollama box.
    """
    import asyncio

    stub_llm_judge((90, 5, 5))
    inner = _elicit.elicit_criterion_score
    in_flight = 0
    overlaps: list[str] = []

    async def _counting(*args, **kwargs):
        nonlocal in_flight
        in_flight += 1
        if in_flight > 1:
            overlaps.append("two elicitations in flight")
        try:
            await asyncio.sleep(0)  # a real interleaving point
            return await inner(*args, **kwargs)
        finally:
            in_flight -= 1

    monkeypatch.setattr(_elicit, "elicit_criterion_score", _counting)
    claim = await _seed(repo, [{}, {}, {}])

    await _op(repo, runner=fake_runner).execute(_work(claim))

    assert overlaps == []


async def test_confirmations_land_before_the_single_weight_computation(
    repo, fake_runner, stub_llm_judge
) -> None:
    """The persisted doubt must belong to the reading that produced the
    verdict, so the confirmations cannot land after it."""
    from unittest.mock import patch

    from andamentum.epistemic.operations import scrutiny as scrutiny_module

    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, [{}])

    real = scrutiny_module.compute_evidence_weight
    seen_confirmed: list[bool] = []

    def counting(items):
        items = list(items)
        seen_confirmed.append(
            any(getattr(e, "direction_opinion", None) is not None for e in items)
        )
        return real(items)

    with patch.object(scrutiny_module, "compute_evidence_weight", counting):
        await _op(repo, runner=fake_runner).execute(_work(claim))

    assert seen_confirmed == [True]


async def test_the_opinion_round_trips_through_evidence_metadata(
    repo, fake_runner, stub_llm_judge
) -> None:
    """PINS persistence. Without the ``_extra_metadata`` / ``_from_metadata``
    pair the field is silently lost on reload — and then the once-per-item
    guard stops working too, so the cost bound quietly fails with it."""
    stub_llm_judge((2, 3, 95))
    claim = await _seed(repo, [{}])

    await _op(repo, runner=fake_runner).execute(_work(claim))

    ev = await repo.get("evidence", claim.evidence_ids[0])
    assert ev.direction_opinion is not None
    assert ev.direction_opinion.reference == "supports"
    assert ev.direction_opinion.overall == "fails"
    assert ev.direction_opinion.dissents is True
    assert ev.direction_opinion.doubt < 0.5


async def test_an_unconfirmed_item_round_trips_as_none(repo) -> None:
    """None survives as None — never as a default opinion, which would read
    as 'a judge looked and agreed'."""
    claim = await _seed(repo, [{}])
    ev = await repo.get("evidence", claim.evidence_ids[0])
    assert ev.direction_opinion is None
