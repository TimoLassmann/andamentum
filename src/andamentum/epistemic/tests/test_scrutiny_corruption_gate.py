"""A single sampled CATEGORICAL must not be able to destroy evidence.

``ScrutiniseClaimOperation._handle_issues`` used to invalidate a piece of
evidence whenever ``epistemic_identify_single_issue`` returned the
``"evidence_corrupted"`` issue type — one sampled branch of a small model's
classification rules, with no distribution, no entropy and no second reading.
Worse, it destroyed whichever item sorted first in ``claim.evidence_ids``,
not the item the agent had actually been shown.

Both are now closed. The sentinel PROPOSES a destroy against the source it
was raised about, and ``quality_doubt.confirm_corruption`` — the same FAST,
one-criterion, ``llm_judge`` machinery that gates the near-zero-quality
destroy — disposes. A declined destroy becomes a NON-BLOCKING uncertainty
rather than vanishing.

Every test here is offline: the elicitation seam is monkeypatched via the
shared ``stub_llm_judge`` fixture, so the suite can never reach a live model.
``embedding_model`` is left ``None`` throughout so representative selection
takes its deterministic quality-score path and performs no I/O.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from andamentum.epistemic.entities import (
    Claim,
    Evidence,
    Objective,
    UncertaintyType,
)
from andamentum.epistemic.entities.uncertainty import BLOCKING_TYPES
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.scrutiny import ScrutiniseClaimOperation
from andamentum.epistemic.thresholds import (
    SCRUTINY_BORDERLINE_BAND,
    SCRUTINY_STRONG_MASS_THRESHOLD,
    SCRUTINY_WEAK_MASS_THRESHOLD,
)
from andamentum.llm_judge import elicit as _elicit

from .conftest import FakeAgentRunner

_CORRUPT_MARKER = "GARBLED"

#: The supporting probability every seeded item carries. Graded (not one-hot),
#: so SITE A — the one-hot direction confirmation — never selects a candidate.
_P_SUPPORTS = 0.90

#: A total directional mass that is clearly NOT borderline: outside the
#: ±``SCRUTINY_BORDERLINE_BAND`` neighbourhood of both the weak and the strong
#: cutoff, so SITE B — the borderline claim-level second opinion — never
#: fires and every recorded elicitation belongs to the gate under test. The
#: bounds are derived from the named thresholds, so a threshold change is
#: caught here rather than silently re-enabling SITE B.
_TARGET_MASS = 0.9
assert (
    SCRUTINY_WEAK_MASS_THRESHOLD * (1 + SCRUTINY_BORDERLINE_BAND)
    < _TARGET_MASS
    < SCRUTINY_STRONG_MASS_THRESHOLD * (1 - SCRUTINY_BORDERLINE_BAND)
)


class _IssueRunner(FakeAgentRunner):
    """Reports ``evidence_corrupted`` for selected identifier calls only.

    ``FakeAgentRunner``'s overrides are static per agent name, but these
    tests need the sentinel to fire for ONE evidence body and not another —
    that is the whole point of the targeting property. So this stub keys on
    the call's arguments instead:

    * a per-evidence call (no ``focus``) fires when ``marker`` appears in the
      evidence body it was shown;
    * the cross-item contradiction call (``focus`` set) fires only when
      ``on_contradiction`` is True.
    """

    def __init__(
        self,
        *,
        marker: str | None = _CORRUPT_MARKER,
        on_contradiction: bool = False,
        issue_type: str = "evidence_corrupted",
    ) -> None:
        super().__init__()
        self._marker = marker
        self._on_contradiction = on_contradiction
        self._issue_type = issue_type

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        if agent_name != "epistemic_identify_single_issue":
            return await super().run(agent_name, **kwargs)

        body = str(kwargs.get("evidence") or "")
        if kwargs.get("focus"):
            fires = self._on_contradiction
        else:
            fires = self._marker is not None and self._marker in body

        if not fires:
            # Falls through to the shared default: has_issue=False.
            return await super().run(agent_name, **kwargs)

        self.calls.append((agent_name, kwargs))
        return SimpleNamespace(
            has_issue=True,
            description="The extracted text is unreadable.",
            issue_type=self._issue_type,
        )


async def _seed(
    repo,
    *,
    contents: list[str],
) -> tuple[Claim, list[Evidence]]:
    """A claim with one representative Evidence per entry in ``contents``.

    Judgments are GRADED and the per-item quality is scaled so the claim's
    total directional mass lands on ``_TARGET_MASS`` whatever the item count.
    Neither of the two pre-existing scrutiny second opinions (one-hot
    direction confirmation; borderline-weight read) therefore fires, and any
    recorded elicitation belongs to the corruption gate under test.
    """
    obj = Objective(description="q", phase="planned")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # mass = sum over items of quality x p_supports (corroboration_count 1).
    quality = _TARGET_MASS / (_P_SUPPORTS * len(contents))
    assert 0.0 < quality <= 1.0, "fixture would need an out-of-range quality score"

    claim = Claim(objective_id=obj.entity_id, statement="X causes Y", scope="adults")
    items: list[Evidence] = []
    for i, content in enumerate(contents):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="openalex",
            source_ref=f"doi:10.1234/{i}",
            extracted_content=content,
            extracted=True,
            quality_score=quality,
            cluster_status="representative",
            judgment_distribution=[_P_SUPPORTS, 0.05, 0.05],
            support_judgment="supports",
        )
        await repo.save(ev)
        items.append(ev)
    claim.evidence_ids = [ev.entity_id for ev in items]
    claim.evidence_count = len(items)
    await repo.save(claim)
    return claim, items


def _op(repo, *, runner, judge_model: str | None = "test:judge"):
    return ScrutiniseClaimOperation(
        repo,
        runner,
        embedding_model=None,
        judge_model=judge_model,
    )


def _work(claim: Claim) -> OperationInput:
    return OperationInput(
        entity_id=claim.entity_id,
        entity_type="claim",
        operation="scrutinise_claim",
    )


async def _uncertainties(repo, claim: Claim):
    rows = await repo.query("uncertainty", objective_id=claim.objective_id)
    return list(rows)


# ── The destroy still works when it should ───────────────────────────


async def test_corrupted_evidence_is_invalidated_when_the_judge_agrees(
    repo, stub_llm_judge
) -> None:
    """PINS: this change GATES the destroy, it does not disable it.

    MUTATION THAT BREAKS IT: deleting the ``ev.invalidated = True`` branch
    in ``_destroy_or_retain_corrupted``, or making the gate always decline.
    """
    recorder = stub_llm_judge((2, 3, 95), reasoning="Mojibake throughout.")
    claim, items = await _seed(repo, contents=[f"{_CORRUPT_MARKER} �� text"])

    await _op(repo, runner=_IssueRunner()).execute(_work(claim))

    reloaded = await repo.get("evidence", items[0].entity_id)
    assert reloaded.invalidated is True
    assert "independent judge" in (reloaded.invalidation_reason or "")
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["criterion"] == "text_is_intact"


# ── The core behaviour this change exists for ────────────────────────


async def test_corrupted_evidence_survives_when_the_judge_disagrees(
    repo, stub_llm_judge
) -> None:
    """THE POINT OF THE GATE.

    The identifier sampled ``evidence_corrupted``; an independent judge reads
    the text as intact. The evidence survives, and the concern is recorded as
    a NON-BLOCKING uncertainty rather than dropped.

    MUTATION THAT BREAKS IT: reverting to the unconditional
    ``ev.invalidated = True``.
    """
    stub_llm_judge((94, 4, 2), reasoning="Clean, readable abstract.")
    claim, items = await _seed(repo, contents=[f"{_CORRUPT_MARKER} but readable"])

    await _op(repo, runner=_IssueRunner()).execute(_work(claim))

    reloaded = await repo.get("evidence", items[0].entity_id)
    assert reloaded.invalidated is False

    rows = await _uncertainties(repo, claim)
    assert len(rows) == 1
    assert rows[0].uncertainty_type is UncertaintyType.EVIDENCE_GAP
    assert rows[0].uncertainty_type not in BLOCKING_TYPES
    assert "retained, not destroyed" in rows[0].description


async def test_corrupted_evidence_survives_when_the_judge_is_doubtful(
    repo, stub_llm_judge
) -> None:
    """Doubt → retain.

    A 34/33/33 judge cannot commit. ``needs_review`` folds in
    ``doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD``, so the destroy is withheld
    even when the argmax reads 'fails'.

    MUTATION THAT BREAKS IT: dropping ``not opinion.needs_review`` from
    ``QualitySecondOpinion.confidently_fails`` — i.e. "simplify: just use the
    overall label".
    """
    stub_llm_judge((33, 33, 34))
    claim, items = await _seed(repo, contents=[f"{_CORRUPT_MARKER} ambiguous"])

    await _op(repo, runner=_IssueRunner()).execute(_work(claim))

    reloaded = await repo.get("evidence", items[0].entity_id)
    assert reloaded.invalidated is False


# ── Targeting ────────────────────────────────────────────────────────


async def test_the_gate_judges_and_destroys_the_item_the_issue_was_about(
    repo, stub_llm_judge
) -> None:
    """A destroy falls on the source the agent READ, not on the first id.

    The corruption sentinel fires only for the SECOND evidence item; the
    first is clean and sorts ahead of it in ``claim.evidence_ids``.

    MUTATION THAT BREAKS IT: restoring the ``for eid in claim.evidence_ids:
    ... break`` victim search — it invalidates item 0 and judges item 0's
    text, so both assertions below fail.
    """
    recorder = stub_llm_judge((2, 3, 95))
    claim, items = await _seed(
        repo,
        contents=["A clean randomised trial report.", f"{_CORRUPT_MARKER} ��"],
    )

    await _op(repo, runner=_IssueRunner()).execute(_work(claim))

    clean = await repo.get("evidence", items[0].entity_id)
    flagged = await repo.get("evidence", items[1].entity_id)
    assert clean.invalidated is False
    assert flagged.invalidated is True

    assert len(recorder.calls) == 1
    judged = recorder.calls[0]["output"]
    assert _CORRUPT_MARKER in judged
    assert "clean randomised" not in judged


async def test_a_cross_item_contradiction_flag_never_destroys_evidence(
    repo, stub_llm_judge
) -> None:
    """The whole-body call names no victim, so it gets none.

    The contradiction check reads every representative at once. A corruption
    sentinel from it identifies no single source, so there is nothing to
    judge and nothing to destroy — the concern is recorded instead.

    MUTATION THAT BREAKS IT: attributing whole-body issues to an arbitrary
    item (which is exactly what the old code did for every corruption
    issue) — then some evidence is invalidated and a judge call is made.
    """
    recorder = stub_llm_judge((2, 3, 95))
    claim, items = await _seed(
        repo, contents=["First clean finding.", "Second clean finding."]
    )

    await _op(repo, runner=_IssueRunner(marker=None, on_contradiction=True)).execute(
        _work(claim)
    )

    for ev in items:
        reloaded = await repo.get("evidence", ev.entity_id)
        assert reloaded.invalidated is False
    assert recorder.calls == []

    rows = await _uncertainties(repo, claim)
    assert len(rows) == 1
    assert rows[0].uncertainty_type is UncertaintyType.EVIDENCE_GAP
    assert "names no single one" in rows[0].description


# ── Cost, wiring, and the non-corruption path ────────────────────────


async def test_no_judge_call_when_no_issue_is_a_corruption_sentinel(
    repo, stub_llm_judge
) -> None:
    """Laziness (P7): the gate fires at the cliff, never on every pass.

    MUTATION THAT BREAKS IT: hoisting the second opinion out of the
    ``issue_type == "evidence_corrupted"`` branch so it runs for every issue,
    or for every evidence item.
    """
    recorder = stub_llm_judge((2, 3, 95))
    claim, items = await _seed(repo, contents=[f"{_CORRUPT_MARKER} text"])

    await _op(repo, runner=_IssueRunner(issue_type="evidence_gap")).execute(
        _work(claim)
    )

    reloaded = await repo.get("evidence", items[0].entity_id)
    assert reloaded.invalidated is False
    assert recorder.calls == []

    rows = await _uncertainties(repo, claim)
    assert [r.uncertainty_type for r in rows] == [UncertaintyType.EVIDENCE_GAP]


async def test_the_gate_is_one_fast_call_per_sentinel(repo, stub_llm_judge) -> None:
    """One criterion, one call, FAST path — no panel, no cost blowup.

    Two flagged items → exactly two calls, both at ``FAST_TEMPERATURE`` with
    a bare ``str`` model. A model SEQUENCE would switch llm_judge to panel
    mode: K sequential calls and an agreement-poll contract, when what a
    destroy gate needs is reproducibility.

    MUTATION THAT BREAKS IT: passing ``[self.judge_model]`` to
    ``confirm_corruption``, or adding a second ``Criterion`` to
    ``CORRUPTION_CRITERIA``.
    """
    recorder = stub_llm_judge((2, 3, 95))
    claim, _items = await _seed(
        repo, contents=[f"{_CORRUPT_MARKER} one", f"{_CORRUPT_MARKER} two"]
    )

    await _op(repo, runner=_IssueRunner()).execute(_work(claim))

    assert len(recorder.calls) == 2
    for call in recorder.calls:
        assert call["temperature"] == _elicit.FAST_TEMPERATURE
        assert isinstance(call["model"], str)
        assert call["criterion"] == "text_is_intact"


async def test_missing_judge_model_raises_instead_of_destroying_ungated(
    repo, stub_llm_judge
) -> None:
    """No silent fallback: a broken construction crashes.

    An ``agent_runner`` present with no ``judge_model`` is a wiring bug.
    Quietly resuming the ungated destroy would hide it behind lost evidence —
    the exact failure this gate exists to stop.

    MUTATION THAT BREAKS IT: replacing the raise with
    ``if not self.judge_model: ev.invalidated = True`` (or with a silent
    ``return``, which the "and nothing was destroyed" assertion alone would
    not catch — hence the ``pytest.raises``).
    """
    stub_llm_judge((2, 3, 95))
    claim, items = await _seed(repo, contents=[f"{_CORRUPT_MARKER} text"])
    op = _op(repo, runner=_IssueRunner(), judge_model=None)

    with pytest.raises(RuntimeError, match="judge_model"):
        await op.execute(_work(claim))

    reloaded = await repo.get("evidence", items[0].entity_id)
    assert reloaded.invalidated is False


async def test_an_already_invalidated_target_costs_no_model_call(
    repo, stub_llm_judge
) -> None:
    """Nothing to destroy → nothing to judge.

    MUTATION THAT BREAKS IT: dropping the liveness check before the
    ``confirm_corruption`` call — the gate then pays a model call to
    re-destroy something already gone.
    """
    recorder = stub_llm_judge((2, 3, 95))
    claim, items = await _seed(repo, contents=[f"{_CORRUPT_MARKER} text"])
    op = _op(repo, runner=_IssueRunner())

    from andamentum.epistemic.operations.scrutiny import FoundIssue

    ev = await repo.get("evidence", items[0].entity_id)
    ev.invalidated = True
    await repo.save(ev)

    outcome = await op._destroy_or_retain_corrupted(
        claim,
        FoundIssue(
            description="unreadable",
            issue_type="evidence_corrupted",
            reversal_test=False,
            evidence_id=items[0].entity_id,
        ),
    )

    assert outcome.destroyed is False
    assert recorder.calls == []
