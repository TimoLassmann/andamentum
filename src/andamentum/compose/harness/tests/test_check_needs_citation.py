"""The decision-model pre-gate (the decision endpoint is stubbed)."""

from __future__ import annotations

import pytest

from andamentum.compose.harness.checks import needs_citation
from andamentum.compose.harness.checks.needs_citation import (
    THRESHOLD,
    NeedsCitationCheck,
)
from andamentum.compose.harness.kernel import (
    CheckContext,
    ProjectStore,
    select,
)
from andamentum.compose.harness.kernel.checks import effective_tier
from andamentum.core.decisions.testing import StubDecisions, noul_answer

from ._helpers import field, make_view

CLAIM = "Urinary osmolality is higher in people who drink less water each day."
OWN = "We enrolled participants from three clinics over twelve weeks."


def server(probabilities: dict[str, float]) -> StubDecisions:
    return StubDecisions(
        lambda state, questions: {
            "needs_citation": noul_answer(probabilities.get(str(state), 0.0))
        }
    )


def asked(stub: StubDecisions) -> list[str]:
    return [str(state) for state, _ in stub.calls]


def ctx(stub: StubDecisions, **kw) -> CheckContext:
    return CheckContext(
        store=kw.pop("store", None),
        offline=True,
        models={"decision_model": stub.model().id},
        decision_model=stub.model(),
        **kw,
    )


def test_registered_and_model_judged():
    (check,) = select(ids=["needs_citation"])
    assert check.spec.needs == ["decision_model"]
    # advisory without a precision record, declared tier with a good one
    assert effective_tier(check, precision=None) == 2
    assert effective_tier(check, precision=0.95) == check.spec.tier


async def test_flags_an_uncited_claim_at_or_above_the_threshold():
    stub = server({CLAIM: 0.93, OWN: 0.2})
    assert THRESHOLD == 0.5
    out = await NeedsCitationCheck().run(make_view((f"{CLAIM} {OWN}", None)), ctx(stub))
    assert asked(stub) == [CLAIM, OWN]  # asked one after another
    (d,) = out
    assert d.severity == "info"
    assert d.evidence == [CLAIM]
    assert (d.start, d.end) == (0, len(CLAIM))
    assert d.data["threshold"] == THRESHOLD


async def test_a_cited_sentence_is_not_asked():
    stub = server({CLAIM: 0.99})
    view = make_view(
        (f"{CLAIM} (Smith, 2020)", None),
        fields=[
            field("citation", "zotero", display="(Smith, 2020)", pid="p0").model_copy(
                update={"start": len(CLAIM) + 1, "end": len(CLAIM) + 15}
            )
        ],
    )
    assert await NeedsCitationCheck().run(view, ctx(stub)) == []
    assert asked(stub) == []


async def test_headings_captions_references_and_short_sentences_are_skipped():
    stub = server({})
    view = make_view(
        ("Introduction and background", 1),
        ("Short one.", None),
        (CLAIM, None, "Caption"),
        ("References", 1),
        (CLAIM, None),
    )
    assert await NeedsCitationCheck().run(view, ctx(stub)) == []
    assert asked(stub) == []


async def test_answers_are_cached_by_the_weights_not_the_name(tmp_path):
    store = ProjectStore(tmp_path / "p.harness")
    stub = server({CLAIM: 0.9})
    view = make_view((CLAIM, None))
    first = await NeedsCitationCheck().run(view, ctx(stub, store=store))
    again = await NeedsCitationCheck().run(view, ctx(stub, store=store))
    assert first == again and len(first) == 1
    assert asked(stub) == [CLAIM]
    # the same name on other weights is a different model: asked again
    moved = CheckContext(
        store=store,
        offline=True,
        decision_model=stub.model(digest="sha256:" + "cd" * 32),
    )
    await NeedsCitationCheck().run(view, moved)
    assert asked(stub) == [CLAIM, CLAIM]


async def test_the_sentence_limit_is_reported(monkeypatch):
    monkeypatch.setattr(needs_citation, "MAX_SENTENCES", 1)
    stub = server({})
    out = await NeedsCitationCheck().run(make_view((f"{CLAIM} {OWN}", None)), ctx(stub))
    (d,) = out
    assert d.paragraph_id is None and "1 sentence(s)" in d.message


async def test_needs_the_decision_model():
    with pytest.raises(ValueError, match="--decision-model"):
        await NeedsCitationCheck().run(
            make_view((CLAIM, None)), CheckContext(store=None, offline=True)
        )
