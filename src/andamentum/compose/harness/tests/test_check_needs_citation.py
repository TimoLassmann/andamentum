"""The decision-model pre-gate (the decision endpoint is stubbed)."""

from __future__ import annotations

import pytest

from andamentum.compose.harness.checks import needs_citation
from andamentum.compose.harness.checks.needs_citation import (
    ADVISORY_THRESHOLD,
    NeedsCitationCheck,
    measured_precision,
    threshold_for_run,
)
from andamentum.compose.harness.kernel import (
    CheckContext,
    ProjectStore,
    select,
)
from andamentum.compose.harness.kernel.checks import effective_tier
from andamentum.core.decisions import CalibrationRecord, DecisionAnswer, calibrate

from ._helpers import field, make_view

CLAIM = "Urinary osmolality is higher in people who drink less water each day."
OWN = "We enrolled participants from three clinics over twelve weeks."


class StubDecide:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.asked: list[str] = []

    async def __call__(self, state, questions, *, model, **_kw):
        assert isinstance(state, str)
        self.asked.append(state)
        p = self.probabilities.get(state, 0.0)
        return {
            "needs_citation": DecisionAnswer(
                choice="yes" if p >= 0.5 else "no",
                probabilities={"yes": p, "no": 1 - p},
                confidence=max(p, 1 - p),
            )
        }


def install(monkeypatch: pytest.MonkeyPatch, stub: StubDecide) -> StubDecide:
    monkeypatch.setattr(needs_citation, "decide", stub)
    return stub


def ctx(**kw) -> CheckContext:
    return CheckContext(
        store=kw.pop("store", None),
        offline=True,
        models={"decision_model": "ollama:nimble:9b"},
        **kw,
    )


def record(pairs: list[tuple[float, bool]]) -> CalibrationRecord:
    return calibrate(pairs, model="ollama:nimble:9b", digest="d", task="needs_citation")


def test_registered_and_model_judged():
    (check,) = select(ids=["needs_citation"])
    assert check.spec.needs == ["decision_model"]
    # advisory without a precision record, declared tier with a good one
    assert effective_tier(check, precision=None) == 2
    assert effective_tier(check, precision=0.95) == check.spec.tier


async def test_flags_an_uncited_claim_at_or_above_the_threshold(monkeypatch):
    stub = install(monkeypatch, StubDecide({CLAIM: 0.93, OWN: 0.2}))
    out = await NeedsCitationCheck().run(make_view((f"{CLAIM} {OWN}", None)), ctx())
    assert stub.asked == [CLAIM, OWN]  # asked one after another
    (d,) = out
    assert d.severity == "info"
    assert d.evidence == [CLAIM]
    assert (d.start, d.end) == (0, len(CLAIM))
    assert d.data["calibrated"] is False
    assert d.data["threshold"] == ADVISORY_THRESHOLD


async def test_a_cited_sentence_is_not_asked(monkeypatch):
    stub = install(monkeypatch, StubDecide({CLAIM: 0.99}))
    view = make_view(
        (f"{CLAIM} (Smith, 2020)", None),
        fields=[
            field("citation", "zotero", display="(Smith, 2020)", pid="p0").model_copy(
                update={"start": len(CLAIM) + 1, "end": len(CLAIM) + 15}
            )
        ],
    )
    assert await NeedsCitationCheck().run(view, ctx()) == []
    assert stub.asked == []


async def test_headings_captions_references_and_short_sentences_are_skipped(
    monkeypatch,
):
    stub = install(monkeypatch, StubDecide({}))
    view = make_view(
        ("Introduction and background", 1),
        ("Short one.", None),
        (CLAIM, None, "Caption"),
        ("References", 1),
        (CLAIM, None),
    )
    assert await NeedsCitationCheck().run(view, ctx()) == []
    assert stub.asked == []


async def test_threshold_comes_from_the_calibration_record(monkeypatch):
    pairs = (
        [(0.95, True)] * 6 + [(0.7, False)] * 3 + [(0.7, True)] * 1 + [(0.2, False)] * 5
    )
    rec = record(pairs)
    threshold, calibrated = threshold_for_run(rec)
    assert calibrated and threshold == 0.95
    assert measured_precision(rec) == 1.0
    install(monkeypatch, StubDecide({CLAIM: 0.8, OWN: 0.97}))
    out = await NeedsCitationCheck().run(
        make_view((f"{CLAIM} {OWN}", None)),
        ctx(calibrations={"needs_citation": rec}),
    )
    assert [d.evidence for d in out] == [[OWN]]
    assert out[0].data["calibrated"] is True


def test_no_threshold_reaching_the_precision_stays_advisory():
    rec = record([(0.9, True), (0.9, False)] * 6)
    assert threshold_for_run(rec) == (ADVISORY_THRESHOLD, False)
    assert measured_precision(rec) is None
    assert threshold_for_run(None) == (ADVISORY_THRESHOLD, False)


async def test_answers_are_cached_in_the_project(monkeypatch, tmp_path):
    store = ProjectStore(tmp_path / "p.harness")
    stub = install(monkeypatch, StubDecide({CLAIM: 0.9}))
    view = make_view((CLAIM, None))
    first = await NeedsCitationCheck().run(view, ctx(store=store))
    again = await NeedsCitationCheck().run(view, ctx(store=store))
    assert first == again and len(first) == 1
    assert stub.asked == [CLAIM]


async def test_the_sentence_limit_is_reported(monkeypatch):
    monkeypatch.setattr(needs_citation, "MAX_SENTENCES", 1)
    install(monkeypatch, StubDecide({}))
    out = await NeedsCitationCheck().run(make_view((f"{CLAIM} {OWN}", None)), ctx())
    (d,) = out
    assert d.paragraph_id is None and "1 sentence(s)" in d.message


async def test_needs_the_decision_model():
    with pytest.raises(ValueError, match="--decision-model"):
        await NeedsCitationCheck().run(
            make_view((CLAIM, None)), CheckContext(store=None, offline=True)
        )
