"""Tests for core.decisions: wire shape (stubbed HTTP) and calibration maths."""

from __future__ import annotations

import json

import httpx
import pytest

from andamentum.core import decisions
from andamentum.core.decisions import (
    CalibrationRecord,
    DecisionQuestion,
    calibrate,
    decide,
    model_digest,
    threshold_for,
)


def _transport(
    response: dict, seen: list[httpx.Request] | None = None, status: int = 200
):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=response)

    return httpx.MockTransport(handler)


QUESTIONS = {
    "kind": DecisionQuestion(
        type="choice", instructions="Which kind?", options=["bug", "ask"]
    ),
    "claim": DecisionQuestion(type="noul", instructions="Is this a factual claim?"),
    "urgency": DecisionQuestion(
        type="score", instructions="How urgent?", options=["low", "high"]
    ),
}

RESPONSE = {
    "model": "nimble",
    "answers": {
        "kind": {
            "type": "choice",
            "choice": "bug",
            "probabilities": {"bug": 0.8, "ask": 0.2},
            "confidence": 0.8,
        },
        "claim": {"type": "noul", "noul": 0.25},
        "urgency": {
            "type": "score",
            "score": 0.7,
            "legend": {"0": "low", "1": "high"},
            "probabilities": {"low": 0.3, "high": 0.7},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


async def test_decide_request_and_parse() -> None:
    seen: list[httpx.Request] = []
    out = await decide(
        {"text": "x"},
        QUESTIONS,
        model="ollama:nimble",
        transport=_transport(RESPONSE, seen),
    )
    request = seen[0]
    assert request.url.path == "/v1/systemone"
    body = json.loads(request.content)
    assert body["model"] == "nimble"
    assert body["state"] == {"text": "x"}
    assert body["questions"]["kind"]["criteria"] == {"bug": "bug", "ask": "ask"}
    assert body["questions"]["urgency"]["criteria"] == ["low", "high"]
    assert "criteria" not in body["questions"]["claim"]
    assert out["kind"].choice == "bug" and out["kind"].probabilities["bug"] == 0.8
    assert out["claim"].probabilities == {"yes": 0.25, "no": 0.75}
    assert out["claim"].choice == "no" and out["claim"].confidence == 0.75
    assert out["urgency"].score == 0.7 and out["urgency"].choice is None


async def test_decide_accepts_string_state() -> None:
    seen: list[httpx.Request] = []
    q = {"claim": QUESTIONS["claim"]}
    resp = {"answers": {"claim": {"type": "noul", "noul": 0.9}}}
    await decide(
        "a sentence", q, model="ollama:nimble", transport=_transport(resp, seen)
    )
    assert json.loads(seen[0].content)["state"] == "a sentence"


async def test_decide_rejects_non_ollama_model() -> None:
    with pytest.raises(ValueError, match="ollama:<name>"):
        await decide("s", QUESTIONS, model="openai:gpt", transport=_transport(RESPONSE))


async def test_decide_rejects_empty_questions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await decide("s", {}, model="ollama:nimble", transport=_transport(RESPONSE))


async def test_decide_http_error_raises() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        await decide(
            "s", QUESTIONS, model="ollama:nimble", transport=_transport({}, status=404)
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["answers"].pop("kind"),
        lambda r: r["answers"]["kind"].update(choice="other"),
        lambda r: r["answers"]["claim"].update(noul=1.5),
        lambda r: r["answers"]["kind"].update(confidence="high"),
        lambda r: r["answers"]["urgency"].pop("score"),
        lambda r: r.pop("answers"),
    ],
)
async def test_decide_malformed_response_raises(mutate) -> None:
    broken = json.loads(json.dumps(RESPONSE))
    mutate(broken)
    with pytest.raises(ValueError):
        await decide(
            "s", QUESTIONS, model="ollama:nimble", transport=_transport(broken)
        )


def test_question_validation() -> None:
    with pytest.raises(ValueError):
        DecisionQuestion(type="choice", instructions="q")
    with pytest.raises(ValueError):
        DecisionQuestion(type="noul", instructions="q", options=["a"])
    with pytest.raises(ValueError):
        DecisionQuestion(type="choice", instructions="q", options=["a", "a"])


async def test_model_digest() -> None:
    seen: list[httpx.Request] = []
    listing = {
        "models": [
            {"name": "other:1b", "digest": "zzz"},
            {"name": "nimble:9b", "digest": "abc"},
        ]
    }
    digest = await model_digest("ollama:nimble:9b", transport=_transport(listing, seen))
    assert digest == "abc"
    assert seen[0].url.path == "/api/tags"
    with pytest.raises(ValueError, match="no digest"):
        await model_digest("ollama:missing", transport=_transport(listing))
    with pytest.raises(ValueError, match="no digest"):
        await model_digest("ollama:nimble:9b", transport=_transport({}))


# --- calibration -----------------------------------------------------------

PAIRS = [(0.9, True)] * 8 + [(0.9, False)] * 2 + [(0.1, False)] * 8 + [(0.1, True)] * 2


def test_calibrate_ece_and_bins() -> None:
    rec = calibrate(PAIRS, model="ollama:m", digest="d", task="t")
    assert rec.n == 20 and rec.base_rate == 0.5
    assert rec.ece == pytest.approx(0.1)
    top, low = rec.bins[9], rec.bins[1]
    assert (
        top.count == 10 and top.fraction_positive == 0.8 and top.mean_probability == 0.9
    )
    assert low.count == 10 and low.fraction_positive == 0.2
    assert rec.bins[5].count == 0 and rec.bins[5].mean_probability is None


def test_perfectly_calibrated_has_zero_ece() -> None:
    pairs = [(1.0, True)] * 5 + [(0.0, False)] * 5
    assert calibrate(pairs, model="m", digest="d", task="t").ece == 0.0


def test_probability_one_lands_in_last_bin() -> None:
    rec = calibrate([(1.0, True)], model="m", digest="d", task="t")
    assert rec.bins[-1].count == 1


def test_precision_recall_at_thresholds() -> None:
    rec = calibrate(PAIRS, model="m", digest="d", task="t")
    at_half = next(t for t in rec.thresholds if t.threshold == 0.5)
    assert (
        at_half.predicted == 10 and at_half.precision == 0.8 and at_half.recall == 0.8
    )
    assert rec.stats_at(0.95).precision is None


def test_threshold_for() -> None:
    pairs = [(0.2, False)] * 5 + [(0.6, False), (0.6, True)] * 3 + [(0.9, True)] * 6
    assert threshold_for(pairs, precision=0.95) == 0.9
    assert threshold_for(pairs, precision=0.5) == 0.2
    assert threshold_for(pairs, precision=1.0, min_support=7) is None
    with pytest.raises(ValueError):
        threshold_for(pairs, precision=0.0)


def test_calibrate_validates() -> None:
    with pytest.raises(ValueError):
        calibrate([], model="m", digest="d", task="t")
    with pytest.raises(ValueError):
        calibrate([(1.2, True)], model="m", digest="d", task="t")


def test_record_roundtrip(tmp_path) -> None:
    rec = calibrate(PAIRS, model="ollama:m", digest="sha256:x", task="needs_citation")
    path = tmp_path / "rec.json"
    rec.save(path)
    again = CalibrationRecord.load(path)
    assert again == rec and again.digest == "sha256:x"
    assert again.threshold_for(precision=0.8) == rec.threshold_for(precision=0.8)


def test_public_exports() -> None:
    import andamentum.core as core

    for name in (
        "decide",
        "DecisionQuestion",
        "DecisionAnswer",
        "calibrate",
        "CalibrationRecord",
    ):
        assert name in core.__all__
    assert core.decide is decisions.decide


@pytest.mark.ollama
async def test_live_decide_noul() -> None:
    out = await decide(
        "Smoking causes lung cancer.",
        {
            "claim": DecisionQuestion(
                type="noul",
                instructions="Does this state a factual claim needing a citation?",
            )
        },
        model="ollama:nimble:9b",
    )
    assert 0.0 <= out["claim"].probabilities["yes"] <= 1.0
