"""Tests for core.decisions: the wire shape, limits and the one-shot helpers (stubbed HTTP)."""

from __future__ import annotations

import json

import httpx
import pytest

from andamentum.core import decisions
from andamentum.core.decisions import (
    DecisionModelUnavailable,
    DecisionQuestion,
    DecisionRequestRejected,
    DecisionRequestTooLarge,
    Question,
    decide,
    model_digest,
    model_name,
    question_hash,
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


async def test_decide_http_errors_are_decision_errors() -> None:
    with pytest.raises(DecisionModelUnavailable, match="ollama pull"):
        await decide(
            "s", QUESTIONS, model="ollama:nimble", transport=_transport({}, status=404)
        )
    with pytest.raises(DecisionModelUnavailable, match="500"):
        await decide(
            "s", QUESTIONS, model="ollama:nimble", transport=_transport({}, status=500)
        )
    with pytest.raises(DecisionRequestRejected) as rejected:
        await decide(
            "s",
            QUESTIONS,
            model="ollama:nimble",
            transport=_transport({"error": "input too long"}, status=400),
        )
    assert rejected.value.status == 400 and "too long" in str(rejected.value)


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


def test_question_options_descriptions_and_criteria() -> None:
    described = Question(
        type="choice", instructions="q", options={"a": "the first", "b": "the second"}
    )
    assert described.wire()["criteria"] == {"a": "the first", "b": "the second"}
    assert Question(type="choice", instructions="q", options=["a", "b"]).wire()[
        "criteria"
    ] == {"a": "a", "b": "b"}
    noul = Question(
        type="noul", instructions="q", criteria={"true": "it holds", "false": "no"}
    )
    assert noul.wire()["criteria"] == {"true": "it holds", "false": "no"}
    with pytest.raises(ValueError, match="'true' and 'false'"):
        Question(type="noul", instructions="q", criteria={"maybe": "x"})
    with pytest.raises(ValueError, match="takes options"):
        Question(type="choice", instructions="q", options=["a", "b"], criteria={})
    with pytest.raises(ValueError, match="2 to 26"):
        Question(type="choice", instructions="q", options=["only"])
    with pytest.raises(ValueError, match="2 to 26"):
        Question(type="choice", instructions="q", options=[str(i) for i in range(27)])
    with pytest.raises(ValueError, match="a list"):
        Question(type="score", instructions="q", options={"a": "x", "b": "y"})
    assert DecisionQuestion is Question


def test_question_hash_follows_what_the_question_asks() -> None:
    base = Question(type="choice", instructions="q", options={"a": "x", "b": "y"})
    same = Question(type="choice", instructions="q", options={"b": "y", "a": "x"})
    assert question_hash(base) == question_hash(same)  # option order is irrelevant
    for changed in (
        Question(type="choice", instructions="other", options={"a": "x", "b": "y"}),
        Question(type="choice", instructions="q", options={"a": "x", "b": "z"}),
        Question(type="choice", instructions="q", options={"a": "x", "c": "y"}),
    ):
        assert question_hash(changed) != question_hash(base)


def test_model_name_accepts_both_forms_and_refuses_other_providers() -> None:
    assert model_name("ollama:nimble:9b") == "nimble:9b"
    assert model_name("nimble:9b") == "nimble:9b"
    assert model_name("ollama:tev1") == "tev1"
    for bad in ("openai:gpt-5", "anthropic:claude", "ollama:", ""):
        with pytest.raises(ValueError, match="ollama:<name>"):
            model_name(bad)


async def test_noul_answers_carry_p_true() -> None:
    q = {"claim": QUESTIONS["claim"]}
    resp = {"answers": {"claim": {"type": "noul", "noul": 0.9}}}
    out = await decide("s", q, model="ollama:nimble", transport=_transport(resp))
    assert out["claim"].p_true == 0.9


async def test_the_request_limits_are_checked_before_sending() -> None:
    seen: list[httpx.Request] = []
    many = {f"q{i}": QUESTIONS["claim"] for i in range(65)}
    with pytest.raises(DecisionRequestTooLarge, match="65 questions"):
        await decide("s", many, model="nimble", transport=_transport(RESPONSE, seen))
    with pytest.raises(DecisionRequestTooLarge, match="bytes"):
        await decide(
            "x" * 70_000,
            {"claim": QUESTIONS["claim"]},
            model="nimble",
            transport=_transport(RESPONSE, seen),
        )
    assert seen == []


async def test_model_digest() -> None:
    seen: list[httpx.Request] = []
    listing = {
        "models": [
            {"name": "other:1b", "digest": "zzz"},
            {"name": "nimble:9b", "digest": "abc"},
            {"name": "tev1:latest", "digest": "lat"},
        ]
    }
    digest = await model_digest("ollama:nimble:9b", transport=_transport(listing, seen))
    assert digest == "abc"
    assert seen[0].url.path == "/api/tags"
    assert await model_digest("tev1", transport=_transport(listing)) == "lat"
    with pytest.raises(ValueError, match="no digest"):
        await model_digest("ollama:missing", transport=_transport(listing))
    with pytest.raises(ValueError, match="no digest"):
        await model_digest("ollama:nimble:9b", transport=_transport({}))


def test_public_exports() -> None:
    import andamentum.core as core

    for name in (
        "decide",
        "DecisionModel",
        "Question",
        "Answer",
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
