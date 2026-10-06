"""Live checks of the determinism the experiments rely on (deselected: ``ollama``).

Run with ``uv run pytest -m ollama src/andamentum/deep_research/v2/tests/test_live_determinism.py``
while no other local model is busy (local Ollama calls must not overlap). The
decision model is ``nimble:9b``; the state is invented.
"""

from __future__ import annotations

import pytest

from andamentum.core.decisions import DecisionModel

from ..strategies import RELEVANT_KEY, relevance_question
from ..types import SubQuestion

pytestmark = pytest.mark.ollama

DECISION_MODEL = "nimble:9b"
STATE = {
    "title": "Apollo cache halves median latency in read-heavy services",
    "abstract": "Enabling the Apollo cache reduced median request latency from "
    "120 ms to 45 ms in every tested workload.",
    "year": 2024,
    "venue": "Journal of Invented Systems",
}


async def test_the_decision_model_answers_one_state_the_same_twice() -> None:
    question = relevance_question(
        SubQuestion(id="Q1", text="How much does the Apollo cache reduce latency?")
    )
    model = await DecisionModel.open(DECISION_MODEL)
    try:
        first = await model.decide_with_usage(STATE, {RELEVANT_KEY: question})
        second = await model.decide_with_usage(STATE, {RELEVANT_KEY: question})
    finally:
        await model.aclose()
    assert first.answers[RELEVANT_KEY].p_true is not None
    assert first.answers[RELEVANT_KEY].p_true == second.answers[RELEVANT_KEY].p_true
    assert (
        first.answers[RELEVANT_KEY].probabilities
        == second.answers[RELEVANT_KEY].probabilities
    )
