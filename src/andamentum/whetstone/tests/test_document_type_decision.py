"""The document-type classifier with and without a decision model (offline)."""

from __future__ import annotations

from unittest import mock

import pytest

from andamentum.core.decisions import Answer, Question
from andamentum.whetstone import _document_type as dt

TITLES = ["Abstract", "Methods"]
MARKDOWN = "# A paper\n\n## Abstract\n\nWe measured things.\n\n## Methods\n\nCarefully."


class StubDecisionModel:
    """Stands in for an opened DecisionModel; records what it was asked."""

    def __init__(self, choice: str) -> None:
        self.choice = choice
        self.calls: list[tuple[object, dict[str, Question]]] = []

    async def decide(
        self, state: object, questions: dict[str, Question]
    ) -> dict[str, Answer]:
        self.calls.append((state, questions))
        return {
            key: Answer(
                choice=self.choice, probabilities={self.choice: 0.9}, confidence=0.9
            )
            for key in questions
        }


def test_the_schema_description_is_what_it_was_before_the_decision_model() -> None:
    description = dt.DocumentTypeDecision.model_fields["document_type"].description
    assert description is not None
    assert description.startswith("academic: manuscripts, theses, conference papers")
    assert description.endswith("anything that fits none of the above.")
    assert dt.QUESTION.option_names == list(dt.DOCUMENT_TYPES)


async def test_without_a_decision_model_the_llm_prompt_is_unchanged() -> None:
    prompts: list[str] = []

    class Agent:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def run(self, prompt: str) -> object:
            prompts.append(prompt)
            return mock.Mock(
                output=dt.DocumentTypeDecision(document_type="essay", reasoning="r")
            )

    with mock.patch("pydantic_ai.Agent", Agent):
        got = await dt.classify(model="fake:test", section_titles=TITLES, markdown="B")
    assert got == "essay"
    assert prompts == [dt._build_prompt(TITLES, "B")]


async def test_with_a_decision_model_the_llm_is_not_called_and_its_choice_is_used() -> (
    None
):
    stub = StubDecisionModel("tutorial")
    llm = mock.AsyncMock(return_value="essay")
    with mock.patch.object(dt, "_classify_with_llm", llm):
        got = await dt.classify(
            model="stub:llm",
            section_titles=TITLES,
            markdown=MARKDOWN,
            decision_model=stub,  # type: ignore[arg-type]
        )
    assert got == "tutorial"
    llm.assert_not_called()
    state, questions = stub.calls[0]
    assert state == {"section_titles": TITLES, "body": MARKDOWN}
    assert list(questions.values()) == [dt.QUESTION]


async def test_a_choice_outside_the_six_types_is_an_error() -> None:
    with pytest.raises(ValueError, match="not a document type"):
        await dt.classify(
            model=None,
            section_titles=TITLES,
            markdown=MARKDOWN,
            decision_model=StubDecisionModel("poem"),  # type: ignore[arg-type]
        )
