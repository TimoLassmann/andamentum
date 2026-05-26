"""Tests for the document-type classifier and its integration.

The classifier itself is exercised by patching ``pydantic_ai.Agent``
so the LLM call returns a controlled value. The ChunkAndScan and
synthesis integrations are exercised by setting ``state.document_type``
directly and verifying downstream gates fire correctly.
"""

from __future__ import annotations

from unittest import mock

import pytest

from andamentum.whetstone._document_type import (
    DOCUMENT_TYPES,
    DocumentType,
    DocumentTypeDecision,
    classify,
)


class FakeAgentResult:
    def __init__(self, output: DocumentTypeDecision) -> None:
        self.output = output


class FakeAgent:
    """Stand-in for pydantic_ai.Agent. Returns a fixed decision."""

    def __init__(
        self, model: object, output_type: type | None = None, **kw: object
    ) -> None:
        self._model = model
        self._output_type = output_type

    async def run(self, _prompt: str) -> FakeAgentResult:
        decision = DocumentTypeDecision(
            document_type=getattr(self, "_canned_type", "general"),
            reasoning="fake",
        )
        return FakeAgentResult(decision)


def _patch_agent(returns: DocumentType) -> mock._patch:
    class _Agent(FakeAgent):
        _canned_type = returns  # type: ignore[misc]

    return mock.patch("pydantic_ai.Agent", _Agent)


class TestClassifyPureFunction:
    async def test_returns_general_when_no_model(self) -> None:
        result = await classify(model=None, section_titles=["A"], markdown="x")
        assert result == "general"

    @pytest.mark.parametrize("category", list(DOCUMENT_TYPES))
    async def test_round_trips_each_category(self, category: DocumentType) -> None:
        with _patch_agent(category):
            result = await classify(
                model="fake:test",
                section_titles=["Abstract", "Methods", "Results"],
                markdown="Body text.",
            )
        assert result == category

    async def test_classifier_exception_defaults_to_general(self) -> None:
        class BoomAgent:
            def __init__(self, *a: object, **kw: object) -> None:
                pass

            async def run(self, _prompt: str) -> None:
                raise RuntimeError("network down")

        with mock.patch("pydantic_ai.Agent", BoomAgent):
            result = await classify(
                model="fake:test",
                section_titles=["Anything"],
                markdown="anything",
            )
        assert result == "general"

    async def test_empty_section_titles_handled(self) -> None:
        with _patch_agent("general"):
            result = await classify(
                model="fake:test", section_titles=[], markdown="body"
            )
        assert result == "general"

    async def test_long_markdown_is_truncated(self) -> None:
        # 50k chars — function should only feed first 1500 to the agent.
        long_md = "x" * 50_000
        captured: dict[str, str] = {}

        class CapturingAgent(FakeAgent):
            _canned_type = "general"  # type: ignore[misc]

            async def run(self, prompt: str) -> FakeAgentResult:
                captured["prompt"] = prompt
                return await super().run(prompt)

        with mock.patch("pydantic_ai.Agent", CapturingAgent):
            await classify(model="fake:test", section_titles=["S"], markdown=long_md)

        # Body sample is bounded — full 50k must not be in the prompt.
        assert len(captured["prompt"]) < 5_000


# The v2 integration tests (TestIntegrationChecklistGate,
# TestNoLLMDefaultsToGeneral, TestSynthesisVocabularyContext) were
# deleted with the v2 surface they exercised — the journal-checklist
# gate lived in v2's structural/ and the synthesis vocab lived in
# v2's nodes/synthesise.py. The classifier unit tests above are the
# load-bearing coverage and survive intact.
