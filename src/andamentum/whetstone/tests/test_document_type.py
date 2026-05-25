"""Tests for the document-type classifier and its integration.

The classifier itself is exercised by patching ``pydantic_ai.Agent``
so the LLM call returns a controlled value. The ChunkAndScan and
synthesis integrations are exercised by setting ``state.document_type``
directly and verifying downstream gates fire correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from andamentum.whetstone._document_type import (
    DOCUMENT_TYPES,
    DocumentType,
    DocumentTypeDecision,
    classify,
)
from andamentum.whetstone.api import review_document


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


# ── Integration: ChunkAndScan gates the journal checklist ────────────────


def _write_manuscript(tmp_path: Path, *, with_markers: bool) -> Path:
    """Write a small fake manuscript that triggers the journal checklist
    when the gate is open. Without statements → multiple checklist findings.
    """
    p = tmp_path / "draft.md"
    body = "# Title\n\n## Introduction\n\nBackground stuff.\n\n## Methods\n\nMethod stuff.\n"
    if with_markers:
        body += (
            "\n## Conflict of interest\n\nNone declared.\n"
            "\n## Data availability\n\nData available on request.\n"
            "\n## Ethics\n\nApproved.\n"
            "\n## Keywords\n\nfoo, bar, baz, qux\n"
        )
    p.write_text(body)
    return p


class TestIntegrationChecklistGate:
    async def test_general_skips_journal_checklist(self, tmp_path: Path) -> None:
        """When document_type is 'general', CoI / data / ethics findings
        do not appear in deterministic_findings."""
        manuscript = _write_manuscript(tmp_path, with_markers=False)

        result = await review_document(
            str(manuscript), model=None, document_type="general"
        )

        titles = " ".join(f.title for f in result.deterministic_findings).lower()
        # The journal-checklist titles must not appear.
        assert "conflict" not in titles
        assert "data availability" not in titles
        assert "ethics" not in titles

    async def test_academic_runs_journal_checklist(self, tmp_path: Path) -> None:
        """When document_type is 'academic', the checklist fires and
        emits findings for the missing required statements."""
        manuscript = _write_manuscript(tmp_path, with_markers=False)

        result = await review_document(
            str(manuscript), model=None, document_type="academic"
        )

        titles = " ".join(f.title for f in result.deterministic_findings).lower()
        # At least one of the journal-checklist findings should fire.
        assert (
            "conflict" in titles
            or "data" in titles
            or "ethics" in titles
            or "abstract" in titles
        ), f"expected journal checklist to fire; got titles: {titles!r}"

    async def test_external_communication_skips_journal_checklist(
        self, tmp_path: Path
    ) -> None:
        manuscript = _write_manuscript(tmp_path, with_markers=False)

        result = await review_document(
            str(manuscript),
            model=None,
            document_type="external_communication",
        )

        titles = " ".join(f.title for f in result.deterministic_findings).lower()
        assert "conflict" not in titles
        assert "data availability" not in titles
        assert "ethics" not in titles


# ── Integration: --no-llm with auto defaults to general ─────────────────


class TestNoLLMDefaultsToGeneral:
    async def test_auto_with_no_model_defaults_to_general(self, tmp_path: Path) -> None:
        """--no-llm + document_type='auto' → classifier returns 'general'.
        Verified by checking the journal checklist did NOT fire."""
        manuscript = _write_manuscript(tmp_path, with_markers=False)

        result = await review_document(
            str(manuscript), model=None, document_type="auto"
        )

        titles = " ".join(f.title for f in result.deterministic_findings).lower()
        assert "conflict" not in titles
        assert "data availability" not in titles


# ── Integration: synthesis injects document-type context ────────────────


class TestSynthesisVocabularyContext:
    """The synthesise node prepends a document-type-aware paragraph to
    its prompt. We verify the per-category vocabulary strings exist
    and are distinct, since the user-visible behaviour is downstream
    LLM prose that we don't unit-test directly.
    """

    def test_vocab_table_has_all_three_categories(self) -> None:
        from andamentum.whetstone.nodes.synthesise import _DOC_TYPE_VOCAB

        assert set(_DOC_TYPE_VOCAB.keys()) == set(DOCUMENT_TYPES)
        # Each entry is distinct, non-empty, and uses appropriate vocab.
        academic = _DOC_TYPE_VOCAB["academic"]
        external = _DOC_TYPE_VOCAB["external_communication"]
        general = _DOC_TYPE_VOCAB["general"]
        assert "manuscript" in academic
        assert "post" in external or "article" in external
        # General should NOT use academic-specific vocabulary in its body.
        assert "manuscript" not in general.lower()

    def test_document_type_context_picks_correct_entry(self) -> None:
        from andamentum.whetstone.nodes.synthesise import _document_type_context

        assert "manuscript" in _document_type_context("academic")
        assert "audience" in _document_type_context("external_communication")
        assert "essay" in _document_type_context("essay")
        assert "tutorial" in _document_type_context("tutorial")
        assert "scene" in _document_type_context("creative")
        assert "neutral" in _document_type_context("general").lower()
