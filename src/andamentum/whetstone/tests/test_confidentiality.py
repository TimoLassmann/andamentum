"""Tests for the confidentiality-marker tripwire."""

from __future__ import annotations

import pytest

from andamentum.whetstone._confidentiality import (
    ConfidentialityMarkerError,
    check_confidentiality,
)


class TestCheckConfidentiality:
    @pytest.mark.parametrize(
        "text",
        [
            "Manuscript ID: 12345",
            "MS#9876 — please review",
            "Submission ID: ABC-123",
            "CONFIDENTIAL — do not distribute",
            "Confidential manuscript under review",
            "Reviewer instructions follow.",
            "From the Editorial Office of XYZ",
            "DECISION LETTER — accepted with revisions",
            "This manuscript is being considered by Journal X",
            "Please do not share this document outside the review panel.",
        ],
    )
    def test_marker_text_raises(self, text: str) -> None:
        with pytest.raises(ConfidentialityMarkerError):
            check_confidentiality(text)

    def test_marker_text_raises_in_long_document(self) -> None:
        text = "Lorem ipsum dolor sit amet. " * 200 + " Manuscript ID: 999\n"
        with pytest.raises(ConfidentialityMarkerError) as ei:
            check_confidentiality(text)
        assert "Manuscript ID:" in str(ei.value)

    def test_error_includes_context(self) -> None:
        text = "Some preamble.\n\nManuscript ID: 7654\n\nThe text follows..."
        with pytest.raises(ConfidentialityMarkerError) as ei:
            check_confidentiality(text)
        assert ei.value.marker == "Manuscript ID:"
        assert "7654" in ei.value.context

    def test_clean_text_no_raise(self) -> None:
        check_confidentiality(
            "This is a perfectly ordinary manuscript draft about robotics. "
            "It discusses methods, results, and discussion."
        )

    def test_empty_no_raise(self) -> None:
        check_confidentiality("")

    def test_marker_in_word_boundary_only(self) -> None:
        # "submanuscript" should not fire the manuscript-id pattern.
        check_confidentiality("Our submanuscript-id approach to indexing.")

    def test_error_message_mentions_override(self) -> None:
        text = "Editorial Office contacted us."
        with pytest.raises(ConfidentialityMarkerError) as ei:
            check_confidentiality(text)
        assert "--confirm-own-draft" in str(ei.value)
        assert "peer review" in str(ei.value).lower()


class TestNodeIntegration:
    """The HarvestSource node should trigger the tripwire after harvest."""

    async def test_node_refuses_when_marker_present_and_unaffirmed(
        self, tmp_path
    ) -> None:
        from andamentum.whetstone.api import review_document

        manuscript = tmp_path / "draft.md"
        manuscript.write_text(
            "# Methods\n\nManuscript ID: NEJM-2026-001\n\nDetails follow."
        )

        with pytest.raises(ConfidentialityMarkerError):
            await review_document(str(manuscript), model=None)

    async def test_node_proceeds_when_confirm_own_draft_set(
        self, tmp_path
    ) -> None:
        from andamentum.whetstone.api import review_document

        manuscript = tmp_path / "draft.md"
        # Use a doc that triggers a marker but is clearly the user's draft.
        manuscript.write_text(
            "# Methods\n\nThis section explains our Editorial Office "
            "communication tooling design.\n"
        )

        # Run with --no-llm equivalent so we don't need a model. Should NOT raise.
        result = await review_document(
            str(manuscript), model=None, confirm_own_draft=True
        )
        # Successful run produces a ReviewResult.
        assert result is not None
