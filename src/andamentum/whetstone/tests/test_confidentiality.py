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

    @pytest.mark.parametrize(
        "text,expected_marker",
        [
            ("Grant panel review notes follow.", "Funding-panel review"),
            ("Funding panel assessment for round 2026.", "Funding-panel review"),
            (
                "Assessor comments must remain confidential.",
                "Assessor report / comments",
            ),
            ("Assessor report for application 12345.", "Assessor report / comments"),
            (
                "Peer review of grant application 5R01CA000000.",
                "Peer review of grant/application",
            ),
            (
                "This peer review of proposal X is confidential.",
                "Peer review of grant/application",
            ),
        ],
    )
    def test_grant_review_act_markers_raise(
        self, text: str, expected_marker: str
    ) -> None:
        """Phrases describing the *act of reviewing* someone else's grant fire.

        Note: scheme prefixes (NHMRC APP, ARC DP, NIH RFA-) and role labels
        (Lead CI, Chief Investigator) deliberately do NOT fire — they appear
        in the user's own draft as much as in a reviewer's copy. The
        responsible-use prohibition on grant peer-review lives in
        RESPONSIBLE_USE.md, with --confirm-own-draft as the affirmation.
        """
        with pytest.raises(ConfidentialityMarkerError) as ei:
            check_confidentiality(text)
        assert ei.value.marker == expected_marker

    @pytest.mark.parametrize(
        "text",
        [
            # User's own grant draft — must NOT fire.
            "Lead CI: Dr Example. Coordinates the project.",
            "The Lead investigator is responsible for delivery.",
            "Chief Investigator A: Dr Smith",
            "NHMRC APP1234567 funded the prior work.",
            "Submitted to the Ideas Grant 2026 round.",
            "Investigator Grant scheme: Leadership Level 2.",
            "Funded under a Synergy Grant.",
            "ARC DP240100000 — Discovery Project.",
            "ARC DECRA2024 awarded.",
            "Linkage Project with industry partners.",
            "Discovery Project grant application for round 2026.",
            "Funding opportunity RFA-CA-25-005 covers...",
            "This application responds to PAR-24-100.",
            "Application ID: APP1234567",
            "Grant ID: 5R01CA000000-05",
            # Plain narrative mentions of grants/funding.
            "Our prior work was supported by a research grant.",
            "We thank funding agencies for the support.",
        ],
    )
    def test_grant_authoring_context_does_not_fire(self, text: str) -> None:
        """Grant-authoring context (the user's own draft) must NOT fire.

        Scheme codes, scheme names, and role labels appear in the user's
        own drafts; gating on them would force a reflex --confirm-own-draft
        bypass and defeat the tripwire's purpose.
        """
        check_confidentiality(text)


# The v2 node-integration tests (TestNodeIntegration) were deleted with
# the v2 review_document surface. v3's equivalent integration coverage
# lives in src/andamentum/whetstone/v3/tests/test_confidentiality.py.
