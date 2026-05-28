"""Confidentiality-marker tripwire.

Refuses to run a review when the document text contains markers that
suggest it is a peer-review submission, examiner report, editorial
office correspondence, or similar confidential material.

The user can override with the explicit ``--confirm-own-draft`` flag
(or ``confirm_own_draft=True`` to the API) — which is the affirmation
that this document is the user's own draft, not something shared with
them in confidence.

The marker list is intentionally short and recognisable. False
positives are tolerable because the override is one extra flag; false
negatives (failing to catch a peer-review run) would be the worse
failure mode.
"""

from __future__ import annotations

import re


class ConfidentialityMarkerError(RuntimeError):
    """Raised when the document text matches a confidentiality marker.

    Includes ``marker`` (the matched phrase) so callers can produce a
    precise error message and the user can verify the false positive.
    """

    def __init__(self, marker: str, context: str = "") -> None:
        self.marker = marker
        self.context = context
        super().__init__(
            f"Document contains confidentiality marker: {marker!r}. "
            f"Refusing to proceed. If this is your own draft and the marker "
            f"is a false positive, pass --confirm-own-draft to override. "
            f"If this document was shared with you in confidence (as a "
            f"peer reviewer, examiner, grant panel member, etc.), do NOT "
            f"run whetstone on it — most publishers and funders prohibit "
            f"AI peer review."
        )


# Pattern → human description. Phrases are matched case-insensitively as
# word-boundary substrings; some are full literal strings to avoid mis-firing
# on common research-writing usage. Ordered so the most specific markers fire
# first, but all are tested — the first match wins.
#
# DESIGN NOTE — what stays here vs. what does not:
# This list is intentionally limited to phrases that describe "this document
# was shared with you in confidence" — not phrases that identify a *type*
# of document. Funder-scheme codes (NHMRC APP, ARC DP, NIH RFA-/PAR-),
# scheme names ("Investigator Grant", "Linkage Project"), and role labels
# ("Lead CI", "Chief Investigator A") all legitimately appear in the user's
# OWN drafts; gating on them produces a false-positive rate so high that
# users learn to reflex-bypass with --confirm-own-draft, defeating the gate.
# The cross-country list would also be unbounded (UKRI, ERC, DFG, NSF, ...).
# The responsible-use prohibition on grant peer-review lives in
# RESPONSIBLE_USE.md; the affirmation surface is --confirm-own-draft.
_CONFIDENTIALITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmanuscript\s+id\s*:", re.IGNORECASE), "Manuscript ID:"),
    (re.compile(r"\bms#\s*\d", re.IGNORECASE), "MS#"),
    (re.compile(r"\bsubmission\s+id\s*:", re.IGNORECASE), "Submission ID:"),
    (
        re.compile(
            r"confidential\s*[-—]\s*do\s+not\s+(?:distribute|share)", re.IGNORECASE
        ),
        "Confidential — do not distribute",
    ),
    (
        re.compile(r"\bconfidential\s+manuscript\b", re.IGNORECASE),
        "Confidential manuscript",
    ),
    (
        re.compile(r"\breviewer\s+instructions?\b", re.IGNORECASE),
        "Reviewer Instructions",
    ),
    (re.compile(r"\beditorial\s+office\b", re.IGNORECASE), "Editorial Office"),
    (re.compile(r"\bdecision\s+letter\b", re.IGNORECASE), "Decision Letter"),
    (
        re.compile(r"this\s+manuscript\s+is\s+being\s+considered", re.IGNORECASE),
        "This manuscript is being considered",
    ),
    (
        re.compile(
            r"please\s+(?:do\s+not\s+share|treat\s+as\s+confidential)", re.IGNORECASE
        ),
        "Please do not share / treat as confidential",
    ),
    # Phrases that describe the *act of reviewing someone else's* funder
    # submission (the prohibited use case) — not the document type itself.
    # Kept because they only appear in cover-letters / panel correspondence
    # that the *reviewer* would receive, not in an author's own draft.
    (
        re.compile(r"\bassessor\s+(?:report|comments?)\b", re.IGNORECASE),
        "Assessor report / comments",
    ),
    (
        re.compile(
            r"\b(?:grant|funding)\s+panel\s+(?:review|assessment|member)\b",
            re.IGNORECASE,
        ),
        "Funding-panel review",
    ),
    (
        re.compile(
            r"\bpeer\s+review\s+of\s+(?:grant|application|proposal)\b", re.IGNORECASE
        ),
        "Peer review of grant/application",
    ),
]


def check_confidentiality(markdown: str) -> None:
    """Raise ``ConfidentialityMarkerError`` if a marker is found.

    No-op when the text is empty or no markers match.
    """
    if not markdown:
        return
    for pattern, label in _CONFIDENTIALITY_PATTERNS:
        match = pattern.search(markdown)
        if match is None:
            continue
        # Capture surrounding context so the error gives the user a
        # ground-truth view of what fired.
        start = max(0, match.start() - 40)
        end = min(len(markdown), match.end() + 40)
        context = markdown[start:end].replace("\n", " ")
        raise ConfidentialityMarkerError(label, context=context)
