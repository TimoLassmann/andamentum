"""Structural validators for scribe documents.

These are deterministic, no-LLM checks: missing citation keys, missing
figure files, unused references, unresolved [verify] / [citation needed]
markers. Surface as `ValidationIssue` records with severity in
{error, warning, info}. No silent failures — callers should display
every issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .models import ValidationIssue
from .parser import find_ai_markers, find_unresolved_markers

if TYPE_CHECKING:  # pragma: no cover
    from .api import Document


def validate_document(doc: "Document") -> list[ValidationIssue]:
    """Run all structural validators against `doc`."""
    issues: list[ValidationIssue] = []
    cite_keys_used = set(doc.citations())
    cite_keys_defined = {r.cite_key for r in doc.references()}

    # Missing citation keys: used but not defined.
    for key in cite_keys_used - cite_keys_defined:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Citation [@{key}] referenced but no matching reference defined.",
                location=key,
            )
        )

    # Unused references: defined but not used.
    for key in cite_keys_defined - cite_keys_used:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=f"Reference {key!r} defined but never cited.",
                location=key,
            )
        )

    # Missing figure files.
    for blk in doc.query(type="figure"):
        path = blk.metadata.get("path")
        if path and not Path(path).exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Figure file not found: {path}",
                    location=blk.metadata.get("label", blk.id),
                )
            )

    # Unresolved citation markers in paragraphs.
    for blk in doc.query(type="paragraph"):
        for marker in find_unresolved_markers(blk.content):
            issues.append(
                ValidationIssue(
                    severity="info",
                    message=f"Unresolved citation marker [{marker}] in block.",
                    location=blk.id,
                )
            )

    # AI-provenance markers in paragraphs — warn so the author confirms
    # the marker is intended and remembers to disclose AI assistance in
    # methods/acknowledgements per their journal's policy.
    for blk in doc.query(type="paragraph"):
        for marker in find_ai_markers(blk.content):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"AI-provenance marker [{marker}] in block — "
                        f"remember to disclose AI assistance in your "
                        f"methods/acknowledgements per your journal's policy."
                    ),
                    location=blk.id,
                )
            )

    return issues
