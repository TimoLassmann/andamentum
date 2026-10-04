"""Structural validators for bundles.

Deterministic, no LLM. Every problem is a ``ValidationIssue``; nothing is
swallowed. Errors stop a render being trustworthy (duplicate titles, heading
lines in a body, missing image files); warnings flag unresolved citations and
markers; info notes empty bodies.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from .markdown import HEADING_RE, ImageBlock, parse_blocks
from .models import ValidationIssue
from .parser import extract_citation_keys, find_unresolved_markers

if TYPE_CHECKING:  # pragma: no cover
    from .bundle import Bundle


def validate_bundle(bundle: "Bundle") -> list[ValidationIssue]:
    """Run every structural check against *bundle*."""
    issues: list[ValidationIssue] = []
    sections = bundle.sections()
    defined = {r.key for r in bundle.references()}

    for title, count in Counter(s.title for s in sections).items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Section title {title!r} is used by {count} sections.",
                    location=title,
                )
            )

    for section in sections:
        if not section.body.strip():
            issues.append(
                ValidationIssue(
                    severity="info",
                    message=f"Section {section.title!r} has an empty body.",
                    location=section.id,
                )
            )
            continue
        for number, line in enumerate(section.body.splitlines(), start=1):
            if HEADING_RE.match(line):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"Heading line in the body of {section.title!r} "
                            f"(line {number}): {line.strip()!r}. Bodies hold no headings."
                        ),
                        location=section.id,
                    )
                )
        for block in parse_blocks(section.body):
            if isinstance(block, ImageBlock):
                image = Path(block.path)
                resolved = image if image.is_absolute() else bundle.path / image
                if not resolved.is_file():
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"Image file not found: {block.path}",
                            location=block.path,
                        )
                    )
        for key in dict.fromkeys(extract_citation_keys(section.body)):
            if key not in defined:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Citation [@{key}] has no entry in references.yaml.",
                        location=section.id,
                    )
                )
        for marker in find_unresolved_markers(section.body):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Unresolved marker [{marker}] in {section.title!r}.",
                    location=section.id,
                )
            )
    return issues
