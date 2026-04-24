"""Lightweight markdown diff renderer and patch applicator.

apply_patches: Apply text_edit patches to content via string replacement.
render_diff: Render patches and issues as a readable markdown diff view.
"""

from __future__ import annotations

from typing import Sequence

from ..models import DocumentPatch
from ..issues import DocumentIssue

_STATUS_MARKER = {"pass": "✓ PASS", "fail": "✗ FAIL", "unclear": "? UNCLEAR"}


def _render_checklist_markdown(items: list) -> str:
    if not items:
        return ""
    from collections import defaultdict

    by_cat: dict[str, list] = defaultdict(list)
    for it in items:
        by_cat[it.category or "other"].append(it)

    passes = sum(1 for i in items if i.status == "pass")
    fails = sum(1 for i in items if i.status == "fail")
    unclears = sum(1 for i in items if i.status == "unclear")

    lines = [
        "## Pre-submission checklist",
        "",
        f"**Summary:** {passes} pass · {fails} fail · {unclears} unclear (of {len(items)} items)",
        "",
    ]
    for cat in sorted(by_cat):
        lines.append(f"### {cat.title()}")
        lines.append("")
        for it in by_cat[cat]:
            marker = _STATUS_MARKER.get(it.status, "?")
            lines.append(f"- **{marker}** {it.name}")
            if it.notes:
                lines.append(f"  - {it.notes}")
        lines.append("")
    return "\n".join(lines)


def apply_patches(content: str, patches: Sequence[DocumentPatch]) -> str:
    """Apply text_edit patches to content via string replacement.

    Patches are sorted by confidence (highest first) so the most
    confident edits take priority when patterns overlap. Comment and
    document_analysis patches are skipped. If a text_pattern is not
    found in the (possibly already-modified) content, that patch is
    silently skipped.

    Args:
        content: The original document text.
        patches: Sequence of DocumentPatch objects to apply.

    Returns:
        The content string with all applicable text_edit replacements made.
    """
    # Only text_edit patches produce replacements
    edits = [p for p in patches if p.patch_type == "text_edit"]

    # Highest confidence first
    edits.sort(key=lambda p: p.confidence, reverse=True)

    for patch in edits:
        if patch.text_pattern and patch.new_text and patch.text_pattern in content:
            content = content.replace(patch.text_pattern, patch.new_text, 1)

    return content


def render_diff(
    *,
    patches: Sequence[DocumentPatch],
    issues: Sequence[DocumentIssue],
    original_content: str,
    synthesis_text: str | None = None,
    checklist: list | None = None,
) -> str:
    """Render a lightweight markdown diff view.

    Sections (separated by ``---``):
    1. Optional synthesis summary
    2. Edits -- ``diff`` fenced blocks with old/new text and rationale
    3. Comments -- bold location lines with blockquoted text
    4. Issues -- grouped by type with headings, descriptions, recommendations
    5. Fallback "No edits or issues found." when everything is empty

    Args:
        patches: Sequence of DocumentPatch objects (edits + comments).
        issues: Sequence of DocumentIssue objects.
        original_content: The original document text (unused currently,
            reserved for future context-aware rendering).
        synthesis_text: Optional high-level synthesis/summary text.

    Returns:
        A markdown string ready for display in a terminal or log file.
    """
    sections: list[str] = []

    # -- Synthesis ----------------------------------------------------------
    if synthesis_text:
        sections.append(f"## Synthesis\n\n{synthesis_text}")

    # -- Edits --------------------------------------------------------------
    edit_patches = [p for p in patches if p.patch_type == "text_edit"]
    if edit_patches:
        lines: list[str] = ["## Edits", ""]
        for patch in edit_patches:
            lines.append("```diff")
            lines.append(f"- {patch.text_pattern}")
            lines.append(f"+ {patch.new_text}")
            lines.append("```")
            lines.append(f"> {patch.explanation}")
            lines.append("")
        sections.append("\n".join(lines))

    # -- Comments -----------------------------------------------------------
    comment_patches = [p for p in patches if p.patch_type == "comment"]
    if comment_patches:
        lines = ["## Comments", ""]
        for patch in comment_patches:
            location = patch.text_pattern or "General"
            lines.append(f"**At:** {location}")
            lines.append(f"> {patch.comment_text}")
            lines.append("")
        sections.append("\n".join(lines))

    # -- Issues -------------------------------------------------------------
    if issues:
        type_order = ["major", "minor", "suggestion", "strength"]
        type_headings = {
            "major": "### Major Issues",
            "minor": "### Minor Issues",
            "suggestion": "### Suggestions",
            "strength": "### Strengths",
        }

        issue_lines: list[str] = ["## Issues", ""]

        for issue_type in type_order:
            group = [i for i in issues if i.issue_type == issue_type]
            if not group:
                continue
            issue_lines.append(type_headings[issue_type])
            issue_lines.append("")
            for issue in group:
                issue_lines.append(f"- **{issue.title}**")
                issue_lines.append(f"  {issue.description}")
                if issue.recommendation:
                    issue_lines.append(f"  *Recommendation:* {issue.recommendation}")
                issue_lines.append("")

        sections.append("\n".join(issue_lines))

    # -- Empty fallback -----------------------------------------------------
    if not sections:
        output = "No edits or issues found."
    else:
        output = "\n\n---\n\n".join(sections)

    if checklist:
        output = _render_checklist_markdown(checklist) + "\n\n" + output
    return output
