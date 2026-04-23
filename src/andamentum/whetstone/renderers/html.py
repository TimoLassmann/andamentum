"""HTML renderer for whetstone — builds andamentum.typeset atoms.

Replaces the former bespoke 1100-line inline-CSS renderer. The ReviewResult
is walked once, turned into a list of typeset atoms (heading, prose, callout,
items, card), and handed to typeset.render() for styling.

Supports all three tasks:
- edit: edits and comments as cards with old→new diffs
- review: issues grouped by severity, plus synthesis prose
- panel: expert profile cards, expert review cards with scores, synthesis
"""

from __future__ import annotations

from typing import Any

from andamentum.typeset import render as typeset_render

from ..issues import DocumentIssue
from ..models import DocumentPatch

# Positioning reminder shown on every report.
_DISCLAIMER = (
    "This report was generated for your own drafts. Whetstone is not a "
    "peer-review tool — do not use it on manuscripts other authors have "
    "sent you confidentially."
)

# Visible title per task.
_TASK_TITLES: dict[str, str] = {
    "edit": "Whetstone · Editing Pass",
    "review": "Whetstone · Review",
    "panel": "Whetstone · Expert Panel",
}


# ---------------------------------------------------------------------------
# Atom builders per section
# ---------------------------------------------------------------------------


def _heading_atom(task: str) -> dict[str, Any]:
    title = _TASK_TITLES.get(task, f"Whetstone · {task.title()}")
    return {
        "kind": "heading",
        "content": title,
        "subtitle": "Structured feedback on your own draft",
    }


def _disclaimer_atom() -> dict[str, Any]:
    return {"kind": "callout", "tone": "note", "content": _DISCLAIMER}


def _synthesis_atom(synthesis: Any) -> list[dict[str, Any]]:
    """Render a synthesis object as prose + items atoms."""
    atoms: list[dict[str, Any]] = []

    summary = getattr(synthesis, "review_summary", None)
    if summary:
        atoms.append({"kind": "prose", "heading": "Summary", "content": summary})

    recommendation = getattr(synthesis, "overall_recommendation", None)
    justification = getattr(synthesis, "recommendation_justification", None)
    if recommendation:
        body = recommendation
        if justification:
            body = f"**{recommendation}** — {justification}"
        atoms.append({"kind": "callout", "tone": "success", "content": body})

    # Panel-specific fields
    strengths = getattr(synthesis, "consensus_strengths", None)
    weaknesses = getattr(synthesis, "consensus_weaknesses", None)
    if strengths:
        atoms.append(
            {
                "kind": "items",
                "heading": "Consensus strengths",
                "variant": "pairs",
                "entries": [{"label": f"{i}.", "body": s} for i, s in enumerate(strengths, 1)],
            }
        )
    if weaknesses:
        atoms.append(
            {
                "kind": "items",
                "heading": "Consensus weaknesses",
                "variant": "pairs",
                "entries": [{"label": f"{i}.", "body": w} for i, w in enumerate(weaknesses, 1)],
            }
        )

    # Standard review: critical_issues list
    critical = getattr(synthesis, "critical_issues", None)
    if critical:
        entries: list[dict[str, str]] = []
        for i, issue in enumerate(critical, 1):
            title = getattr(issue, "title", "")
            description = getattr(issue, "description", "")
            rec = getattr(issue, "recommendation", "")
            body = description
            if rec:
                body = f"{description}\n\n*Recommendation:* {rec}"
            entries.append({"label": f"{i}. {title}", "body": body})
        atoms.append(
            {
                "kind": "items",
                "heading": "Critical issues",
                "variant": "pairs",
                "entries": entries,
            }
        )

    recommendations_text = getattr(synthesis, "recommendations", None)
    if recommendations_text:
        atoms.append(
            {
                "kind": "prose",
                "heading": "Recommendations",
                "content": recommendations_text,
            }
        )

    return atoms


def _edit_card(patch: DocumentPatch) -> dict[str, Any]:
    """Convert a text_edit patch into a typeset card atom."""
    old = patch.text_pattern or ""
    new = patch.new_text or ""
    explanation = patch.explanation or ""
    confidence_pct = f"{patch.confidence * 100:.0f}%"
    body = f"**Before:** {old}\n\n**After:** {new}\n\n{explanation}"
    return {
        "kind": "card",
        "content": body,
        "badge": confidence_pct,
    }


def _comment_card(patch: DocumentPatch) -> dict[str, Any]:
    location = patch.text_pattern or "(general)"
    body = f"**At:** {location}\n\n{patch.comment_text or ''}\n\n*{patch.explanation or ''}*"
    return {"kind": "card", "content": body}


def _analysis_card(patch: DocumentPatch) -> dict[str, Any]:
    return {
        "kind": "card",
        "content": patch.analysis_text or "",
        "badge": "analysis",
    }


def _edit_section(patches: list[DocumentPatch]) -> list[dict[str, Any]]:
    """Build atoms for the Edits / Comments / Analyses sections."""
    edits = [p for p in patches if p.patch_type == "text_edit"]
    comments = [p for p in patches if p.patch_type == "comment"]
    analyses = [p for p in patches if p.patch_type == "document_analysis"]

    atoms: list[dict[str, Any]] = []
    if edits:
        atoms.append({"kind": "prose", "heading": f"Proposed edits ({len(edits)})", "content": ""})
        atoms.extend(_edit_card(p) for p in edits)
    if comments:
        atoms.append({"kind": "prose", "heading": f"Comments ({len(comments)})", "content": ""})
        atoms.extend(_comment_card(p) for p in comments)
    if analyses:
        atoms.append({"kind": "prose", "heading": f"Analyses ({len(analyses)})", "content": ""})
        atoms.extend(_analysis_card(p) for p in analyses)
    if not (edits or comments or analyses):
        atoms.append({"kind": "prose", "content": "_No edits or comments were proposed._"})
    return atoms


def _issue_card(issue: DocumentIssue) -> dict[str, Any]:
    """Convert a DocumentIssue into a card atom. Severity goes in the badge."""
    body_parts = [issue.description]
    if issue.recommendation:
        body_parts.append(f"*Recommendation:* {issue.recommendation}")
    if issue.location:
        body_parts.append(f"*Location:* {issue.location}")
    return {
        "kind": "card",
        "content": f"**{issue.title}**\n\n" + "\n\n".join(body_parts),
        "badge": issue.issue_type,
    }


def _issue_section(issues: list[DocumentIssue]) -> list[dict[str, Any]]:
    """Group issues by severity, each group rendered under its own heading."""
    if not issues:
        return []

    buckets: dict[str, list[DocumentIssue]] = {
        "major": [],
        "minor": [],
        "suggestion": [],
        "strength": [],
    }
    for issue in issues:
        buckets.setdefault(issue.issue_type, []).append(issue)

    label = {
        "major": "Major issues",
        "minor": "Minor issues",
        "suggestion": "Suggestions",
        "strength": "Strengths",
    }
    atoms: list[dict[str, Any]] = []
    for severity in ("major", "minor", "suggestion", "strength"):
        group = buckets.get(severity, [])
        if not group:
            continue
        atoms.append(
            {
                "kind": "prose",
                "heading": f"{label[severity]} ({len(group)})",
                "content": "",
            }
        )
        atoms.extend(_issue_card(i) for i in group)
    return atoms


def _expert_profile_card(profile: Any) -> dict[str, Any]:
    body = (
        f"**{profile.name}**\n\n"
        f"{profile.position}\n\n"
        f"*Education:* {profile.education}\n\n"
        f"*Research:* {profile.research}\n\n"
        f"*Key contributions:* {profile.contributions}"
    )
    return {"kind": "card", "content": body, "badge": profile.discipline}


def _expert_review_card(review: Any) -> dict[str, Any]:
    body = (
        f"**{review.expert_name}** — {review.discipline}\n\n"
        f"**Overall:** {review.overall_score}/10 — {review.overall_assessment}\n\n"
        f"- Scientific rigor: {review.scientific_rigor_score}/10 — "
        f"{review.scientific_rigor_justification}\n"
        f"- Methodology: {review.methodology_score}/10 — "
        f"{review.methodology_justification}\n"
        f"- Novelty: {review.novelty_score}/10 — {review.novelty_justification}\n"
        f"- Clarity: {review.clarity_score}/10 — {review.clarity_justification}\n\n"
        f"**Strengths:** {', '.join(review.strengths)}\n\n"
        f"**Weaknesses:** {', '.join(review.weaknesses)}\n\n"
        f"**Recommendation:** {review.recommendation} — "
        f"{review.recommendation_justification}"
    )
    return {"kind": "card", "content": body, "badge": review.recommendation}


def _panel_section(result: Any) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    profiles = getattr(result, "expert_profiles", []) or []
    reviews = getattr(result, "expert_reviews", []) or []
    if profiles:
        atoms.append(
            {
                "kind": "prose",
                "heading": f"Expert panel ({len(profiles)})",
                "content": "",
            }
        )
        atoms.extend(_expert_profile_card(p) for p in profiles)
    if reviews:
        atoms.append(
            {
                "kind": "prose",
                "heading": "Expert reviews",
                "content": "",
            }
        )
        atoms.extend(_expert_review_card(r) for r in reviews)
    return atoms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_html(
    *,
    result: Any,
    original_content: str,
    style: str = "article",
) -> str:
    """Render a ReviewResult to a standalone HTML string via andamentum.typeset.

    Args:
        result: ReviewResult from sharpen_document().
        original_content: The draft text (reserved for future use — e.g.
            inline-diff rendering).
        style: Built-in typeset style: "article" (default), "cv", or "report".

    Returns:
        A complete HTML document string.
    """
    task = getattr(result, "task", "review")

    atoms: list[dict[str, Any]] = [
        _heading_atom(task),
        _disclaimer_atom(),
    ]

    synthesis = getattr(result, "synthesis", None)
    if synthesis is not None:
        atoms.extend(_synthesis_atom(synthesis))

    if task == "edit":
        atoms.extend(_edit_section(list(getattr(result, "patches", []) or [])))
    elif task == "review":
        atoms.extend(_issue_section(list(getattr(result, "issues", []) or [])))
    elif task == "panel":
        atoms.extend(_panel_section(result))

    return typeset_render(atoms, style=style, title=_TASK_TITLES.get(task, "Whetstone"))
