"""Word document renderer for ``ReviewResult``.

Reuses the in-tree ``whetstone.docx.finalization.finalize_reviewed_document``
(the existing 5k-line track-changes machinery) by adapting v2's flat
``Edit`` and ``Finding`` types into v1's ``DocumentPatch`` shape:

  • Edit       → DocumentPatch(patch_type="text_edit",
                                text_pattern=original_text,
                                new_text=new_text,
                                explanation=rationale)
  • Finding    → DocumentPatch(patch_type="comment",
                                text_pattern=first_quote_text,
                                comment_text="<title>\\n\\n<rationale>",
                                explanation=rationale)

Findings without quotes are skipped — there's nothing to anchor the
comment to in the .docx structure. Edits whose ``original_text`` doesn't
appear in the .docx are also dropped silently (``apply_patches`` already
handles that).

Panel mode (``mode="panel"``) is now ALSO rendered: expert biosketches,
scored expert reviews, and the panel synthesis are folded into the
prepended review report. ``finalize_reviewed_document`` already accepts
``expert_reviews`` and ``generated_experts`` keywords; the v2 adapter
just passes them through. v2's ExpertProfile / ExpertReview pydantic
schemas are field-compatible with the v1 reader (which uses
``model_dump()`` then categorises by ``_score`` / ``_justification``
suffix).

Novelty findings (category="novelty") are also pulled out and passed as
``novelty_findings`` so they appear in their own report subsection
rather than mixed with the lens findings.

Requires an existing .docx file as input — Word's track-changes work
against a pre-existing structure. For PDF/HTML sources, render to
markdown or HTML instead, OR convert the source to .docx first via your
preferred tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import Edit, Finding, PanelSynthesis, ReviewResult


def render_docx(
    result: ReviewResult,
    *,
    source_docx_path: str | Path,
    output_path: str | Path,
    author: str = "Whetstone Review",
) -> Any:
    """Render a ReviewResult as a .docx with track changes + comments.

    Edits become tracked changes (deletion of the original span,
    insertion of the new span). Findings become Word comments anchored
    to the first verbatim quote of each finding. The executive summary
    becomes a prepended review report.

    For panel-mode results, the prepended report includes the panel
    synthesis prose, scored per-expert reviews, and expert biosketches.

    Parameters
    ----------
    result:
        The ReviewResult to render.
    source_docx_path:
        Path to the original .docx file. Track changes are applied
        against this document's structure.
    output_path:
        Where to write the reviewed .docx.
    author:
        Author name attributed to the tracked changes.

    Returns
    -------
    The ``PatchApplicationResult`` from the finalisation machinery
    (counts of applied/failed patches).
    """
    # Defer the heavy docx import: the track-changes machinery is large,
    # and we want v2 to be importable without dragging it into the import
    # graph by default.
    from andamentum.whetstone.docx.finalization import finalize_reviewed_document
    from andamentum.whetstone.models import DocumentPatch

    patches = _to_document_patches(result, DocumentPatch)
    findings = list(result.findings) + list(result.deterministic_findings)

    review_summary = result.summary or _fallback_summary(result)
    if result.panel_synthesis is not None:
        # Prepend the panel synthesis prose to whatever ``Synthesise``
        # produced. Both can be present in panel mode if the standard
        # synthesis path also ran; concatenating keeps the panel
        # narrative front-and-centre at the top of the report.
        review_summary = (
            _format_panel_synthesis(result.panel_synthesis)
            + ("\n\n" + review_summary if review_summary.strip() else "")
        )

    novelty_findings_text = _collect_novelty_findings(result)

    _, patch_result = finalize_reviewed_document(
        original_file_path=Path(source_docx_path),
        patches=patches,
        review_summary=review_summary,
        issues_count=len(findings),
        output_path=Path(output_path),
        author=author,
        # Panel-mode payload (None / empty when not in panel mode)
        expert_reviews=list(result.expert_reviews) or None,
        generated_experts=list(result.expert_profiles) or None,
        # Novelty-check payload (empty string when not used)
        novelty_findings=novelty_findings_text,
    )
    return patch_result


# ── Adapter: v2 → v1 DocumentPatch ──────────────────────────────────────


def _to_document_patches(result: ReviewResult, DocumentPatch) -> list:
    """Convert v2's Edit + Finding into v1's DocumentPatch list."""
    patches: list = []

    for e in result.edits:
        patches.append(_edit_to_patch(e, DocumentPatch))

    findings = list(result.findings) + list(result.deterministic_findings)
    for f in findings:
        # Novelty findings are surfaced separately via novelty_findings;
        # don't ALSO emit them as anchored comments because they have
        # no anchor in the manuscript text.
        if f.category == "novelty":
            continue
        patch = _finding_to_patch(f, DocumentPatch)
        if patch is not None:
            patches.append(patch)

    return patches


def _edit_to_patch(edit: Edit, DocumentPatch):
    return DocumentPatch(
        patch_type="text_edit",
        text_pattern=edit.original_text,
        new_text=edit.new_text,
        explanation=edit.rationale or edit.title or "Suggested edit",
        confidence=_confidence_to_float(edit.confidence),
    )


def _finding_to_patch(finding: Finding, DocumentPatch):
    if not finding.quotes:
        return None
    anchor = finding.quotes[0].text
    if not anchor or not anchor.strip():
        return None
    body = finding.title
    if finding.rationale:
        body = f"{finding.title}\n\n{finding.rationale}"
    return DocumentPatch(
        patch_type="comment",
        text_pattern=anchor,
        comment_text=body,
        explanation=finding.rationale or finding.title,
        confidence=_confidence_to_float(finding.confidence),
    )


def _confidence_to_float(level: str) -> float:
    """v2 uses low/medium/high enums; v1's DocumentPatch uses a 0..1 float."""
    return {"low": 0.4, "medium": 0.7, "high": 0.95}.get(level, 0.7)


def _fallback_summary(result: ReviewResult) -> str:
    """Generate a plain summary for when Synthesise didn't run."""
    n_findings = len(result.findings) + len(result.deterministic_findings)
    n_edits = len(result.edits)
    n_questions = len(result.author_questions)
    parts = [
        f"Whetstone review: {n_findings} finding(s), {n_edits} edit(s), "
        f"{n_questions} author question(s)."
    ]
    if result.expert_reviews:
        parts.append(f"Panel: {len(result.expert_reviews)} expert review(s).")
    if not (result.summary or "").strip():
        parts.append("Run with model= for synthesis prose.")
    return " ".join(parts)


def _format_panel_synthesis(s: PanelSynthesis) -> str:
    """Render PanelSynthesis as a markdown block for the prepended report.

    The structure is designed for the markdown→Word parser in
    ``whetstone.docx.low_level.prepend_review_section``:

      • ``## Panel Synthesis``      → Heading 2 in Word
      • ``> Recommendation: ...``   → Quote-styled callout (left bar)
      • ``### Subsection``          → Heading 3
      • ``- item``                  → real Word bullet list

    The recommendation gets its own quote block so it's the first
    visually-prominent thing the reader sees.
    """
    lines = [
        "## Panel Synthesis",
        "",
        # Quote-styled callout: Word renders this with a left bar +
        # italic, giving the headline recommendation visual weight.
        f"> **Recommendation: {s.overall_recommendation}** "
        f"(confidence: {s.confidence_level}) — average score "
        f"**{s.average_overall_score:.1f}/10** "
        f"(range: {s.score_range}, n={s.number_of_experts})",
        "",
        s.recommendation_justification,
        "",
    ]
    if (s.review_summary or "").strip():
        lines += ["### Review summary", "", s.review_summary.strip(), ""]
    if s.consensus_strengths:
        lines += ["### Consensus strengths", ""]
        lines += [f"- {item}" for item in s.consensus_strengths]
        lines += [""]
    if s.consensus_weaknesses:
        lines += ["### Consensus weaknesses", ""]
        lines += [f"- {item}" for item in s.consensus_weaknesses]
        lines += [""]
    if s.divergent_opinions:
        lines += ["### Divergent opinions", ""]
        lines += [f"- {item}" for item in s.divergent_opinions]
        lines += [""]
    by_criterion = [
        ("Scientific rigor", s.scientific_rigor_summary),
        ("Methodology", s.methodology_summary),
        ("Novelty", s.novelty_summary),
        ("Clarity", s.clarity_summary),
    ]
    if any((body or "").strip() for _, body in by_criterion):
        lines += ["### By criterion", ""]
        for label, body in by_criterion:
            if (body or "").strip():
                lines += [f"**{label}:** {body.strip()}", ""]
    if s.key_decision_factors:
        lines += ["### Key decision factors", ""]
        lines += [f"- {item}" for item in s.key_decision_factors]
        lines += [""]
    return "\n".join(lines)


def _collect_novelty_findings(result: ReviewResult) -> str:
    """Concatenate novelty-category findings into a markdown block.

    The v1 finalize_reviewed_document accepts a ``novelty_findings``
    string and renders it as its own report subsection. Pulling the
    novelty findings out of the lens-finding pool prevents them from
    becoming anchored comments against fictional text spans (they are
    document-level claims, not section-anchored observations).
    """
    novelty = [
        f
        for f in (list(result.findings) + list(result.deterministic_findings))
        if f.category == "novelty"
    ]
    if not novelty:
        return ""

    lines = [
        f"{len(novelty)} novelty claim(s) flagged by literature search:",
        "",
    ]
    for f in novelty:
        lines.append(f"- **{f.title}** ({f.severity}, {f.confidence} confidence)")
        if f.rationale:
            # Indent rationale under its bullet
            for ln in f.rationale.splitlines():
                lines.append(f"    {ln}")
        lines.append("")
    return "\n".join(lines)
