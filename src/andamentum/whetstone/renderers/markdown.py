"""Markdown renderer for ``ReviewResult``.

Ports the SECTION STRUCTURE of v1's ``whetstone/renderers/diff.py``,
adapted to v2's data shape (Findings, Edits, AuthorQuestions, summary,
document_map). Output is a single markdown string ready to write to a
``.md`` file or paste into any markdown-capable surface.

Section order (each separated by ``---``):
  1. Title, then the summary and any partial-review warning
  2. Author questions (when present — only the author can answer them)
  3. Findings, grouped must fix / should fix / consider and in document order
     within each group. Each finding carries its fix, and the editor's answer
     to it: the rewrite as a ```diff``` block, or what is left to the author.
  4. Other edits — the editor's own, which answer no finding
  5. Deterministic findings (structural)
  6. Novelty claims, the document map (reference), and the configuration

Empty sections are omitted so a clean document produces a clean report.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..schemas import (
    AuthorQuestion,
    CustomEvaluation,
    DeclinedFinding,
    NoveltyClaim,
    Edit,
    ExpertProfile,
    ExpertReview,
    Finding,
    GuidelineEvaluation,
    PanelSynthesis,
    ReviewResult,
    SectionCard,
)
from .._watermark import (
    banner_markdown,
    metadata_markdown_comment,
)
from ._finding_text import (
    NEXT_STEP_HEADING,
    REFERENCE_CHECK_HEADING,
    PRIORITY_LABEL,
    PRIORITY_ORDER,
    EditorAnswers,
    confidence_label,
    corroboration_clause,
    coverage_warning,
    declined_line,
    editor_answers,
    fix_line,
    reading_order,
    severity_note,
    summary_body,
)
from ._panel_layout import body_markdown
from ._reviewer_header import configuration_markdown, header_markdown

_TITLE = "# Whetstone Review"


def render_markdown(
    result: ReviewResult,
    output_path: str | Path | None = None,
    *,
    model: str | None = None,
    visible_watermark: bool = True,
) -> str:
    """Render a ReviewResult as markdown.

    Returns the markdown string. If ``output_path`` is given, also writes
    the string to that path (utf-8) — the path is created if missing.

    ``visible_watermark`` (default True) adds a top-of-file banner
    identifying this as AI-generated review content. The invisible
    HTML-comment metadata header is always written regardless of this
    flag — set it explicitly to False only when producing a derived
    artifact that should not carry a visible banner.
    """
    # Prelude (metadata header + title + optional banner) is appended
    # AFTER the review-content sections are assembled, so the
    # "looks clean" check counts review content only.
    sections: list[str] = []

    panel_mode = bool(result.expert_profiles or result.expert_reviews)

    # section_id → title map, used by the finding renderer to label
    # each per-finding location header (e.g. "Methods · s1").
    section_titles = {c.section_id: c.title for c in result.document_map}

    if result.summary.strip():
        sections.append(f"## Summary\n\n{summary_body(result.summary)}")

    # ── Partial-coverage notice — a check crashed and reviewed nothing ──
    warning = coverage_warning(result)
    if warning:
        sections.append(f"> {warning}")

    # ── Panel-mode sections (priority order) ─────────────────────────
    if result.panel_synthesis is not None:
        sections.append(
            _render_panel_synthesis(result.panel_synthesis, result.expert_reviews)
        )

    if result.expert_reviews:
        sections.append(_render_expert_reviews(result.expert_reviews))

    if result.expert_profiles:
        sections.append(_render_expert_profiles(result.expert_profiles))

    # ── Guidelines / custom mode sections ───────────────────────────
    if result.guideline_evaluations:
        sections.append(_render_guideline_evaluations(result.guideline_evaluations))

    if result.custom_evaluations:
        sections.append(_render_custom_evaluations(result.custom_evaluations))

    # ── Standard review-mode sections ────────────────────────────────
    if result.author_questions:
        sections.append(_render_questions(result.author_questions))

    # Findings come before edits: an edit is an answer, and it is shown under
    # the finding it answers rather than in a list of its own that the reader
    # had to match back by hand.
    answers = editor_answers(result)
    llm_findings = list(result.findings)
    if llm_findings:
        sections.append(
            _render_findings(
                llm_findings,
                heading="Findings",
                section_titles=section_titles,
                answers=answers,
            )
        )

    if answers.own_edits:
        sections.append(_render_edits(answers.own_edits, section_titles=section_titles))

    if answers.unlinked_declined:
        sections.append(_render_declined(answers.unlinked_declined))

    if result.deterministic_findings:
        sections.append(
            _render_findings(
                result.deterministic_findings,
                heading="Deterministic findings (structural)",
                section_titles=section_titles,
                answers=answers,
            )
        )

    if result.novelty_claims:
        sections.append(_render_novelty_claims(result.novelty_claims))

    if result.reference_summary:
        sections.append(f"## {REFERENCE_CHECK_HEADING}\n\n{result.reference_summary}")

    if result.next_steps:
        sections.append(f"## {NEXT_STEP_HEADING}\n\n" + "\n\n".join(result.next_steps))

    # ── Document map — reference, after the findings it helps locate ──
    if result.document_map and not panel_mode:
        sections.append(_render_document_map(result.document_map))

    # Provenance, last and quiet. See configuration_markdown for why it is not
    # at the top with the reviewer's identity and brief.
    if result.reviewer_header is not None:
        sections.append(configuration_markdown(result.reviewer_header))

    # Fallback: if there's nothing but a panel, still emit the doc map.
    if result.document_map and panel_mode and not sections:
        sections.append(_render_document_map(result.document_map))

    if not sections:
        # No review content. Be explicit rather than emitting a blank file.
        sections.append("_No findings, edits, or questions — document looks clean._")

    # Prepend prelude: invisible metadata header + visible title + optional
    # banner + the reviewer block, when a reviewer other than the default ran.
    prelude: list[str] = [metadata_markdown_comment(model=model), _TITLE]
    if visible_watermark:
        prelude.append(banner_markdown(model=model))
    if result.reviewer_header is not None:
        prelude.append(header_markdown(result.reviewer_header).rstrip())
    output = "\n\n---\n\n".join(prelude + sections) + "\n"

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")

    return output


# ── Section renderers ──────────────────────────────────────────────────


def _render_questions(qs: Iterable[AuthorQuestion]) -> str:
    qs = list(qs)
    lines = [f"## Author questions ({len(qs)})", ""]
    for q in qs:
        lines.append(f"- **{q.question}**")
        if q.why:
            lines.append(f"  _{q.why}_")
        if q.sections_involved:
            lines.append(f"  · sections: {', '.join(q.sections_involved)}")
    return "\n".join(lines)


def _render_novelty_claims(claims: list[NoveltyClaim]) -> str:
    """What the draft stakes as new — found on every run, verified on none.

    The heading and the first line both say "checked against nothing", because
    the one way this section can do harm is by being read as a verification.
    It is a list of the author's own assertions, quoted back.

    Omitted entirely when the list is empty. Silence means the extractor found
    no assertion of being first, novel or unique, which is the ordinary case
    for a tutorial or a short story — an explicit "none found" line would put a
    reassuring notice on every document that never made such a claim.
    """
    lines = [
        f"## Claims you are staking as novel ({len(claims)})",
        "",
        "_These are checked against nothing. This is what the draft asserts, not what the literature says._",
        "",
    ]
    for claim in claims:
        lines.append(f'- "{claim.claim_text}"')
        if claim.why_load_bearing:
            lines.append(f"  - _Why it matters:_ {claim.why_load_bearing}")
    lines += [
        "",
        "_To check them automatically, re-run with --check-novelty. That sends these sentences to a search engine._",
        "",
    ]
    return "\n".join(lines)


def _render_declined(declined: list[DeclinedFinding]) -> str:
    """Findings the editor read and could not answer with a rewrite.

    Placed directly after the edits, because the two together are the whole
    answer to "what did the editor do with the review": these are the ones it
    handed back, with its reason. Without them a finding with no edit against
    it is indistinguishable from one nobody looked at.
    """
    lines = [
        f"## Left to the author ({len(declined)})",
        "",
        "_The editor read these and judged that rewriting a passage cannot "
        "answer them. Each needs a decision, new content, or a restructure._",
        "",
    ]
    for d in declined:
        lines.append(f"- {d.issue}")
        if d.why:
            lines.append(f"  - _{declined_line(d.why)}_")
    lines.append("")
    return "\n".join(lines)


def _diff_block(edit: Edit) -> list[str]:
    lines = ["```diff"]
    lines.extend(f"- {ln}" for ln in edit.original_text.splitlines() or [""])
    lines.extend(f"+ {ln}" for ln in edit.new_text.splitlines() or [""])
    lines.append("```")
    return lines


def _render_edits(
    edits: Iterable[Edit], *, section_titles: dict[str, str] | None = None
) -> str:
    """The editor's own edits — the ones that answer no finding.

    Placed by section title, not by character offsets: "s3 (180–193)" is the
    pipeline's address, and the author reads section names.
    """
    edits = list(edits)
    section_titles = section_titles or {}
    lines = [f"## Other edits ({len(edits)})", ""]
    for e in edits:
        where = section_titles.get(e.section_id, e.section_id)
        lines.append(f"### {e.title}")
        lines.append("")
        lines.append(f"_{where}_")
        lines.append("")
        lines.extend(_diff_block(e))
        if e.rationale:
            lines.append(f"> {e.rationale}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_findings(
    findings: Iterable[Finding],
    *,
    heading: str,
    section_titles: dict[str, str] | None = None,
    answers: EditorAnswers | None = None,
) -> str:
    """Editorial-annotation layout: per-finding section header → quoted
    passage (blockquote) → the problem → the fix → the editor's answer.

    Mirrors the HTML renderer's layout so the two outputs read the same.
    The section_titles map (section_id → title) comes from the document
    map; missing ids render with the id itself.
    """
    findings = reading_order(list(findings))
    section_titles = section_titles or {}
    answers = answers or EditorAnswers()
    by_priority: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_priority[f.priority].append(f)

    counts = [
        f"{len(by_priority[p])} {PRIORITY_LABEL[p].lower()}"
        for p in PRIORITY_ORDER
        if by_priority.get(p)
    ]
    counts_line = " · ".join(counts) if counts else "0 findings"
    lines = [f"## {heading} ({len(findings)})", "", f"_{counts_line}_", ""]

    for priority in PRIORITY_ORDER:
        group = by_priority.get(priority, [])
        if not group:
            continue
        lines.append(f"### {PRIORITY_LABEL[priority]} ({len(group)})")
        lines.append("")
        for f in group:
            # Per-finding location header, by the section's title.
            section_id = (
                f.quotes[0].section_id
                if f.quotes
                else (f.sections_involved[0] if f.sections_involved else "")
            )
            lines.append(
                f"#### {section_titles.get(section_id, section_id or 'The document as a whole')}"
            )
            lines.append("")

            # Quoted passage — the thing being commented on. Verbatim
            # span from the source, wrapped in a blockquote so it
            # visually leads the comment that follows.
            for q in f.quotes[:1]:
                preview = q.text.replace("\n", " ").strip()
                lines.append(f"> {preview}")
                lines.append("")

            tags = [
                t
                for t in (f.perspective or "", severity_note(f), confidence_label(f))
                if t
            ]
            tag_part = f" &nbsp; _{' · '.join(tags)}_" if tags else ""
            lines.append(f"**{f.title}**{tag_part}")
            if f.rationale:
                lines.append("")
                lines.append(f.rationale)
            if fix_line(f):
                lines.append("")
                lines.append(f"**Fix:** {f.fix.strip()}")
            for edit in answers.edits_by_finding.get(f.id, []):
                lines.append("")
                lines.append("Suggested rewrite:")
                lines.append("")
                lines.extend(_diff_block(edit))
            why = answers.declined_by_finding.get(f.id, "")
            if why:
                lines.append("")
                lines.append(f"_{declined_line(why)}_")
            clause = corroboration_clause(f)
            if clause:
                lines.append("")
                lines.append(f"_{clause}_")
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_panel_synthesis(s: PanelSynthesis, reviews: list[ExpertReview]) -> str:
    """Top-of-report panel synthesis with score table + consensus/divergence."""
    return "## Panel synthesis\n\n" + body_markdown(s, reviews)


def _render_expert_reviews(reviews: Iterable[ExpertReview]) -> str:
    """Per-expert reviews. Collapsible details via blockquote indentation."""
    reviews = list(reviews)
    lines = [f"## Expert reviews ({len(reviews)})", ""]
    for r in reviews:
        lines += [
            f"### {r.expert_name} — {r.discipline}",
            "",
            f"**Overall: {r.overall_score}/10** · "
            f"Recommendation: **{r.recommendation}**",
            "",
            r.overall_assessment,
            "",
            "| Criterion | Score | Justification |",
            "| --- | --- | --- |",
            f"| Scientific rigor | {r.scientific_rigor_score}/10 | "
            f"{_oneline(r.scientific_rigor_justification)} |",
            f"| Methodology | {r.methodology_score}/10 | "
            f"{_oneline(r.methodology_justification)} |",
            f"| Novelty | {r.novelty_score}/10 | {_oneline(r.novelty_justification)} |",
            f"| Clarity | {r.clarity_score}/10 | {_oneline(r.clarity_justification)} |",
            "",
        ]
        if r.strengths:
            lines += ["**Strengths**", ""]
            lines += [f"- {item}" for item in r.strengths]
            lines += [""]
        if r.weaknesses:
            lines += ["**Weaknesses**", ""]
            lines += [f"- {item}" for item in r.weaknesses]
            lines += [""]
        if r.recommendation_justification.strip():
            lines += [
                f"_{r.recommendation_justification.strip()}_",
                "",
            ]
    return "\n".join(lines).rstrip()


def _render_expert_profiles(profiles: Iterable[ExpertProfile]) -> str:
    """Footnote-style biosketches at the bottom of the report."""
    profiles = list(profiles)
    lines = [f"## Expert biosketches ({len(profiles)})", ""]
    for p in profiles:
        lines += [
            f"### {p.name} — {p.discipline}",
            "",
            f"**Position.** {p.position}",
            "",
            f"**Education.** {p.education}",
            "",
            f"**Contributions.** {p.contributions}",
            "",
            f"**Research.** {p.research}",
            "",
        ]
    return "\n".join(lines).rstrip()


def _oneline(s: str) -> str:
    """Squash newlines + collapse whitespace so the markdown table doesn't break."""
    return " ".join(s.split())


_STATUS_HEADINGS = {
    "fail": "FAIL",
    "unclear": "UNCLEAR",
    "pass": "PASS",
}


def _render_guideline_evaluations(
    evaluations: Iterable[GuidelineEvaluation],
) -> str:
    """Group evaluations by status (fail / unclear / pass), fail first."""
    evaluations = list(evaluations)
    by_status: dict[str, list[GuidelineEvaluation]] = {
        "fail": [],
        "unclear": [],
        "pass": [],
    }
    for e in evaluations:
        by_status.setdefault(e.status, []).append(e)

    lines = [f"## Journal-guideline checks ({len(evaluations)})", ""]
    for status in ("fail", "unclear", "pass"):
        group = by_status.get(status, [])
        if not group:
            continue
        lines.append(f"### {_STATUS_HEADINGS[status]} ({len(group)})")
        lines.append("")
        for e in group:
            cat = f" · _{e.category}_" if e.category else ""
            lines.append(f"- **{e.item_name}**{cat}")
            if e.notes:
                lines.append(f"  {e.notes}")
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_custom_evaluations(
    evaluations: Iterable[CustomEvaluation],
) -> str:
    """Group custom-criteria evaluations by status, fail first."""
    evaluations = list(evaluations)
    by_status: dict[str, list[CustomEvaluation]] = {
        "fail": [],
        "unclear": [],
        "pass": [],
    }
    for e in evaluations:
        by_status.setdefault(e.status, []).append(e)

    lines = [f"## Custom-criteria evaluation ({len(evaluations)})", ""]
    for status in ("fail", "unclear", "pass"):
        group = by_status.get(status, [])
        if not group:
            continue
        lines.append(f"### {_STATUS_HEADINGS[status]} ({len(group)})")
        lines.append("")
        for e in group:
            lines.append(f"- **{e.criterion}**")
            if e.notes:
                lines.append(f"  {e.notes}")
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_document_map(cards: Iterable[SectionCard]) -> str:
    cards = list(cards)
    lines = [f"## Document map ({len(cards)} sections)", ""]
    for c in cards:
        gist = (c.one_line_gist or "").strip()
        gist_part = f" — {gist}" if gist else ""
        lines.append(f"- **{c.section_id}** {c.title}{gist_part}")
    return "\n".join(lines)
