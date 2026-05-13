"""Cochrane-style audit report — parallel renderer to ``typeset_report.py``.

The classic renderer (``typeset_report.build_typeset_report``) produces a
clean, prose-heavy report. This module produces a different shape:
**audit-first**, optimised for someone who wants to inspect the
system's reasoning trail rather than just read the answer.

Differences from the classic layout:

1. **Headline panel** at the top — claim, verdict badge, posterior pill.
2. **Summary of findings table** (Cochrane-style) immediately under
   the headline — small, scannable, the bottom line in 30 seconds.
3. **Plain-language summary** — same prose body as classic, but
   visually separated from the evidence breakdown.
4. **Key evidence per claim** — the claim card shows the 3-5 strongest
   supporting items and the strongest counter-evidence inline, with
   clickable source links. The 98-item flat reference list that
   currently dominates the classic claim card is *gone* — the full
   list is in the appendix as a collapsible section.
5. **Audit trail per claim** — investigation rounds as a proper list,
   IBE chain candidates as a markdown table, adversarial probe as
   explicit prose. Each section is a collapsible ``<details>`` block
   so the reader can opt in.
6. **Caveats & Limitations** — same as classic.
7. **Appendix: full evidence trail** — every retrieved item with its
   judgement and source link, in collapsible groups by direction.

The module uses **only the 7 built-in typeset atoms** (heading,
prose, callout, items, aside, card, reference). Tables go in
markdown inside ``prose()`` / ``card(details=)`` content; the typeset
renderer's ``TableExtension`` handles them. Collapsible sections use
``card(details=)`` which already emits ``<details>``.

Run modes supported:

- **Verify mode** (single claim seeded via ``claim_to_verify``): one
  claim card with full audit trail.
- **Research mode** (decomposition into sub-investigations): the
  combined verdict shows at the top, and each sub-claim renders as
  its own audit card under "Sub-investigations". The combination
  rule is surfaced as a one-liner.
"""

from __future__ import annotations

import re
from typing import Any

from andamentum.typeset import Report

from .report_data import (
    ClaimSummary,
    EvidenceSummary,
    QUESTION_TYPE_LABELS,
    ReportData,
)

# How many top supporting / contradicting items to show inline per claim.
# Items beyond this go into the appendix's full list.
_INLINE_TOP_K = 5


# ──────────────────────────────────────────────────────────────────────────────
# Source-ref → clickable URL
# ──────────────────────────────────────────────────────────────────────────────


_DOI_RE = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_PMID_RE = re.compile(r"^(?:pmid:)?(\d{6,9})$", re.IGNORECASE)
_NCT_RE = re.compile(r"^(NCT\d{8})$", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _source_url(source_ref: str) -> str:
    """Convert a source identifier into a clickable URL.

    - DOI (with or without ``doi:`` prefix) → ``https://doi.org/<doi>``.
    - PMID (with or without ``PMID:`` prefix) → PubMed URL.
    - ClinicalTrials.gov NCT number → trial URL.
    - Already an http(s) URL → unchanged.
    - Anything else → unchanged (the renderer will show it as bare text).
    """
    if not source_ref:
        return source_ref
    s = source_ref.strip()
    if _URL_RE.match(s):
        return s
    m = _DOI_RE.match(s)
    if m:
        return f"https://doi.org/{m.group(1)}"
    m = _PMID_RE.match(s)
    if m:
        return f"https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/"
    m = _NCT_RE.match(s)
    if m:
        return f"https://clinicaltrials.gov/study/{m.group(1)}"
    return s


def _short_source(source_ref: str) -> str:
    """Compact display label for a source identifier."""
    if not source_ref:
        return ""
    s = source_ref.strip()
    if _URL_RE.match(s):
        # Strip protocol + leading www. for compactness.
        return re.sub(r"^https?://(www\.)?", "", s)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Verdict + posterior framing
# ──────────────────────────────────────────────────────────────────────────────


def _verdict_label(data: ReportData) -> str:
    """Top-level verdict label — Supported / Refuted / Insufficient / Suspended.

    Reads from confidence_scores + claim stages, not from the artefact
    prose, so the badge is consistent with the posterior even when the
    written summary uses different language.
    """
    cs = data.confidence_scores
    if cs is None or cs.terminal_state != "completed":
        # Any non-completed terminal (retrieval_failed, oscillation_detected, …)
        # is rendered as "Insufficient" — the system suspended judgment.
        return "Insufficient evidence"
    if cs.posterior is None:
        return "Insufficient evidence"
    if cs.posterior >= 0.7:
        return "Supported"
    if cs.posterior <= 0.3:
        return "Refuted"
    return "Inconclusive"


def _verdict_tone(label: str) -> str:
    """Map a verdict label to a callout tone (info / warning / success / note)."""
    return {
        "Supported": "success",
        "Refuted": "warning",
        "Inconclusive": "note",
        "Insufficient evidence": "warning",
    }.get(label, "info")


def _posterior_pill(data: ReportData) -> str:
    """Compact posterior expression for the headline pill."""
    cs = data.confidence_scores
    if cs is None or cs.terminal_state != "completed" or cs.posterior is None:
        return "Suspended"
    return f"P(YES) ≈ {cs.posterior * 100:.0f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Markdown helpers — table rendering inside prose / card details
# ──────────────────────────────────────────────────────────────────────────────


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table the typeset renderer's TableExtension will
    pick up. Empty rows → empty body (just the header)."""
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_row, sep_row]
    for row in rows:
        cells = [str(c) if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_link(label: str, url: str) -> str:
    """Inline markdown link, with the label escaped of brackets."""
    safe_label = label.replace("[", "\\[").replace("]", "\\]")
    return f"[{safe_label}]({url})"


def _pct(numerator: int, denominator: int) -> str:
    """Render a percentage; safe on zero denominator."""
    if denominator <= 0:
        return "—"
    return f"{100 * numerator / denominator:.0f}%"


def _sanitize_excerpt(text: str, max_chars: int = 600) -> str:
    """Trim an evidence excerpt to a reasonable inline length without
    cutting mid-sentence where possible."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind(". ", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].rstrip() + "…"


# ──────────────────────────────────────────────────────────────────────────────
# Per-claim renderers
# ──────────────────────────────────────────────────────────────────────────────


def _evidence_for_claim(
    claim: ClaimSummary, all_evidence: list[EvidenceSummary]
) -> tuple[list[EvidenceSummary], list[EvidenceSummary], list[EvidenceSummary]]:
    """Bucket a claim's evidence into supports / contradicts / no_bearing
    lists, preserving the order in which they appear in ``data.evidence``
    (which is already sorted with supports first)."""
    claim_evidence = [
        ev for ev in all_evidence if ev.evidence_id in claim.evidence_ids
    ]
    supports = [e for e in claim_evidence if e.support_judgment == "supports"]
    contradicts = [e for e in claim_evidence if e.support_judgment == "contradicts"]
    no_bearing = [e for e in claim_evidence if e.support_judgment == "no_bearing"]
    return supports, contradicts, no_bearing


def _evidence_line(ev: EvidenceSummary) -> str:
    """One markdown line representing an evidence item for inline display.

    Format: ``[provider] [judgement-reasoning] · [clickable source]``
    """
    parts: list[str] = []
    head: list[str] = []
    if ev.source_type:
        head.append(f"**{ev.source_type}**")
    if ev.quality_score is not None:
        head.append(f"quality {ev.quality_score:.2f}")
    if head:
        parts.append(" · ".join(head))
    if ev.judgment_reasoning:
        parts.append(_sanitize_excerpt(ev.judgment_reasoning, max_chars=400))
    line = " — ".join(p for p in parts if p)
    if ev.source_ref:
        url = _source_url(ev.source_ref)
        label = _short_source(ev.source_ref)
        if url != ev.source_ref:
            line += f" · {_md_link(label, url)}"
        else:
            line += f" · {label}"
    return line


def _render_audit_trail_for_claim(claim: ClaimSummary) -> str:
    """Build the markdown body for the per-claim "How we got here"
    collapsible. Returns an empty string if the claim has no audit-trail
    content (no investigation rounds AND no IBE candidates)."""
    if not claim.investigation_rounds and not claim.ibe_candidates:
        return ""

    parts: list[str] = []

    if claim.investigation_rounds:
        parts.append("**Investigation rounds**")
        parts.append("")
        parts.append(
            "When initial gather couldn't fully resolve the claim, the "
            "gap-analysis agent proposed new methodological angles. Each "
            "intent below was routed to providers; the yield count is how "
            "many evidence items the routing returned."
        )
        parts.append("")
        for rnd in claim.investigation_rounds:
            plural = "" if rnd.evidence_count == 1 else "s"
            parts.append(
                f"- **Round {rnd.round_index}** "
                f"_(yielded {rnd.evidence_count} item{plural})_ — {rnd.intent}"
            )
        parts.append("")

    if claim.ibe_candidates:
        parts.append("**Alternative explanations considered (IBE chain)**")
        parts.append("")
        parts.append(
            "The integration step enumerated alternative explanations of "
            "the evidence pattern. Each was scored on **loveliness** (how "
            "well the explanation fits) and **likeliness** (prior "
            "probability). The candidate with the strongest combined "
            "score was selected as the integrated verdict."
        )
        parts.append("")
        rows: list[list[str]] = []
        for c in claim.ibe_candidates:
            status_parts: list[str] = []
            if c.chosen:
                status_parts.append("**selected**")
            elif c.runner_up:
                status_parts.append("runner-up")
            else:
                status_parts.append("rejected")
            status = " ".join(status_parts)
            lovel = f"{c.loveliness:.2f}" if c.loveliness is not None else "—"
            likl = f"{c.likeliness:.2f}" if c.likeliness is not None else "—"
            rows.append(
                [
                    c.candidate_id,
                    status,
                    c.verdict,
                    lovel,
                    likl,
                    _sanitize_excerpt(c.description, max_chars=200),
                ]
            )
        parts.append(
            _md_table(
                ["ID", "Status", "Verdict", "Lovel.", "Likl.", "Description"],
                rows,
            )
        )
        if claim.integrated_assessment:
            parts.append("")
            parts.append(
                f"**Integrated assessment**: {claim.integrated_assessment}"
            )

    return "\n".join(parts)


def _render_claim_section(
    r: Report,
    claim: ClaimSummary,
    all_evidence: list[EvidenceSummary],
    adv_by_claim: dict[str, list[Any]],
    *,
    show_label_prefix: str = "",
) -> None:
    """Render one claim's section: card + top supports + top counter-evidence +
    collapsible audit-trail.

    ``show_label_prefix`` is prepended to the claim text in cards when there
    are multiple claims (research mode), to make the report navigable.
    """
    supports, contradicts, no_bearing = _evidence_for_claim(claim, all_evidence)

    # The claim card — no inline numeric reference list. Counts are
    # rendered as compact prose in details. Audit trail (rounds + IBE)
    # goes in details as a collapsible block.
    card_details_parts: list[str] = []
    if claim.scope:
        card_details_parts.append(f"**Scope:** {claim.scope}")
    if claim.verification_summary:
        card_details_parts.append(f"**Verification:** {claim.verification_summary}")
    if claim.assumptions:
        card_details_parts.append(
            "**Assumptions:** " + "; ".join(claim.assumptions)
        )

    # Evidence summary counts on the claim card — replaces the
    # 98-number inline list of the classic layout.
    counts_line = (
        f"**Evidence:** {len(supports)} supporting, "
        f"{len(contradicts)} contradicting, "
        f"{len(no_bearing)} no bearing"
    )
    card_details_parts.append(counts_line)

    audit_md = _render_audit_trail_for_claim(claim)
    if audit_md:
        card_details_parts.append("")
        card_details_parts.append("---")
        card_details_parts.append("")
        card_details_parts.append(audit_md)

    card_kw: dict[str, Any] = {
        "badge": claim.stage,
        "id": _claim_slug(claim.claim_id),
    }
    if card_details_parts:
        card_kw["details"] = "\n\n".join(card_details_parts)

    statement = claim.statement
    if show_label_prefix:
        statement = f"{show_label_prefix} — {statement}"
    r.card(statement, **card_kw)

    # Top supporting evidence inline (3-5 items max).
    if supports:
        top_supports = supports[:_INLINE_TOP_K]
        body_lines = [_evidence_line(ev) for ev in top_supports]
        body_lines = [f"- {b}" for b in body_lines if b]
        if len(supports) > _INLINE_TOP_K:
            body_lines.append(
                f"\n_({len(supports) - _INLINE_TOP_K} additional supporting "
                "items in the appendix below.)_"
            )
        r.prose(
            "\n".join(body_lines),
            heading=f"Strongest supporting evidence ({len(supports)} total)",
            id=f"supports-{_claim_slug(claim.claim_id)}",
        )

    # Adversarial probe — counter-evidence framed as the explicit probe.
    counterargs = adv_by_claim.get(claim.claim_id, [])
    if contradicts or counterargs:
        body_parts: list[str] = []
        body_parts.append(
            "The system explicitly searched for evidence that would "
            "**contradict** this claim: replication failures, null "
            "results, and rival findings. Items below were either "
            "judged as contradicting by the evidence judge or surfaced "
            "by the adversarial-search step."
        )
        body_parts.append("")
        # Combine the two sources of contradicting evidence; dedup by
        # source_ref so a paper that appears in both lists shows once.
        seen: set[str] = set()
        lines: list[str] = []
        for ev in contradicts[:_INLINE_TOP_K]:
            key = ev.source_ref or f"ev-{ev.evidence_id}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {_evidence_line(ev)}")
        for adv in counterargs[:_INLINE_TOP_K]:
            key = adv.source_ref or f"adv-{id(adv)}"
            if key in seen:
                continue
            seen.add(key)
            content = _sanitize_excerpt(adv.counterargument, max_chars=400)
            if adv.source_ref:
                url = _source_url(adv.source_ref)
                label = _short_source(adv.source_ref)
                if url != adv.source_ref:
                    content += f" · {_md_link(label, url)}"
                else:
                    content += f" · {label}"
            lines.append(f"- {content}")
        body_parts.extend(lines)
        total = len(contradicts) + len(counterargs)
        if total > _INLINE_TOP_K:
            extra = max(0, total - _INLINE_TOP_K)
            body_parts.append("")
            body_parts.append(
                f"_({extra} additional challenge"
                f"{'s' if extra != 1 else ''} in the appendix.)_"
            )
        r.prose(
            "\n".join(body_parts),
            heading=f"Adversarial probe ({total} challenge"
            f"{'s' if total != 1 else ''})",
            id=f"adversarial-{_claim_slug(claim.claim_id)}",
        )


def _claim_slug(claim_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", claim_id.lower())


# ──────────────────────────────────────────────────────────────────────────────
# Top-level builder
# ──────────────────────────────────────────────────────────────────────────────


def build_audit_report(data: ReportData) -> list[dict[str, Any]]:
    """Build the audit-style report's atom list.

    Mirrors the shape of ``typeset_report.build_typeset_report`` —
    returns a list of typeset atom dicts ready to be passed to the
    typeset renderer. Caller produces HTML via ``typeset.render``.
    """
    r = Report(style="article")

    # ── Heading ─────────────────────────────────────────────────────────
    meta: dict[str, Any] = {
        "date": data.investigation_date.strftime("%Y-%m-%d"),
        "model": data.model_used,
    }
    if data.question_type:
        qt_label = QUESTION_TYPE_LABELS.get(
            data.question_type, data.question_type
        )
        meta["mode"] = qt_label
    r.heading(data.research_question, meta=meta)

    # ── Verdict callout — the headline pill ─────────────────────────────
    verdict_label = _verdict_label(data)
    posterior_pill = _posterior_pill(data)
    multi_claim = len(data.claims) > 1
    headline_parts: list[str] = [
        f"**{verdict_label}** — {posterior_pill}",
    ]
    if data.verdict:
        headline_parts.append(data.verdict)
    if multi_claim:
        headline_parts.append(
            f"_Decomposed into {len(data.claims)} sub-claims; combined "
            "verdict above. Per-sub-claim audit trails follow._"
        )
    r.callout("\n\n".join(headline_parts), tone=_verdict_tone(verdict_label))

    # ── Summary of findings (Cochrane-style table) ──────────────────────
    s = data.stats
    judged_total = s.evidence_supports + s.evidence_contradicts + s.evidence_no_bearing
    sof_rows = [
        [
            "Supporting",
            str(s.evidence_supports),
            _pct(s.evidence_supports, judged_total),
        ],
        [
            "Contradicting",
            str(s.evidence_contradicts),
            _pct(s.evidence_contradicts, judged_total),
        ],
        [
            "No bearing _(retained for audit, not weighted into verdict)_",
            str(s.evidence_no_bearing),
            _pct(s.evidence_no_bearing, judged_total),
        ],
    ]
    sof_lines: list[str] = [
        f"The system retrieved **{s.total_evidence} evidence items** "
        f"and judged each piece against the claim. The directional split:",
        "",
        _md_table(["Direction", "Items", "Share"], sof_rows),
    ]
    if s.investigation_rounds_total:
        sof_lines.append("")
        sof_lines.append(
            f"_{s.investigation_rounds_total} investigation round"
            f"{'s' if s.investigation_rounds_total != 1 else ''} across "
            f"{len([c for c in data.claims if c.investigation_rounds])} "
            "claim(s) with audit trail visible below._"
        )
    r.prose(
        "\n".join(sof_lines),
        heading="Summary of findings",
        id="summary-of-findings",
    )

    # ── Plain-language summary ──────────────────────────────────────────
    if data.direct_answer:
        r.prose(
            data.direct_answer,
            heading="Plain-language summary",
            id="plain-language-summary",
        )

    # ── Per-claim sections ──────────────────────────────────────────────
    adv_by_claim: dict[str, list[Any]] = {}
    for adv in data.adversarial:
        adv_by_claim.setdefault(adv.claim_id, []).append(adv)

    if multi_claim:
        # Research mode — each sub-claim becomes its own audit card.
        r.prose(
            f"The question was decomposed into {len(data.claims)} sub-claims, "
            "each investigated separately. Below: each sub-claim's verdict, "
            "key supporting and counter-evidence, and the methodological "
            "trail the system followed.",
            heading="Sub-investigations",
            id="sub-investigations",
        )
        for i, claim in enumerate(data.claims, start=1):
            _render_claim_section(
                r,
                claim,
                data.evidence,
                adv_by_claim,
                show_label_prefix=f"#{i}",
            )
    else:
        # Verify mode — one claim, no decomposition.
        r.prose(
            "Below: the system's verdict, the strongest supporting "
            "evidence, the strongest counter-evidence the adversarial "
            "probe surfaced, and the audit trail.",
            heading="Key evidence",
            id="key-evidence",
        )
        for claim in data.claims:
            _render_claim_section(r, claim, data.evidence, adv_by_claim)

    # ── Caveats & Limitations ───────────────────────────────────────────
    unresolved = [u for u in data.uncertainties if not u.is_resolved]
    caveats = [u for u in unresolved if not u.is_blocking]
    blocking = [u for u in unresolved if u.is_blocking]

    if caveats:
        r.prose(
            "\n".join(f"- {u.description}" for u in caveats),
            heading="Caveats",
            id="caveats",
        )

    if blocking:
        r.prose(
            "\n".join(f"- {u.description}" for u in blocking),
            heading="Limitations",
            id="limitations",
        )

    if data.open_questions:
        r.prose(
            "\n".join(f"1. {q}" for q in data.open_questions),
            heading="Open questions",
            id="open-questions",
        )

    # ── Appendix: full evidence trail (collapsible) ─────────────────────
    if data.evidence:
        appendix_parts: list[str] = []
        groups = [
            ("Supporting evidence", [e for e in data.evidence if e.support_judgment == "supports"]),
            ("Contradicting evidence", [e for e in data.evidence if e.support_judgment == "contradicts"]),
            (
                "Evidence judged as having no bearing on the claim",
                [e for e in data.evidence if e.support_judgment == "no_bearing"],
            ),
        ]
        for group_label, group in groups:
            if not group:
                continue
            appendix_parts.append(f"### {group_label} ({len(group)})")
            appendix_parts.append("")
            for ev in group:
                appendix_parts.append(f"- {_evidence_line(ev)}")
            appendix_parts.append("")

        r.card(
            "**Full evidence trail** — every retrieved item with its "
            "one-sentence judgement and clickable source.",
            badge="audit",
            id="appendix-evidence",
            details="\n".join(appendix_parts),
        )

    return r.atoms


# ──────────────────────────────────────────────────────────────────────────────
# Public — sidebar of styles
# ──────────────────────────────────────────────────────────────────────────────


__all__ = [
    "build_audit_report",
]
