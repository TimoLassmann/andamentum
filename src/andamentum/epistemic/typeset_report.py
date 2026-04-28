"""Adapter: convert epistemic ReportData to typeset atom list.

Thin mapping from the existing ReportData dataclass (produced by
report_generator.py) to a list of atom dicts consumable by
andamentum.typeset.render(). No data loading, no database access —
just field-to-atom mapping.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from andamentum.typeset import Report

from .html_report import ReportData


QUESTION_TYPE_LABELS: dict[str, str] = {
    "verificatory": "yes/no factual question",
    "explanatory": "explanation or mechanism question",
    "exploratory": "open-ended exploration",
    "comparative": "comparison question",
    "predictive": "prediction or forecast",
    "methodological": "methodology question",
    "normative": "value judgment or recommendation",
}

STAGE_DISPLAY_ORDER: tuple[str, ...] = (
    "supported",
    "provisional",
    "robust",
    "actionable",
    "hypothesis",
    "abandoned",
)


def _short_source(url: str) -> str:
    """Return a concise display label for a source URL.

    Prefers DOI, PMC id, or arXiv id when recognisable; otherwise returns
    the hostname plus the last path segment if it's short.
    """
    if not url:
        return ""
    m = re.search(r"(10\.\d{4,9}/\S+)$", url)
    if m:
        return m.group(1).rstrip("/")
    m = re.search(r"/(PMC\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"arxiv\.org/abs/(\S+)$", url)
    if m:
        return f"arXiv:{m.group(1)}"
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = parsed.netloc.replace("www.", "")
    last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if last and len(last) < 40 and not last.isdigit():
        return f"{host}/{last}"
    return host or url


def _sanitize_excerpt(text: str, max_chars: int = 800) -> str:
    """Make raw scraped markdown safe for a reference body.

    Strips markdown/HTML that would pollute the document outline:
    - removes heading prefixes (``# Foo`` → ``Foo``)
    - unwraps ``[label](url)`` to ``label``
    - drops raw URLs and HTML tags
    - collapses whitespace, preserves paragraph breaks
    - truncates to *max_chars* with an ellipsis on a word boundary

    The ``max_chars=800`` default is a UX choice (readable digest length),
    not a legal requirement. The report is a private single-user research
    artifact with attributed quotations, which sits comfortably inside
    Australian fair dealing for research (Copyright Act §40) and US fair
    use. Revisit this budget only if the report is ever published or
    redistributed beyond the user.
    """
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    paras = [re.sub(r"\s+", " ", p).strip() for p in cleaned.split("\n\n")]
    result = "\n\n".join(p for p in paras if p)
    if len(result) > max_chars:
        truncated = result[:max_chars]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        result = truncated + "…"
    return result


def _claim_slug(claim_id: str) -> str:
    """Short stable id for in-page anchoring."""
    return f"claim-{claim_id.split('-', 1)[0]}"


def build_typeset_report(data: ReportData) -> list[dict[str, Any]]:
    """Convert a ReportData into a typeset atom list.

    Args:
        data: The fully-populated ReportData from ReportGenerator.

    Returns:
        List of atom dicts ready for ``andamentum.typeset.render()``.
    """
    r = Report(style="article")

    # ── Heading ──────────────────────────────────────────────────────

    meta: dict[str, str] = {
        "date": data.investigation_date.strftime("%Y-%m-%d %H:%M"),
        "model": data.model_used,
        "project": data.database_name,
    }
    qt_label = QUESTION_TYPE_LABELS.get(data.question_type or "", "")
    r.heading(data.research_question, meta=meta)

    # ── Verdict ──────────────────────────────────────────────────────

    if data.verdict:
        r.callout(data.verdict)

    # ── Question type ────────────────────────────────────────────────

    if qt_label:
        r.callout(f"This is a {qt_label}", tone="note")

    # ── Posterior interpretation ─────────────────────────────────────

    if (
        data.confidence_scores
        and data.confidence_scores.terminal_state == "retrieval_failed"
    ):
        r.callout(
            "**Retrieval failed** — evidence extraction returned empty content "
            "repeatedly (3+ times consecutively). The posterior is uninformative; "
            "the investigation could not gather enough data to form a reasoned answer.",
            tone="warning",
        )
    elif data.confidence_scores and data.confidence_scores.posterior is not None:
        posterior_pct = data.confidence_scores.posterior * 100
        cs = data.confidence_scores
        interp = (
            f"**P(YES) ≈ {posterior_pct:.1f}%** — "
            f"{cs.posterior_supporting} supporting vs "
            f"{cs.posterior_contradicting} contradicting evidence points."
        )
        r.callout(interp, tone="info")

    # ── Key Findings Q&A ─────────────────────────────────────────────

    qa_entries: list[dict[str, str]] = [
        {"label": "What was studied?", "body": data.clarified_question},
    ]
    if data.verdict:
        challenged = sum(
            1
            for c in data.claims
            if c.adversarial_balance is not None and c.adversarial_balance < 0.3
        )
        verdict_body = data.verdict
        if challenged:
            verdict_body += f". {challenged} challenged by counter-evidence"
        qa_entries.append({"label": "What did we find?", "body": verdict_body})

    if data.confidence_scores:
        cs = data.confidence_scores
        conf_body = (
            f"Posterior: {cs.posterior:.2%}"
            if cs.posterior is not None
            else "No posterior computed"
        )
        qa_entries.append({"label": "How confident are we?", "body": conf_body})

        qa_entries.append(
            {
                "label": "How thorough was the investigation?",
                "body": f"{data.stats.total_evidence} evidence sources examined",
            }
        )

    r.items(entries=qa_entries)

    # ── Summary narrative ────────────────────────────────────────────

    if data.direct_answer:
        summary_parts: list[str] = []

        summary_parts.append(f"**Research Question:** *{data.research_question}*")

        providers_str = f"**Evidence Sources:** {data.stats.total_evidence}"
        claims_supported = sum(
            1
            for c in data.claims
            if c.stage.lower() in ("supported", "robust", "provisional", "actionable")
        )
        providers_str += f" | **Claims Established:** {claims_supported} of {data.stats.total_claims}"
        summary_parts.append(providers_str)

        summary_parts.append(data.direct_answer)
        r.prose("\n\n".join(summary_parts), heading="Summary", id="summary")

    # ── Claims as cards, each followed by its counterarguments ───────

    if data.claims:
        r.prose(
            f"The investigation produced {len(data.claims)} distinct findings, "
            f"each traced to specific evidence sources. "
            f"Findings are ordered by strength of support.",
            heading="Findings",
            id="findings",
        )

    adv_by_claim: dict[str, list[Any]] = {}
    for adv in data.adversarial:
        adv_by_claim.setdefault(adv.claim_id, []).append(adv)

    for claim in data.claims:
        refs = [str(n) for n in claim.evidence_refs_display]

        details_parts: list[str] = []
        if claim.scope:
            details_parts.append(f"**Scope:** {claim.scope}")
        if claim.verification_summary:
            details_parts.append(f"**Verification:** {claim.verification_summary}")
        if claim.assumptions:
            details_parts.append("**Assumptions:** " + "; ".join(claim.assumptions))

        card_kw: dict[str, Any] = {
            "badge": claim.stage,
            "id": _claim_slug(claim.claim_id),
        }
        if refs:
            card_kw["refs"] = refs
        if details_parts:
            card_kw["details"] = "\n\n".join(details_parts)

        r.card(claim.statement, **card_kw)

        # Consecutive run of reference atoms with shared group → clustered under a heading
        claim_advs = adv_by_claim.get(claim.claim_id, [])
        if claim_advs:
            short_stmt = claim.statement[:70].rstrip()
            if len(claim.statement) > 70:
                short_stmt += "…"
            group_label = f"Counterarguments — {short_stmt}"
            for adv in claim_advs:
                ref_kw: dict[str, Any] = {
                    "badge": "contradicts",
                    "group": group_label,
                }
                if adv.source_ref:
                    ref_kw["source"] = adv.source_ref
                    ref_kw["source_label"] = _short_source(adv.source_ref)
                r.reference(_sanitize_excerpt(adv.counterargument), **ref_kw)

    # ── Supporting evidence as references ────────────────────────────

    supporting_ev = [e for e in data.evidence if e.support_judgment == "supports"]
    contradicting_ev = [e for e in data.evidence if e.support_judgment == "contradicts"]
    other_ev = [
        e
        for e in data.evidence
        if e.support_judgment not in ("supports", "contradicts")
    ]

    if data.evidence:
        r.prose("", heading="Sources", id="sources")

        for group_label, group in [
            ("Supporting", supporting_ev),
            ("Contradicting", contradicting_ev),
            ("Other", other_ev),
        ]:
            if not group:
                continue
            for ev in group:
                number = data.evidence_index_map.get(ev.evidence_id)

                header_parts: list[str] = []
                if ev.source_type:
                    header_parts.append(ev.source_type)
                if ev.provider:
                    header_parts.append(ev.provider)
                header_line = " ".join(header_parts)
                if ev.quality_score is not None and header_line:
                    header_line += f"  quality: {ev.quality_score:.2f}"

                content_parts: list[str] = []
                if header_line:
                    content_parts.append(header_line)
                if ev.judgment_reasoning:
                    content_parts.append(ev.judgment_reasoning)
                if ev.limitations:
                    content_parts.append("*" + "; ".join(ev.limitations) + "*")

                ref_kw_ev: dict[str, Any] = {"group": group_label}
                if number is not None:
                    ref_kw_ev["number"] = number
                if ev.source_ref:
                    ref_kw_ev["source"] = ev.source_ref
                    ref_kw_ev["source_label"] = _short_source(ev.source_ref)
                if ev.support_judgment:
                    ref_kw_ev["badge"] = ev.support_judgment

                r.reference(
                    "\n\n".join(p for p in content_parts if p),
                    **ref_kw_ev,
                )

    # ── Caveats (non-blocking unresolved) + Limitations (blocking) ──
    # Matches legacy's split: non-blocking → Caveats, blocking → Limitations.

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

    # ── Open questions ───────────────────────────────────────────────

    if data.open_questions:
        r.prose(
            "\n".join(f"1. {q}" for q in data.open_questions),
            heading="Open Questions",
            id="open-questions",
        )

    # ── Investigation process ────────────────────────────────────────

    if data.investigation_narrative:
        r.prose(
            data.investigation_narrative,
            heading="Investigation Process",
            id="investigation",
        )

    # ── Sidebar ──────────────────────────────────────────────────────

    stage_counts = data.stats.claims_by_stage or {}
    claims_group: dict[str, str] = {"Total": str(data.stats.total_claims)}
    for stage in STAGE_DISPLAY_ORDER:
        n = stage_counts.get(stage, 0)
        if n:
            claims_group[stage.title()] = str(n)

    sidebar_groups: dict[str, dict[str, str]] = {
        "Investigation": {
            "Evidence": str(data.stats.total_evidence),
            "Claims": str(data.stats.total_claims),
            "Uncertainties": str(
                data.stats.blocking_uncertainties
                + data.stats.non_blocking_uncertainties
                + data.stats.resolved_uncertainties
            ),
        },
        "Claims by stage": claims_group,
    }

    if data.confidence_scores:
        cs = data.confidence_scores
        confidence_data: dict[str, str] = {}
        if cs.posterior is not None:
            confidence_data["P(YES)"] = f"{cs.posterior:.2%}"
            confidence_data["Supporting"] = str(cs.posterior_supporting)
            confidence_data["Contradicting"] = str(cs.posterior_contradicting)
        else:
            confidence_data["Posterior"] = "N/A"
        sidebar_groups["Confidence"] = confidence_data

    sidebar_groups["Model"] = {"LLM": data.model_used}

    r.aside(groups=sidebar_groups)

    return r.atoms
