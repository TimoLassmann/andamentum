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

from .report_data import QUESTION_TYPE_LABELS, ReportData
from .thresholds import ADVERSARIAL_REFUTED_THRESHOLD

STAGE_DISPLAY_ORDER: tuple[str, ...] = (
    "supported",
    "provisional",
    "robust",
    "actionable",
    "hypothesis",
    "abandoned",
)


def _pct(numerator: int, denominator: int) -> str:
    """Render N/D as a percentage. Returns the empty string when D is 0
    so callers can drop the parenthetical cleanly."""
    if denominator <= 0:
        return "—"
    return f"{100 * numerator / denominator:.0f}%"


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
    #
    # Invariant: a directional posterior (P(YES)) is rendered ONLY when
    # ``terminal_state == "completed"``. Every non-completed terminal
    # gets a state-specific callout instead. Adding a new terminal in
    # ``confidence.py`` without a renderer branch falls into the
    # ``else`` arm and surfaces the raw state name — failing loud
    # rather than silently emitting a misleading 50% number.

    if data.confidence_scores:
        cs = data.confidence_scores
        ts = cs.terminal_state
        if ts == "completed" and cs.posterior is not None:
            posterior_pct = cs.posterior * 100
            interp = (
                f"**P(YES) ≈ {posterior_pct:.1f}%** — "
                f"{cs.posterior_supporting} supporting vs "
                f"{cs.posterior_contradicting} contradicting evidence points."
            )
            r.callout(interp, tone="info")
        elif ts == "retrieval_failed":
            r.callout(
                "**Retrieval failed** — evidence extraction returned empty content "
                "repeatedly (3+ times consecutively). The posterior is uninformative; "
                "the investigation could not gather enough data to form a reasoned answer.",
                tone="warning",
            )
        elif ts == "oscillation_detected":
            r.callout(
                "**No certified verdict** — the IBE chain "
                "(inference-to-best-explanation) did not certify a directional "
                "verdict for any active claim. The posterior is suspended at 0.5; "
                "counting evidence on uncertified claims is unsafe (run-to-run "
                "noise can flip direction with no real change in epistemic state).",
                tone="warning",
            )
        else:
            r.callout(
                f"**Inquiry terminated: {ts}** — no directional verdict rendered.",
                tone="warning",
            )

    # ── Key Findings Q&A ─────────────────────────────────────────────

    qa_entries: list[dict[str, str]] = [
        {"label": "What was studied?", "body": data.clarified_question},
    ]
    if data.verdict:
        challenged = sum(
            1
            for c in data.claims
            if c.adversarial_balance is not None
            and c.adversarial_balance < ADVERSARIAL_REFUTED_THRESHOLD
        )
        verdict_body = data.verdict
        if challenged:
            verdict_body += f". {challenged} challenged by counter-evidence"
        qa_entries.append({"label": "What did we find?", "body": verdict_body})

    if data.confidence_scores:
        cs = data.confidence_scores
        ts = cs.terminal_state
        # Same invariant as the posterior callout above: only emit a
        # directional probability when the inquiry actually completed.
        if ts == "completed" and cs.posterior is not None:
            conf_body = f"Posterior: {cs.posterior:.2%}"
        elif ts == "retrieval_failed":
            conf_body = "No posterior — retrieval failed before evidence converged."
        elif ts == "oscillation_detected":
            conf_body = "No posterior — no IBE-certified verdict."
        elif cs.posterior is None:
            conf_body = "No posterior computed"
        else:
            conf_body = f"No posterior — inquiry terminated: {ts}."
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

        # ── Investigation rounds (audit trail of how this claim was investigated) ──
        # Empty when the claim reached a verdict on initial gather alone.
        if claim.investigation_rounds:
            short_stmt = claim.statement[:70].rstrip()
            if len(claim.statement) > 70:
                short_stmt += "…"
            rounds_lines = [
                "Scrutiny flagged gaps after the initial gather. The "
                "gap-analysis agent proposed the following methodological "
                "angles across rounds; each was routed to providers, "
                "with the routing yield (how many evidence items came back) "
                "shown per round.",
            ]
            for rnd in claim.investigation_rounds:
                rounds_lines.append(
                    f"- **Round {rnd.round_index}** "
                    f"_(yielded {rnd.evidence_count} item"
                    f"{'s' if rnd.evidence_count != 1 else ''})_ — "
                    f"{rnd.intent}"
                )
            r.prose(
                "\n".join(rounds_lines),
                heading=f"How this claim was investigated — {short_stmt}",
                id=f"investigation-{_claim_slug(claim.claim_id)}",
            )

        # ── IBE chain candidates (alternative explanations the system considered) ──
        # Empty when the claim never reached IBE (cycle-capped, abandoned,
        # or insufficient evidence to enumerate candidates).
        if claim.ibe_candidates:
            short_stmt = claim.statement[:70].rstrip()
            if len(claim.statement) > 70:
                short_stmt += "…"
            ibe_lines = [
                "The integration step enumerated alternative explanations "
                "of the evidence and scored each on **loveliness** (how "
                "well the explanation fits) and **likeliness** (prior "
                "probability). The candidate with the best combined "
                "score was selected as the integrated verdict.",
                "",
            ]
            for c in claim.ibe_candidates:
                tag_parts: list[str] = []
                if c.chosen:
                    tag_parts.append("**selected**")
                elif c.runner_up:
                    tag_parts.append("_runner-up_")
                tag_parts.append(f"verdict: {c.verdict}")
                tag = " · ".join(tag_parts)
                score_parts: list[str] = []
                if c.loveliness is not None:
                    score_parts.append(f"loveliness {c.loveliness:.2f}")
                if c.likeliness is not None:
                    score_parts.append(f"likeliness {c.likeliness:.2f}")
                scores = ", ".join(score_parts) if score_parts else "unscored"
                ibe_lines.append(
                    f"- **Candidate {c.candidate_id}** ({tag}) — {c.description}  \n"
                    f"  _{scores}_"
                )
            if claim.integrated_assessment:
                ibe_lines.append("")
                ibe_lines.append(
                    f"**Integrated assessment**: {claim.integrated_assessment}"
                )
            r.prose(
                "\n".join(ibe_lines),
                heading=f"Inference to the best explanation — {short_stmt}",
                id=f"ibe-{_claim_slug(claim.claim_id)}",
            )

        # ── Adversarial probe: surface the probe itself, not just the result ──
        claim_advs = adv_by_claim.get(claim.claim_id, [])
        if claim_advs:
            short_stmt = claim.statement[:70].rstrip()
            if len(claim.statement) > 70:
                short_stmt += "…"
            probe_intro = (
                "The system explicitly searched for evidence that would "
                "**contradict** this claim — counter-evidence, replication "
                f"failures, and rival findings. {len(claim_advs)} challenge"
                f"{'s' if len(claim_advs) != 1 else ''} surfaced:"
            )
            r.prose(
                probe_intro,
                heading=f"Adversarial probe — {short_stmt}",
                id=f"adversarial-{_claim_slug(claim.claim_id)}",
            )
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
        # Evidence judgement breakdown — the audit-trail view of the
        # split. Stats are computed across all retrieved items (not just
        # the deduped/filtered subset that ends up rendered as
        # references), so the reader can compare "we asked for evidence
        # and got N items" against "X% were directionally informative."
        s = data.stats
        total_judged = (
            s.evidence_supports + s.evidence_contradicts + s.evidence_no_bearing
        )
        breakdown_lines = [
            f"The system retrieved **{s.total_evidence} evidence items** "
            f"and judged each against the claim:",
            "",
            f"- **{s.evidence_supports} supports** "
            f"({_pct(s.evidence_supports, total_judged)})",
            f"- **{s.evidence_contradicts} contradicts** "
            f"({_pct(s.evidence_contradicts, total_judged)})",
            f"- **{s.evidence_no_bearing} no bearing** — items that "
            f"matched the claim's lexicon but didn't directly address "
            f"its specific outcome / population / context "
            f"({_pct(s.evidence_no_bearing, total_judged)})",
        ]
        if s.evidence_invalidated:
            breakdown_lines.append(
                f"- _{s.evidence_invalidated} invalidated_ as cross-provider duplicates"
            )
        breakdown_lines.extend([
            "",
            "Each item below is shown with its one-sentence judgement "
            "and (where available) source link.",
        ])
        r.prose(
            "\n".join(breakdown_lines), heading="Sources", id="sources"
        )

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
