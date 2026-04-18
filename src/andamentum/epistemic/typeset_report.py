"""Adapter: convert epistemic ReportData to typeset atom list.

Thin mapping from the existing ReportData dataclass (produced by
report_generator.py) to a list of atom dicts consumable by
andamentum.typeset.render(). No data loading, no database access —
just field-to-atom mapping.
"""

from __future__ import annotations

from typing import Any

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
            f"Posterior: {cs.posterior:.2%}" if cs.posterior is not None else "No posterior computed"
        )
        qa_entries.append({"label": "How confident are we?", "body": conf_body})

        qa_entries.append({
            "label": "How thorough was the investigation?",
            "body": f"{data.stats.total_evidence} evidence sources examined",
        })

    r.items(entries=qa_entries)

    # ── Summary narrative ────────────────────────────────────────────

    if data.direct_answer:
        summary_parts: list[str] = []

        summary_parts.append(
            f"**Research Question:** *{data.research_question}*"
        )

        providers_str = f"**Evidence Sources:** {data.stats.total_evidence}"
        claims_supported = sum(
            1 for c in data.claims if c.stage.lower() in ("supported", "robust", "provisional", "actionable")
        )
        providers_str += f" | **Claims Established:** {claims_supported} of {data.stats.total_claims}"
        summary_parts.append(providers_str)

        summary_parts.append(data.direct_answer)
        r.prose("\n\n".join(summary_parts), heading="Summary")

    # ── Claims as cards ──────────────────────────────────────────────

    if data.claims:
        r.prose(
            f"The investigation produced {len(data.claims)} distinct findings, "
            f"each traced to specific evidence sources. "
            f"Findings are ordered by strength of support.",
            heading="Findings",
        )

    for claim in data.claims:
        refs = [str(n) for n in claim.evidence_refs_display]

        details_parts: list[str] = []
        if claim.scope:
            details_parts.append(f"**Scope:** {claim.scope}")
        if claim.verification_summary:
            details_parts.append(f"**Verification:** {claim.verification_summary}")
        if claim.assumptions:
            details_parts.append(
                "**Assumptions:** " + "; ".join(claim.assumptions)
            )

        card_kw: dict[str, Any] = {"badge": claim.stage}
        if refs:
            card_kw["refs"] = refs
        if details_parts:
            card_kw["details"] = "\n\n".join(details_parts)

        r.card(claim.statement, **card_kw)

    # ── Evidence as references ───────────────────────────────────────

    if data.evidence:
        r.prose("", heading="Sources")

        # Group by judgment
        supporting = [e for e in data.evidence if e.support_judgment == "supports"]
        contradicting = [e for e in data.evidence if e.support_judgment == "contradicts"]
        other = [
            e for e in data.evidence
            if e.support_judgment not in ("supports", "contradicts")
        ]

        for group_label, group in [
            ("Supporting", supporting),
            ("Contradicting", contradicting),
            ("Other", other),
        ]:
            if not group:
                continue
            r.prose(f"#### {group_label}")

            for ev in group:
                number = data.evidence_index_map.get(ev.evidence_id)
                content_parts: list[str] = []

                # Source type + judgment header
                header_parts: list[str] = []
                if ev.source_type:
                    header_parts.append(ev.source_type)
                if ev.provider:
                    header_parts.append(ev.provider)

                # Reasoning or extracted content
                if ev.judgment_reasoning:
                    content_parts.append(ev.judgment_reasoning)
                elif ev.extracted_content:
                    content_parts.append(ev.extracted_content)

                # Limitations
                if ev.limitations:
                    content_parts.append(
                        "*" + "; ".join(ev.limitations) + "*"
                    )

                ref_kw: dict[str, Any] = {}
                if number is not None:
                    ref_kw["number"] = number
                if ev.source_ref:
                    ref_kw["source"] = ev.source_ref
                if ev.support_judgment:
                    ref_kw["badge"] = ev.support_judgment
                if ev.quality_score is not None:
                    # Append quality to the header line
                    content_parts.insert(
                        0,
                        " ".join(header_parts) if header_parts else ""
                    )
                    if header_parts:
                        content_parts[0] += f"  quality: {ev.quality_score:.2f}"
                elif header_parts:
                    content_parts.insert(0, " ".join(header_parts))

                r.reference(
                    "\n\n".join(p for p in content_parts if p),
                    **ref_kw,
                )

    # ── Adversarial counterarguments ─────────────────────────────────

    if data.adversarial:
        claim_ids_with_adv = {a.claim_id for a in data.adversarial}
        for claim in data.claims:
            if claim.claim_id not in claim_ids_with_adv:
                continue
            claim_advs = [
                a for a in data.adversarial if a.claim_id == claim.claim_id
            ]
            if not claim_advs:
                continue

            r.prose(
                f"#### Counterarguments to: *{claim.statement[:80]}...*"
                if len(claim.statement) > 80
                else f"#### Counterarguments to: *{claim.statement}*"
            )
            for adv in claim_advs:
                ref_kw_adv: dict[str, Any] = {"badge": "contradicts"}
                if adv.source_ref:
                    ref_kw_adv["source"] = adv.source_ref
                r.reference(adv.counterargument, **ref_kw_adv)

    # ── Uncertainties / limitations ──────────────────────────────────

    unresolved = [u for u in data.uncertainties if not u.is_resolved]
    if unresolved:
        caveats = [
            u for u in unresolved
            if u.uncertainty_type in ("caveat", "limitation", "scope_limitation")
        ]
        blocking = [u for u in unresolved if u.is_blocking and u not in caveats]

        parts: list[str] = []
        if caveats:
            parts.append("**Caveats**\n\n" + "\n".join(
                f"- {u.description}" for u in caveats
            ))
        if blocking:
            parts.append("**Unresolved blocking issues**\n\n" + "\n".join(
                f"- {u.description}" for u in blocking
            ))
        if parts:
            r.prose("\n\n".join(parts), heading="Limitations")

    # ── Open questions ───────────────────────────────────────────────

    if data.open_questions:
        r.prose(
            "\n".join(f"1. {q}" for q in data.open_questions),
            heading="Open Questions",
        )

    # ── Investigation process ────────────────────────────────────────

    if data.investigation_narrative:
        r.prose(data.investigation_narrative, heading="Investigation Process")

    # ── Confidence sidebar ───────────────────────────────────────────

    sidebar_groups: dict[str, dict[str, str]] = {
        "Investigation": {
            "Evidence": str(data.stats.total_evidence),
            "Claims": f"{data.stats.total_claims}",
            "Uncertainties": str(
                data.stats.blocking_uncertainties
                + data.stats.non_blocking_uncertainties
                + data.stats.resolved_uncertainties
            ),
        },
    }

    if data.confidence_scores:
        cs = data.confidence_scores
        confidence_data: dict[str, str] = {}
        if cs.posterior is not None:
            confidence_data["Posterior"] = f"{cs.posterior:.2%}"
            confidence_data["Claims supported"] = str(cs.posterior_supporting)
            confidence_data["Claims contradicted"] = str(cs.posterior_contradicting)
        else:
            confidence_data["Posterior"] = "N/A"
        sidebar_groups["Confidence"] = confidence_data

    sidebar_groups["Model"] = {"LLM": data.model_used}

    r.aside(groups=sidebar_groups)

    return r.atoms
