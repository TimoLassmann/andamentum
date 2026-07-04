"""Worker: assemble the final ``EvidenceReport``.

Engine-free (L2). Owns every way a report gets assembled — one deep
module rather than four shallow ones, because they share the report
invariants (counter stamping, source dedup, degradation stamp):

- the lead-agent synthesis, with evidence-quality framing and a
  deterministic fallback when synthesis itself fails;
- the zero-pages bail-out (no LLM call);
- the iteration-limit early-exit report (pure);
- the L7 aggregate-loudness stamp — a run that skipped most of its work
  is not green.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.usage import UsageLimits

from .build_agent import AgentOverrides, build_agent
from .models import EvidenceReport, FetchedPage, PageSummary
from .reporter import NOOP_REPORTER, SearchReporter

logger = logging.getLogger(__name__)

# In-agent (client-level) retry ceiling for the synthesis call.
SYNTHESIS_REQUEST_LIMIT = 25

# When falling back after a synthesis failure, cap how many summaries /
# key points feed the hand-assembled report.
FALLBACK_SUMMARY_CAP = 10
FALLBACK_POINTS_PER_SUMMARY = 2


def build_iteration_limit_report(
    *,
    fetched_pages: list[FetchedPage],
    total_searches: int,
    total_pages_fetched: int,
    iteration_count: int,
) -> EvidenceReport:
    """Deterministic report for the max-iterations early exit."""
    sources = [f"{p.title} - {p.url}" for p in fetched_pages if p.is_relevant]
    if not sources:
        sources = ["Research incomplete - max iterations reached"]
    return EvidenceReport(
        evidence_summary="Max iterations reached. Research incomplete.",
        key_findings=["Incomplete research - iteration limit reached"],
        sources=sources,
        total_searches_performed=total_searches,
        total_pages_fetched=total_pages_fetched,
        iterations_required=iteration_count,
    )


def _stamp_degradation(
    report: EvidenceReport,
    *,
    search_attempts: int,
    search_failures: int,
    fetch_attempts: int,
    fetch_failures: int,
    threshold: float,
) -> EvidenceReport:
    """L7 aggregate loudness: flip the report to degraded when the
    soft-failure rate crosses ``threshold`` (a Deps value)."""
    reasons: list[str] = []
    if search_attempts > 0 and search_failures / search_attempts >= threshold:
        reasons.append(f"{search_failures}/{search_attempts} searches failed")
    if fetch_attempts > 0 and fetch_failures / fetch_attempts >= threshold:
        reasons.append(f"{fetch_failures}/{fetch_attempts} page fetches failed")
    if reasons:
        report.degraded = True
        report.degraded_reason = (
            "Aggregate soft-failure threshold crossed: " + "; ".join(reasons) + "."
        )
    return report


async def synthesize_report(
    *,
    goal: str,
    summaries: list[PageSummary],
    iteration_count: int,
    total_searches: int,
    total_pages_fetched: int,
    n_search_errors: int,
    n_fetch_errors: int,
    soft_failure_threshold: float,
    model: Any,
    overrides: AgentOverrides | None = None,
    reporter: SearchReporter = NOOP_REPORTER,
) -> EvidenceReport:
    """Synthesize the final report from the ranked page summaries.

    ``summaries`` is empty only when zero pages were fetched at all
    (search/fetch failures across the whole run). After the
    SummarizePages filter-removal change, low-relevance summaries are NO
    LONGER discarded — they're ranked and surfaced via the
    limited-evidence framing below.
    """
    max_relevance = max((s.relevance_score for s in summaries), default=0.0)
    reporter.synthesis_starting(
        n_summaries=len(summaries),
        max_relevance=max_relevance,
    )

    if not summaries:
        report = EvidenceReport(
            evidence_summary=(
                f"Research on '{goal}' could not gather "
                f"evidence — no pages were fetched. "
                f"Searches performed: {total_searches}; "
                f"fetch failures: {n_fetch_errors}."
            ),
            key_findings=["No pages were fetched"],
            sources=["No sources"],
            total_searches_performed=total_searches,
            total_pages_fetched=total_pages_fetched,
            iterations_required=iteration_count,
        )
        return _stamp_degradation(
            report,
            search_attempts=total_searches,
            search_failures=n_search_errors,
            fetch_attempts=total_pages_fetched + n_fetch_errors,
            fetch_failures=n_fetch_errors,
            threshold=soft_failure_threshold,
        )

    agent = build_agent("lead_agent", model, overrides)

    # Summaries are already sorted by relevance descending in
    # SummarizePages; preserve that order in the synthesis prompt.
    summaries_text = []
    for i, s in enumerate(summaries, 1):
        excerpts = ""
        if s.key_excerpts:
            excerpt_lines = "\n".join('  "' + e + '"' for e in s.key_excerpts)
            excerpts = "\n\nVerbatim Excerpts:\n" + excerpt_lines
        summaries_text.append(f"""
Source {i}: {s.title} ({s.url})
Relevance: {s.relevance_score:.2f}

Summary:
{s.summary}

Key Points:
{chr(10).join(f"  • {point}" for point in s.key_points)}{excerpts}
""")

    # Evidence-quality framing — tell the lead agent how confident the
    # underlying summarisation was, so it can frame the output
    # appropriately rather than over- or under-claiming.
    if max_relevance >= 0.6:
        quality_note = ""
    elif max_relevance >= 0.3:
        quality_note = (
            "\nEVIDENCE QUALITY: MODERATE. The page summaries are "
            "topically relevant but no single source directly answers "
            "the question. Synthesise carefully and flag any gaps.\n"
        )
    else:
        quality_note = (
            "\nEVIDENCE QUALITY: LIMITED. Every page summary scored "
            "below the relevance threshold (max relevance "
            f"{max_relevance:.2f}). The pages found may be tangential "
            "to the research question. Frame your output as 'partial "
            "findings' or 'limited evidence', acknowledge what the "
            "available pages actually cover, and explicitly note that "
            "the question was not directly answered by any source. "
            "Do NOT pad the answer with speculation — report only what "
            "the sources actually say.\n"
        )

    prompt = f"""Question: {goal}
{quality_note}
Research Process:
- Iterations: {iteration_count}
- Total Searches: {total_searches}
- Pages Fetched: {total_pages_fetched}
- Pages Summarized: {len(summaries)}
- Max Relevance Score: {max_relevance:.2f}

Page Summaries (sorted by relevance, highest first):
{"".join(summaries_text)}

Synthesise these into an EvidenceReport. Apply the writing rules in your
instructions, with hedging calibrated to the max relevance score above.
Sources must contain UNIQUE URLs only — if the same URL appears in
several summaries, list it once."""

    try:
        result = await agent.run(
            prompt, usage_limits=UsageLimits(request_limit=SYNTHESIS_REQUEST_LIMIT)
        )
        report: EvidenceReport = result.output

        if not report.sources or report.sources == ["No sources"]:
            report.sources = list(dict.fromkeys([s.url for s in summaries]))
        else:
            # Defence-in-depth: even if the agent populated sources, dedupe
            # — the synthesis prompt instructs unique URLs, but small models
            # sometimes repeat. Order is preserved (dict.fromkeys keeps
            # first-occurrence order).
            report.sources = list(dict.fromkeys(report.sources))

        report.total_searches_performed = total_searches
        report.total_pages_fetched = total_pages_fetched
        report.iterations_required = iteration_count

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        key_findings: list[str] = []
        for s in summaries[:FALLBACK_SUMMARY_CAP]:
            key_findings.extend(s.key_points[:FALLBACK_POINTS_PER_SUMMARY])

        report = EvidenceReport(
            evidence_summary=f'Research on "{goal}" completed. Automatic synthesis failed: {e}',
            key_findings=key_findings if key_findings else ["Synthesis failed"],
            sources=list(dict.fromkeys([s.url for s in summaries])),
            total_searches_performed=total_searches,
            total_pages_fetched=total_pages_fetched,
            iterations_required=iteration_count,
        )

    return _stamp_degradation(
        report,
        search_attempts=total_searches,
        search_failures=n_search_errors,
        fetch_attempts=total_pages_fetched + n_fetch_errors,
        fetch_failures=n_fetch_errors,
        threshold=soft_failure_threshold,
    )
