"""Aggregate per-paper results into the decision-grade readout.

The headline is the count of issues the whole-document arm (B) caught that
whetstone (A) missed AND that are both critical and cross-section — the
architecture gaps. Everything else is context: noise rates, the local misses
(lens bugs, not architecture), and how often A's synthesis matched B's verdict.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .types import AdjudicatedFinding, PaperResult


class PaperReadout(BaseModel):
    slug: str
    both: int = 0
    a_only: int = 0
    b_only: int = 0
    # the headline subset
    b_only_critical_crosssection: int = 0
    # other diagnostic slices
    b_only_critical_local: int = 0  # whetstone SHOULD have caught (lens bug)
    a_only_total: int = 0  # whetstone-only: real catches or noise
    verdict_match: bool | None = None


class Readout(BaseModel):
    papers: list[PaperReadout] = Field(default_factory=list)
    # aggregates
    headline_arch_gaps: int = 0  # Σ b_only & critical & cross_section
    total_b_only_critical: int = 0
    total_local_misses: int = 0
    total_both: int = 0
    total_a_only: int = 0
    verdict_match_rate: float | None = None


def _is(
    f: AdjudicatedFinding, bucket: str, sev: str | None = None, loc: str | None = None
) -> bool:
    return (
        f.bucket == bucket
        and (sev is None or f.severity == sev)
        and (loc is None or f.locality == loc)
    )


def summarise_paper(result: PaperResult) -> PaperReadout:
    adj = result.adjudications
    return PaperReadout(
        slug=result.paper.slug,
        both=sum(1 for f in adj if f.bucket == "both"),
        a_only=sum(1 for f in adj if f.bucket == "a_only"),
        b_only=sum(1 for f in adj if f.bucket == "b_only"),
        b_only_critical_crosssection=sum(
            1 for f in adj if _is(f, "b_only", "critical", "cross_section")
        ),
        b_only_critical_local=sum(
            1 for f in adj if _is(f, "b_only", "critical", "local")
        ),
        a_only_total=sum(1 for f in adj if f.bucket == "a_only"),
        verdict_match=result.verdict_match,
    )


def aggregate(results: list[PaperResult]) -> Readout:
    papers = [summarise_paper(r) for r in results]
    matches = [p.verdict_match for p in papers if p.verdict_match is not None]
    return Readout(
        papers=papers,
        headline_arch_gaps=sum(p.b_only_critical_crosssection for p in papers),
        total_b_only_critical=sum(
            p.b_only_critical_crosssection + p.b_only_critical_local for p in papers
        ),
        total_local_misses=sum(p.b_only_critical_local for p in papers),
        total_both=sum(p.both for p in papers),
        total_a_only=sum(p.a_only for p in papers),
        verdict_match_rate=(sum(matches) / len(matches)) if matches else None,
    )


def render_markdown(readout: Readout) -> str:
    """Human-readable readout."""
    lines = [
        "# Whetstone evaluation — readout",
        "",
        f"**Architecture gaps** (critical, cross-section issues B caught & A missed): "
        f"**{readout.headline_arch_gaps}**",
        "",
        f"- Total critical issues B-only: {readout.total_b_only_critical} "
        f"(of which {readout.total_local_misses} are *local* — lens bugs, not architecture)",
        f"- Found by both arms: {readout.total_both}",
        f"- Whetstone-only (A-only): {readout.total_a_only}",
    ]
    if readout.verdict_match_rate is not None:
        lines.append(
            f"- Synthesis matched B's central weaknesses: "
            f"{readout.verdict_match_rate:.0%} of papers"
        )
    lines += [
        "",
        "## Per paper",
        "",
        "| paper | both | A-only | B-only | arch-gap | local-miss | verdict✓ |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in readout.papers:
        vm = "—" if p.verdict_match is None else ("yes" if p.verdict_match else "no")
        lines.append(
            f"| {p.slug} | {p.both} | {p.a_only} | {p.b_only} | "
            f"{p.b_only_critical_crosssection} | {p.b_only_critical_local} | {vm} |"
        )
    return "\n".join(lines) + "\n"
