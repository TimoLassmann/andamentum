"""Worker: evaluate research completeness via the ``gap_analyzer`` agent.

Engine-free (L2): summaries in, a ``GapAnalysis`` out. The ``AnalyzeGaps``
node owns the refine-vs-synthesize routing that branches on the result.
"""

from __future__ import annotations

from typing import Any

from .build_agent import AgentOverrides, build_agent
from .models import GapAnalysis, PageSummary


async def analyze_gaps(
    *,
    goal: str,
    summaries: list[PageSummary],
    model: Any,
    overrides: AgentOverrides | None = None,
) -> GapAnalysis:
    """Ask the ``gap_analyzer`` agent whether the evidence answers ``goal``."""
    agent = build_agent("gap_analyzer", model, overrides)

    evidence = [
        f"{s.title} (relevance {s.relevance_score:.2f}): {s.summary}"
        for s in summaries
    ]
    sources = [s.url for s in summaries]

    prompt = f"""Research Question: {goal}

Evidence Gathered ({len(evidence)} items):
{chr(10).join(f"- {e}" for e in evidence)}

Sources Consulted ({len(sources)}):
{chr(10).join(f"- {s}" for s in sources)}

Evaluate:
1. Does this evidence comprehensively answer the research question?
2. What specific information is missing?
3. What targeted searches would fill the gaps?

If research is complete, explain why. If gaps exist, be specific about what's missing."""

    result = await agent.run(prompt)
    gap_analysis: GapAnalysis = result.output
    return gap_analysis
