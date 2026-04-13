"""Novelty assessment agent definition."""

from ..novelty.checker import NoveltyAssessment
from . import AgentDefinition, register_agent

# ── Novelty Assessor ────────────────────────────────────────────────────

NOVELTY_ASSESSMENT_PROMPT = """\
You are a novelty assessor. Given a claim and research findings about prior work,
assess whether the claim is novel.

Guidelines:
- is_novel=True if no prior work directly addresses the claim
- is_novel=False if prior work already covers the claim substantially
- confidence: 0.0-1.0 based on evidence quality and completeness
- For each similar work found, specify:
  - relevance: "direct" (same claim), "partial" (related), or "tangential" (loosely related)
  - summary: Brief explanation of how it relates

Be conservative: if prior work exists that addresses the core of the claim, it's NOT novel."""

register_agent(AgentDefinition(
    name="novelty_assessor",
    prompt=NOVELTY_ASSESSMENT_PROMPT,
    output_model=NoveltyAssessment,
    retries=3,
    output_retries=5,
))


def build_assessment_prompt(
    claim: str, evidence_summary: str, key_findings: list[str], sources: list[str]
) -> str:
    """Build the user-message prompt for novelty assessment."""
    return f"""Assess novelty of this claim:

CLAIM: {claim}

RESEARCH FINDINGS:
{evidence_summary}

KEY FINDINGS:
{chr(10).join(f"- {f}" for f in key_findings) if key_findings else "None"}

SOURCES FOUND:
{chr(10).join(f"- {s}" for s in sources) if sources else "None"}

Based on this research, is the claim novel? Identify any similar prior work."""
