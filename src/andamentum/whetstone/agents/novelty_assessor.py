"""Novelty-assessor agent.

Wraps deep_research's ``NoveltyAssessment`` schema in a v2 AgentDefinition
so the ``NoveltyCheck`` node can use deep_research's existing
``check_novelty`` function as-is. This is the ``assess_fn`` half of the
``check_novelty`` interface — given a claim plus retrieved evidence
(summary + key findings + sources), decide whether the claim is novel.

We re-use the ``NoveltyAssessment`` schema directly from deep_research
so the wire format stays interoperable. This is the only file in v2
that imports a deep_research data type at module level — everywhere
else uses runtime-only imports to keep the dep graph clean.
"""

from __future__ import annotations

from andamentum.deep_research.novelty.checker import NoveltyAssessment

from ._definition import AgentDefinition


NOVELTY_ASSESSOR_PROMPT = """You are assessing whether a CLAIM is novel given evidence retrieved from web search.

You receive:
  • claim — the claim to verify
  • evidence_summary — what the search returned, summarised
  • key_findings — list of specific findings extracted from the search
  • sources — list of URLs the search consulted

Decide and return a NoveltyAssessment:

  • is_novel — True if the claim represents new ground; False if prior
    work already established it (or something close to it)
  • confidence — float 0.0-1.0; how sure are you of is_novel?
  • assessment — 2-3 sentences explaining your decision. Reference
    specific prior work if you flag the claim as not novel.
  • similar_works — list of dicts with keys: title, url, relevance,
    summary. relevance is one of "direct" (same claim), "partial"
    (overlapping), "tangential" (loosely related). Up to 5 items;
    most-relevant first. Empty list is valid if nothing similar found.

Important:
  • Be honest. If the search returned nothing relevant, say is_novel=True
    with low confidence — don't manufacture certainty.
  • If the search returned strong direct hits, say is_novel=False with
    high confidence and cite them.
  • If the search returned tangentially related work, say is_novel=True
    with moderate confidence and note the partial overlap in assessment."""


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="novelty_assessor",
        prompt=NOVELTY_ASSESSOR_PROMPT,
        output_model=NoveltyAssessment,
        retries=2,
        output_retries=2,
    )


NOVELTY_ASSESSOR_AGENT = _build()
