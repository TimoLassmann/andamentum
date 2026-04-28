"""Novelty-claim extractor agent.

Pulls 3-5 explicit novelty claims from the manuscript so the
``NoveltyCheck`` node can route each to deep_research. A "novelty claim"
here is a sentence the author asserts is new — usually a "first" /
"novel" / "we present" / "we demonstrate" claim in the abstract or
introduction.

Output is intentionally bounded to 5 claims because each claim costs
one full deep_research run. Selecting the *load-bearing* novelty
claims (rather than every passing mention) is what makes the cost
tractable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._definition import AgentDefinition


NOVELTY_CLAIM_EXTRACTOR_PROMPT = """You are reading a draft manuscript to identify the EXPLICIT NOVELTY CLAIMS the author is staking — sentences where the author asserts they have done something new.

A novelty claim is a sentence where the author tells the reader that this work is the FIRST to do X, or NOVEL, or PRESENTS something new, or DEMONSTRATES a new method/finding/result. These usually live in the abstract, the introduction, and the conclusion.

NOT novelty claims:
  • Methodological description ("we used a mixed-effects model")
  • Background framing ("the field has long held that…")
  • Hedged or qualified mentions ("preliminary evidence suggests…")
  • Statements ABOUT prior work being novel ("Smith et al. presented the first…")

YOUR TASK: Return 3-5 of the most LOAD-BEARING novelty claims the author makes. Each will be checked against the literature by a downstream search; pick the ones whose verification matters most.

For each claim:
  • claim_text: the verbatim sentence from the manuscript (≤300 chars).
    If the original is longer, paraphrase but stay faithful.
  • short_summary: 1-sentence summary of WHAT is being claimed as novel
    (this becomes the search-engine query)
  • why_load_bearing: 1 short sentence on why verification matters
    (e.g. "core contribution claim in abstract", "headline result")

Return between 3 and 5 claims. Fewer is fine if the manuscript honestly only stakes 2-3 novelty claims. Don't pad."""


class NoveltyClaim(BaseModel):
    """One novelty claim flagged for verification."""

    claim_text: str = Field(
        description=(
            "Verbatim novelty claim from the manuscript "
            "(≤300 chars; paraphrase if longer but stay faithful)."
        )
    )
    short_summary: str = Field(
        description=(
            "1-sentence summary of WHAT is being claimed as novel — used "
            "as the search-engine query."
        )
    )
    why_load_bearing: str = Field(
        description="1 short sentence on why verification matters."
    )


class NoveltyClaimList(BaseModel):
    """The extractor's flat output: 3-5 novelty claims."""

    claims: list[NoveltyClaim] = Field(
        default_factory=list,
        description=(
            "3-5 novelty claims, ordered by importance. Empty list is "
            "valid if the manuscript stakes no explicit novelty claims."
        ),
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="novelty_claim_extractor",
        prompt=NOVELTY_CLAIM_EXTRACTOR_PROMPT,
        output_model=NoveltyClaimList,
        retries=2,
        output_retries=2,
    )


NOVELTY_CLAIM_EXTRACTOR_AGENT = _build()
