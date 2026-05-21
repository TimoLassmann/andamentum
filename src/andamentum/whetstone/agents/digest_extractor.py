"""Contribution-claim extractor: the MAP step of claim substantiation.

For ONE section, extract the claims the work makes ABOUT ITSELF — what the
system/method/study does, achieves, or outperforms. These are the load-bearing
assertions a reader is asked to believe; they are the ones worth checking for
support. Background facts, method mechanics, and definitions are NOT claims to
substantiate here.

Quote discipline: every claim carries a VERBATIM span and whether a citation
accompanies it. The node drops any claim whose quote can't be anchored in the
section text, so a hallucinated claim cannot enter the digest.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

DIGEST_EXTRACTOR_PROMPT = """You are extracting the claims a piece of work \
makes ABOUT ITSELF from ONE section, so they can later be checked for support.

Extract CONTRIBUTION CLAIMS: assertions about what THIS work/system/method/\
study does, achieves, enables, or outperforms — the things the authors want \
the reader to believe about their contribution. Examples: "our method recovers \
1.3-4.8x more associations than single-source baselines", "the system is \
robust to irrelevant input", "this is the first approach to do X".

For each claim give:
  • text: the claim in one short sentence.
  • quote: a VERBATIM span copied EXACTLY from the section that states it.
  • has_citation: true if a citation marker (e.g. [12], [@smith2020], \
"(Smith et al., 2020)") accompanies the claim.

Do NOT extract: background facts about the field, descriptions of how a \
method works (without a performance/capability claim), definitions, or \
section headings. Only claims the work makes about its own contribution.

Copy quotes EXACTLY — do not paraphrase or fix typos. If the section makes no \
contribution claims, return an empty list."""


class RawClaim(BaseModel):
    """One extracted contribution claim. Flat — small models fill it reliably."""

    text: str = Field(description="The claim in one short sentence.")
    quote: str = Field(
        description="A verbatim span copied exactly from the section text."
    )
    has_citation: bool = Field(
        description="True if a citation marker accompanies this claim."
    )


class SectionClaims(BaseModel):
    """digest_extractor's output for one section."""

    claims: list[RawClaim] = Field(default_factory=list)


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="digest_extractor",
        prompt=DIGEST_EXTRACTOR_PROMPT,
        output_model=SectionClaims,
        retries=2,
        output_retries=2,
    )


DIGEST_EXTRACTOR_AGENT = _build()
