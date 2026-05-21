"""Claim-substantiation agent: the REDUCE step, verified against full text.

Given one contribution claim and the full text of the document's body
sections, decide whether the claim is substantiated — by the work's own
results/data OR by a citation — and say briefly why. Support is a reasoning
task, so the model reads the actual text (no embedding/similarity shortcut).

Flat output (a bool + a short reason) so small local models fill it reliably.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

CLAIM_SUPPORT_PROMPT = """You are checking whether a CLAIM a document makes \
about its own contribution is substantiated by the document itself.

You are shown one claim and the full text of the document's substantive \
sections (results, methods, discussion, etc.).

Decide whether the claim is substantiated, where substantiated means EITHER:
  • the document's own results / data / experiments support it, OR
  • it is backed by a citation.

  supported = true
    → The text contains evidence (a result, number, experiment, or citation) \
that backs the claim.

  supported = false
    → Nothing in the text substantiates the claim: there is no result, data, \
or citation that supports it. (e.g. a "robustness" claim with no experiment \
testing robustness; a comparative claim with no comparison reported.)

In `reason`, say briefly (one sentence) what support you found, or what is \
missing. When genuinely unsure, answer true (do not flag a claim whose support \
is plausibly present)."""


class ClaimSupport(BaseModel):
    """claim_support's flat output."""

    supported: bool = Field(
        description=(
            "True if the document's text substantiates the claim (by data or "
            "citation); false if nothing supports it. Default to true when unsure."
        ),
    )
    reason: str = Field(
        default="",
        description="One sentence: what support was found, or what is missing.",
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="claim_support",
        prompt=CLAIM_SUPPORT_PROMPT,
        output_model=ClaimSupport,
        retries=2,
        output_retries=2,
    )


CLAIM_SUPPORT_AGENT = _build()
