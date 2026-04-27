"""Challenge agent: try to refute one finding, return a verdict.

Tiniest possible schema (decision + reason). The agent re-reads the
sections cited by the finding and asks: "is this finding actually true?"

Three verdicts:
  • stand    — the finding holds; keep as is
  • weaken   — the finding is partly right but overstated; lower confidence
  • withdraw — the finding is wrong (false positive); drop it

Default to "stand" when uncertain. Withdraws are the most consequential,
so the prompt requires explicit textual evidence before allowing them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

CHALLENGE_PROMPT = """You are a challenger. A finding has been made about a document.
Your job: re-read the cited sections and decide if the finding holds.

You have:
  • the finding (title, rationale, severity, confidence, quotes)
  • the FULL TEXT of every section the finding cites

Three possible verdicts:

  verdict = "stand"
    → The finding is correct. Keep as is.

  verdict = "weaken"
    → The finding has some truth but is overstated, missing nuance, or
      the severity is too high. Explain WHY in `reason`.

  verdict = "withdraw"
    → The finding is wrong. The cited evidence does NOT support it, OR
      important context elsewhere in the cited sections refutes it.
      Explain WHY in `reason` and quote the refuting evidence.

Default to "stand" when uncertain. Only "withdraw" if the section text
demonstrably refutes the finding — never withdraw on vibes."""


class ChallengeVerdict(BaseModel):
    """challenge_agent's flat output."""

    verdict: Literal["stand", "weaken", "withdraw"] = Field(
        description="What to do with the finding."
    )
    reason: str = Field(
        default="",
        description=(
            "Why you reached this verdict. For 'withdraw', cite the "
            "refuting evidence. Maximum 3 sentences."
        ),
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="challenge",
        prompt=CHALLENGE_PROMPT,
        output_model=ChallengeVerdict,
        retries=2,
        output_retries=2,
    )


CHALLENGE_AGENT = _build()
