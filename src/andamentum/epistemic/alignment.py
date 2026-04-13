"""Unified alignment validation for the epistemic pipeline.

Detects semantic drift at three stages: question clarification,
assertion extraction, and claim drafting. Uses a single pydantic-ai
agent with dynamic system prompts via dependency injection.

Architecture: Cross-cutting validation infrastructure. Lives outside
the agent registry because it serves all pipeline stages uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from pydantic import BaseModel, Field


@dataclass
class AlignmentCheck:
    """Dependencies for the alignment validator agent.

    Passed via pydantic-ai's deps mechanism. The check_type field
    selects which mode-specific instructions the agent receives.
    """

    check_type: str  # "clarification", "assertion", or "claim"
    research_question: str  # original user question
    output_to_validate: str  # the text being checked
    context: str  # mode-specific context


class AlignmentResult(BaseModel):
    """Output from the alignment validator."""

    aligned: bool = Field(description="Whether the output is aligned with the research question")
    issue: str = Field(default="", description="What drifted, if not aligned")
    suggestion: str = Field(default="", description="How to fix it, if not aligned")


# ── Mode-specific instructions ────────────────────────────────────────────

_CLARIFICATION_INSTRUCTIONS = """\
## Clarification Check

Compare the clarified question against the original. Check three things:

1. **Same subject?** The clarified question must be about the same thing as the original. \
If the user asked about topic X, the clarification must be about X, not a related topic Y.

2. **Same breadth?** If the original question was broad, the clarification must stay broad. \
Narrowing a general question to one specific aspect is not faithful.

3. **No invented constraints?** The clarification must not add demographic groups, time periods, \
geographies, or methodological requirements that the user did not mention.

Set aligned=true if same subject, same breadth, no invented constraints. \
Set aligned=false if any of these are violated. Explain what drifted in `issue` \
and suggest a correction in `suggestion`."""

_ASSERTION_INSTRUCTIONS = """\
## Assertion Check

Compare the extracted assertion against the evidence it came from and the research question.

1. **States a finding?** The assertion must say what the evidence FOUND, concluded, or measured. \
If it merely describes that a study exists, was conducted, or evaluated something, it is not aligned. \
An assertion must report an outcome, result, or conclusion.

2. **Relevant to the research question?** The assertion must bear on what the user asked. \
An assertion about an unrelated finding is not aligned.

3. **Specific enough to challenge?** The assertion should be concrete enough that someone could \
search for counterevidence. If it is trivially true (e.g., "a trial was performed"), \
it cannot be challenged and is not aligned.

Set aligned=true only if all three checks pass. When aligned=false, explain which check \
failed in `issue` and suggest how to rewrite the assertion as a proper finding in `suggestion`."""

_CLAIM_INSTRUCTIONS = """\
## Claim Check

Compare the drafted claim against the research question and the assertions it was built from.

1. **Falsifiable?** Could someone find evidence against this claim? If it is a methodological \
truism, a statement about what research exists, or a description of study design, it is not \
a valid claim. A claim must make a testable statement about reality.

2. **Addresses the research question?** The claim must help answer what the user asked. \
A claim that describes the research landscape without taking a position is not aligned.

3. **Same breadth as the question?** A broad question should produce claims about the broad \
topic. A claim that only addresses one narrow sub-study without connecting to the broader \
question is not aligned.

Set aligned=true only if all three checks pass. When aligned=false, explain which check \
failed in `issue` and suggest how to rewrite the claim as a falsifiable statement in `suggestion`."""

_MODE_INSTRUCTIONS = {
    "clarification": _CLARIFICATION_INSTRUCTIONS,
    "assertion": _ASSERTION_INSTRUCTIONS,
    "claim": _CLAIM_INSTRUCTIONS,
}


def _get_mode_instructions(check_type: str) -> str:
    """Return mode-specific check instructions for the given check type."""
    instructions = _MODE_INSTRUCTIONS.get(check_type)
    if instructions is None:
        raise ValueError(f"Unknown check_type: {check_type!r}. Must be one of: {sorted(_MODE_INSTRUCTIONS)}")
    return instructions


# ── Agent construction and public API ─────────────────────────────────────

_BASE_PROMPT = """\
You verify that outputs of the epistemic system remain aligned with the user's \
original research question. You check whether the output faithfully serves the \
question's intent without drifting, narrowing, or becoming unfalsifiable.

You will receive:
- research_question: The user's original question
- output_to_validate: The text to check
- context: Additional information relevant to this check

Apply the mode-specific checks below. Be strict: if the output drifts from the \
research question's intent in any way, mark it as not aligned."""


def _build_agent(model: Any) -> Any:
    """Build the alignment validator agent with dynamic system prompt."""
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        model,
        system_prompt=_BASE_PROMPT,
        deps_type=AlignmentCheck,
        output_type=AlignmentResult,
        retries=2,
        output_retries=3,
    )

    @agent.system_prompt
    def inject_mode_instructions(ctx: RunContext[AlignmentCheck]) -> str:
        return _get_mode_instructions(ctx.deps.check_type)

    return agent


# Module-level agent cache keyed by model identity
_agent_cache: dict[int, Any] = {}


async def validate_alignment(
    check_type: str,
    research_question: str,
    output_to_validate: str,
    context: str,
    model: Any,
) -> AlignmentResult:
    """Validate that a pipeline output is aligned with the research question.

    Args:
        check_type: "clarification", "assertion", or "claim"
        research_question: The user's original question
        output_to_validate: The text to check for drift
        context: Mode-specific context (evidence content, reasoning, assertions)
        model: pydantic-ai model (string or resolved model object)

    Returns:
        AlignmentResult with aligned=True/False and optional issue/suggestion
    """
    cache_key = id(model) if not isinstance(model, str) else hash(model)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = _build_agent(model)

    agent = _agent_cache[cache_key]
    deps = AlignmentCheck(
        check_type=check_type,
        research_question=research_question,
        output_to_validate=output_to_validate,
        context=context,
    )

    user_message = (
        f"research_question: {research_question}\n"
        f"output_to_validate: {output_to_validate}\n"
        f"context: {context}"
    )

    result = await agent.run(user_message, deps=deps)
    return result.output
