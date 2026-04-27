"""Stage 3 — LLM judge for grey-zone boundaries.

Optional. Only invoked for cuts whose semantic-drop percentile falls in the
grey zone (60–90th by default). The judge sees a small window of context
on each side of the proposed cut and answers `keep | merge | move_left | move_right`.

Per the project rule "schemas FLAT and SIMPLE because small local models
fill them", the output is a single enum-ish field with a 1-line reason.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

# An executor with the same signature the rest of the codebase uses, so
# callers can re-use whatever they already wired up for AgentRunner.
ExecutorFn = Callable[..., Awaitable[object]]

JUDGE_SYSTEM_PROMPT = """You are a boundary judge for a document chunker.

You see a proposed cut between two adjacent text blocks (LEFT and RIGHT).
Decide whether the cut should be kept, the two blocks merged, or
ignored — based on whether they discuss the SAME or DIFFERENT topics.

Rules:
- Output ONE of: "keep", "merge".
- "keep" — LEFT ends a topic; RIGHT begins a different one. Cut is good.
- "merge" — LEFT and RIGHT continue the same topic / paragraph. Cut is wrong.
- Default to "keep" when uncertain — over-segmentation is preferred to
  silently merging two distinct topics.
"""


class JudgeVerdict(BaseModel):
    """LLM judge's decision on one proposed cut."""

    decision: Literal["keep", "merge"] = Field(
        description="`keep` if the two blocks should remain separate, `merge` if they're the same topic."
    )
    reason: str = Field(
        default="",
        description="One short sentence explaining the call (for telemetry, optional).",
    )


def _build_judge_prompt(left: str, right: str, *, ctx_chars: int = 400) -> str:
    """Assemble the user prompt with bounded context on each side."""
    left_ctx = left[-ctx_chars:] if len(left) > ctx_chars else left
    right_ctx = right[:ctx_chars] if len(right) > ctx_chars else right
    # Leading "..." marks truncation so the model knows the snippet isn't the
    # whole story and shouldn't read a full-stop into a missing one.
    left_marker = "…" if len(left) > ctx_chars else ""
    right_marker = "…" if len(right) > ctx_chars else ""
    return f"""Should this cut be kept (different topics) or merged (same topic)?

--- LEFT (ends here) ---
{left_marker}{left_ctx}
--- end LEFT ---

--- RIGHT (begins here) ---
{right_ctx}{right_marker}
--- end RIGHT ---

Decision: keep or merge?"""


async def judge_cut(
    *,
    executor: ExecutorFn,
    source: str,
    cut_offset: int,
    ctx_chars: int = 400,
) -> JudgeVerdict:
    """Ask the LLM judge whether the cut at `cut_offset` should be kept.

    `executor` is the same signature used elsewhere — accepts kwargs
    `instructions`, `user_message`, `output_type`, optional `validators`.
    Defaults to "keep" if the LLM call fails (we never silently swallow
    user-affecting errors, but a judge call is purely advisory — failing
    it shouldn't crash the whole chunking run).
    """
    left = source[max(0, cut_offset - ctx_chars * 2) : cut_offset]
    right = source[cut_offset : cut_offset + ctx_chars * 2]
    prompt = _build_judge_prompt(left, right, ctx_chars=ctx_chars)

    try:
        result = await executor(
            instructions=JUDGE_SYSTEM_PROMPT,
            user_message=prompt,
            output_type=JudgeVerdict,
            validators=[],
        )
        if isinstance(result, JudgeVerdict):
            return result
        # Defensive: a fallback executor might return dict-like
        return JudgeVerdict(**result.__dict__)  # type: ignore[arg-type]
    except Exception as exc:
        # Default to keeping the cut when the judge fails. The caller can
        # inspect the reason field to flag this for telemetry.
        return JudgeVerdict(decision="keep", reason=f"judge unavailable: {exc}")
