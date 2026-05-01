"""Prompt construction for ``critique_figure``.

Single helper, isolated so the prompt text is easy to inspect, tweak,
and snapshot in tests. Kept terse on purpose — small local vision models
follow short imperative prompts more reliably than chatty ones.
"""

from __future__ import annotations


_BASE_PROMPT = """\
You are reviewing a rendered scientific figure for layout and \
readability problems only — not the data, not the choice of chart \
type, not the colour palette unless it actively hurts legibility.

Look at the attached image and produce the requested critique.

Rules:
- Be specific. If labels overlap, say so.
- Suggested fixes must come from the allowed set in the schema.
- Use 'no_change_needed' only when the figure has no real issues.
- Confidence is your honest 0..1 read; an obvious problem is ~1.0, a \
borderline call is ~0.5.
"""


def build_prompt(*, extra_context: str | None = None) -> str:
    """Build the user prompt for a critique call.

    The schema itself is communicated via pydantic-ai's structured-output
    machinery — this prompt only adds high-level intent and any
    caller-supplied context.
    """
    parts = [_BASE_PROMPT.strip()]
    if extra_context:
        parts.append(f"\nContext: {extra_context.strip()}")
    return "\n".join(parts)
