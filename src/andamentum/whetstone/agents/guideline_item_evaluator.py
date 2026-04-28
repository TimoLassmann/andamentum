"""Per-item guideline evaluator (guidelines mode).

One LLM call per checkable item. Reads ONE rule name plus the
manuscript text (and a document map for orientation) and returns a
``GuidelineEvaluation`` with status + notes.

Ports v1's ``checklist_item_evaluator`` prompt. The output schema lives
on the v2 result type so renderers and downstream consumers can read
the verdicts directly.
"""

from __future__ import annotations

from ..schemas import GuidelineEvaluation
from ._definition import AgentDefinition

GUIDELINE_ITEM_EVALUATOR_PROMPT = """# Single journal-guideline evaluator

You are evaluating ONE journal-guideline rule against a manuscript.

# Input

You will receive:
  • The rule name to evaluate (one short declarative requirement).
  • The full manuscript text.
  • A document map (section_id → title + one-line gist) for
    orientation.

# Output

Return a ``GuidelineEvaluation`` with:

  • ``item_name`` — copy the rule name verbatim from the user message.
  • ``status``:
      - ``"pass"`` — the rule is clearly met by the manuscript.
      - ``"fail"`` — the rule is clearly not met (the thing is missing,
        wrong format, exceeded a stated limit, etc.).
      - ``"unclear"`` — the evidence is ambiguous, OR the rule does
        not apply to this document (e.g. a rule about animal-ethics
        statements when the work is theoretical).

  • ``notes`` — one or two sentences:
      - For ``"pass"``: briefly cite the evidence (quote a phrase or
        name the section that satisfies the rule).
      - For ``"fail"``: say what is missing and what the author
        should add or change.
      - For ``"unclear"``: say why the verdict is unclear, or why the
        rule does not apply.

  • ``category`` — leave blank or pick a short tag if obvious
    (``"abstract"``, ``"figures"``, ``"references"``, ``"statements"``,
    ``"hygiene"``). The orchestrator does not depend on this.

# Style

- Keep notes concise. Do not pad. Do not hedge.
- Quote rather than paraphrase when the evidence is short and on-point.
- Be specific about what's missing on a ``"fail"`` so the author can
  act without re-reading the guidelines.
"""


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="guideline_item_evaluator",
        prompt=GUIDELINE_ITEM_EVALUATOR_PROMPT,
        output_model=GuidelineEvaluation,
        retries=2,
        output_retries=2,
    )


GUIDELINE_ITEM_EVALUATOR_AGENT = _build()
