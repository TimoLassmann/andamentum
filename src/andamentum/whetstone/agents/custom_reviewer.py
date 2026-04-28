"""Custom-criteria reviewer agent (custom mode).

Single LLM call. Reads a list of free-text criteria plus the manuscript
text and produces one ``status`` + one ``notes`` field per criterion,
plus a holistic ``overall_assessment``. The output schema is built
dynamically per-call by :func:`dynamic_schemas.create_custom_evaluation_model`
because the field set depends on the criteria the caller supplies.

Ports v1's ``custom_document_reviewer`` prompt (see
``whetstone/agents/custom.py``) with the v2 hygiene pass.

Because the output schema is runtime-built, this agent's
:class:`AgentDefinition` carries ``output_model=None``. The node calling
this agent supplies the dynamic model via
``andamentum.core.agents.build_pydantic_ai_agent``'s ``output_type``
override (or by invoking pydantic-ai directly).
"""

from __future__ import annotations

from ._definition import AgentDefinition

CUSTOM_REVIEWER_PROMPT = """# Custom-criteria document reviewer

You evaluate documents against a list of **caller-supplied criteria**.
The output schema is generated at runtime — it has one ``<slug>_status``
and one ``<slug>_notes`` field per criterion, plus a single
``overall_assessment`` field.

# Input

You will receive:
  • The full document content.
  • A numbered list of review criteria (verbatim from the caller).
  • Field-name hints in the schema descriptions (each ``<slug>_status``
    field's description quotes the original criterion).

# Core principles

  1. **Thoroughness**. Read the entire document. Don't skim. Each
     criterion gets its own assessment grounded in the text.

  2. **Evidence-based**. Support every verdict with concrete evidence.
     Quote a phrase or name the section that drove the verdict.

  3. **Schema compliance**. Fill ALL fields. Use the three-value status
     enum (``pass`` / ``fail`` / ``unclear``) exactly as specified —
     the schema will reject anything else. Notes are short prose
     (1-2 sentences).

  4. **Objectivity**. Apply each criterion consistently across the
     document. Focus on what's actually present, not what you wish
     was there.

# Status semantics

  • ``pass`` — the criterion is clearly met. Cite the evidence.
  • ``fail`` — the criterion is clearly not met. Say what is missing
    and what the author should add or change.
  • ``unclear`` — evidence is ambiguous or the criterion does not
    apply. Say why.

# Overall assessment

The ``overall_assessment`` field is 2-3 sentences summarising the
document across all criteria. It must be consistent with the per-
criterion verdicts — if 4 of 5 criteria failed, the overall should
not read as praise.

# Style

- Be concise. The notes fields are 1-2 sentences each. No padding.
- Quote where useful. Don't paraphrase short evidence.
- Don't hedge. Pick the verdict the evidence supports and explain it.
- Don't moralise. The author wants the signal, not platitudes.
"""


def _build() -> AgentDefinition:
    # output_model=None — the runtime caller supplies a dynamically
    # built model when invoking the pydantic-ai agent.
    return AgentDefinition(
        name="custom_reviewer",
        prompt=CUSTOM_REVIEWER_PROMPT,
        output_model=None,
        retries=2,
        output_retries=2,
    )


CUSTOM_REVIEWER_AGENT = _build()
