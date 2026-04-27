"""AuthorQuestion agent: turn an unresolved hypothesis into a sharp question.

Used when an investigator can't resolve a hypothesis from the document
text alone — the gap is something only the document's author can answer.
For users reviewing their own papers, these questions are the most
directly actionable output: each one points at a real ambiguity to fix.
"""

from __future__ import annotations


from pydantic import BaseModel, Field

from ._definition import AgentDefinition

AUTHOR_QUESTION_PROMPT = """An investigation could not resolve a hypothesis from the document alone.

You have:
  • the original hypothesis text
  • the relevant section text(s) the investigator already read
  • a brief note on what made the hypothesis unresolvable

Your job: formulate ONE sharp question for the document's author. The
question should point at a specific ambiguity that, once answered, would
let a reviewer reach a verdict.

Rules:
  • One question, one sentence. No multi-part questions.
  • Concrete, not vague. ("Is the sample size in §3 really 50, or 48 as
    §5 implies?" is good. "Could you clarify the methods?" is bad.)
  • Reference specific sections by id when possible.
  • In `why`: ONE sentence explaining what's unresolved (so the author
    knows why we're asking).

Return an AuthorQuestionOutput."""


class AuthorQuestionOutput(BaseModel):
    """author_question_agent's flat output."""

    question: str = Field(description="One sharp, specific question for the author.")
    why: str = Field(default="", description="One sentence: what's unresolved.")
    sections_involved: list[str] = Field(default_factory=list)


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="author_question",
        prompt=AUTHOR_QUESTION_PROMPT,
        output_model=AuthorQuestionOutput,
        retries=2,
        output_retries=2,
    )


AUTHOR_QUESTION_AGENT = _build()
