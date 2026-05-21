"""Section classifier: should a writing reviewer critique this section?

A small-LLM, flat-enum classifier that decides what is worth prose review and
what isn't — replacing brittle heading/keyword or regex detection. General
across document types and section positions; one cheap call per section.

Three categories:
  • review      — substantive prose the author wrote to make their case
                  (argument, methods, results, discussion, narrative). Critique it.
  • reference   — a bibliography / list of citations. Not prose; skip.
  • boilerplate — administrative or structured non-prose: acknowledgements,
                  author contributions, conflict/funding statements,
                  data-availability notices, raw data tables, code listings,
                  metadata. Not worth prose critique; skip.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

SECTION_CLASSIFIER_PROMPT = """You decide whether a writing reviewer should \
critique a section of a document.

You are shown a section's title and the start of its text. Choose ONE label:

  review
    → Substantive prose the author wrote to make their case: argument,
      explanation, methods, results, discussion, narrative — anything a
      reviewer should read closely and comment on.

  reference
    → A bibliography or list of citations / works cited. Not prose to critique.

  boilerplate
    → Administrative or structured content that is not the author's argument
      and not worth prose critique: acknowledgements, author contributions,
      conflict-of-interest or funding statements, data/code availability
      notices, raw data tables, code listings, or pure metadata.

When a section is mixed, judge by what dominates. When unsure between review
and the others, choose review (better to consider it than to skip real prose)."""


class SectionClass(BaseModel):
    """section_classifier's flat output — one label."""

    kind: Literal["review", "reference", "boilerplate"] = Field(
        description=(
            "review = substantive prose to critique; reference = a "
            "bibliography/citation list; boilerplate = administrative or "
            "structured non-prose. Default to review when unsure."
        ),
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="section_classifier",
        prompt=SECTION_CLASSIFIER_PROMPT,
        output_model=SectionClass,
        retries=2,
        output_retries=2,
    )


SECTION_CLASSIFIER_AGENT = _build()
