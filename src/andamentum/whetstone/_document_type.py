"""Tier-1 document-type classifier.

Decides whether the harvested text is academic writing, external
communication, or general — driving downstream choices about which
deterministic checks run and what vocabulary the synthesis agent uses.

Single LLM call per run. Same model as the rest of the pipeline (no
separate router-model setting). On any failure (no model configured,
network error, malformed output after pydantic-ai's own retries) the
caller defaults to ``"general"`` — no deterministic backup, no hidden
heuristic.

Public API: ``classify(model, sections, markdown) -> DocumentType``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("andamentum.whetstone")

DocumentType = Literal[
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
]
DOCUMENT_TYPES: tuple[DocumentType, ...] = (
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
)

# How much body text to send to the classifier alongside the section
# titles. 1500 chars is enough to disambiguate when titles are absent or
# misleading; small enough to be cheap on any provider.
_BODY_SAMPLE_CHARS: int = 1500


class DocumentTypeDecision(BaseModel):
    """Classifier output. Same shape as brain's RouterClassification."""

    document_type: DocumentType = Field(
        description=(
            "academic: manuscripts, theses, conference papers, white "
            "papers — scholarly writing intended for academic "
            "publication. external_communication: blog posts, LinkedIn "
            "articles, emails, op-eds, press releases — text written for "
            "a broad non-academic audience. essay: personal essays, "
            "narrative essays, opinion essays — first-person argument "
            "from observation or experience. tutorial: how-tos, "
            "technical walkthroughs, cookbooks — reader is trying to "
            "accomplish a task. creative: short fiction, memoir, "
            "narrative non-fiction — story craft is the substance. "
            "general: notes, drafts, technical documentation, internal "
            "writeups — anything that fits none of the above."
        )
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


def _build_prompt(section_titles: list[str], body_sample: str) -> str:
    titles_block = (
        "\n".join(f"- {t}" for t in section_titles if t.strip())
        if section_titles
        else "(no section headings detected)"
    )
    return (
        "Classify the document below as one of six categories: "
        "academic, external_communication, essay, tutorial, creative, "
        "or general. Return the category plus a one-sentence "
        "rationale.\n\n"
        "SECTION TITLES:\n"
        f"{titles_block}\n\n"
        "BODY SAMPLE (first part of the document):\n"
        "--- BEGIN ---\n"
        f"{body_sample}\n"
        "--- END ---"
    )


async def classify(
    *,
    model: Any,
    section_titles: list[str],
    markdown: str,
) -> DocumentType:
    """Classify document type via one LLM call.

    Returns the resolved category. Any failure (no model, network error,
    schema-validation failure after pydantic-ai retries) returns
    ``"general"`` — the safest neutral default. The exception is logged
    so the user can see what happened but never propagates.
    """
    if model is None:
        return "general"

    from pydantic_ai import Agent

    body_sample = markdown[:_BODY_SAMPLE_CHARS]
    prompt = _build_prompt(section_titles, body_sample)

    try:
        agent: Agent[None, DocumentTypeDecision] = Agent(
            model, output_type=DocumentTypeDecision
        )
        result = await agent.run(prompt)
        decision = result.output
        logger.info(
            "[document_type] classified as %s — %s",
            decision.document_type,
            decision.reasoning,
        )
        return decision.document_type
    except Exception as exc:
        logger.warning(
            "[document_type] classifier failed (%s); defaulting to 'general'",
            exc,
        )
        return "general"
