"""Tier-1 document-type classifier.

Decides whether the harvested text is academic writing, external
communication, or general — driving downstream choices about which
deterministic checks run and what vocabulary the synthesis agent uses.

Single LLM call per run. Same model as the rest of the pipeline (no
separate router-model setting), resolved through
``core.models.resolve_model`` like every other agent in whetstone — see
:func:`classify` for why that is load-bearing rather than incidental.

Failure is split in two, deliberately. A model that answered something
unusable is a genuine classification failure and falls back to
``"general"`` — the safest neutral default, with no deterministic backup
and no hidden heuristic. Anything else (provider unreachable,
misconfigured, out of credit) propagates: it is not a statement about
the document, and every call after this one is about to fail the same
way.

Public API: ``classify(model, sections, markdown) -> DocumentType``.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

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
    model: str | None,
    section_titles: list[str],
    markdown: str,
) -> DocumentType:
    """Classify document type via one LLM call.

    *model* is a model id string (``"ollama:gemma4:26b-nvfp4"``,
    ``"openai:gpt-5.4-nano"``), resolved through
    :func:`~andamentum.core.models.resolve_model` exactly as every other
    agent in whetstone resolves its own. Handing the raw string to
    ``Agent()`` instead looks equivalent and is not: the resolver is what
    supplies provider setup the bare constructor has no way to know about
    — for ``ollama:`` ids that includes the base URL, so a raw string
    failed on every local model while the rest of the pipeline ran fine.
    ``None`` skips the call and returns ``"general"``.

    Returns ``"general"`` when the model answered something unusable —
    a genuine classification failure, logged and absorbed.

    Raises
    ------
    Exception
        Anything that is not a classification failure — provider
        unreachable, misconfigured, out of credit — propagates
        unchanged. Such an error says nothing about the document, and
        every LLM call after this one is about to fail identically;
        surfacing it at the first call is strictly better than degrading
        here and failing N calls later with the cause buried. This is the
        no-silent-failures rule: a misconfiguration must not be
        indistinguishable from an inconclusive answer.
    """
    if model is None:
        return "general"

    # Imported lazily to match the rest of this module: it is on the import
    # path of the CLI, which must stay cheap for the no-LLM subcommands.
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from andamentum.core.models import resolve_model

    body_sample = markdown[:_BODY_SAMPLE_CHARS]
    prompt = _build_prompt(section_titles, body_sample)

    agent: Agent[None, DocumentTypeDecision] = Agent(
        resolve_model(model), output_type=DocumentTypeDecision
    )
    try:
        result = await agent.run(prompt)
        decision = result.output
    except (UnexpectedModelBehavior, ValidationError) as exc:
        # The model was reached and answered; the answer was unusable
        # (retries exhausted, off-schema output, content filter). That IS
        # a classification failure, so the neutral default applies.
        logger.warning(
            "[document_type] classifier could not produce a usable answer "
            "(%s); defaulting to 'general'",
            exc,
        )
        return "general"

    logger.info(
        "[document_type] classified as %s — %s",
        decision.document_type,
        decision.reasoning,
    )
    return decision.document_type
