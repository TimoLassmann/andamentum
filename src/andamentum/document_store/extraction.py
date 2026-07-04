"""LLM-based metadata extraction using PydanticAI structured output.

Uses PydanticAI agents with output_type to extract typed metadata.
The Pydantic model's field descriptions are passed directly to the LLM
as tool definitions — no manual JSON prompting needed.

Self-correction for local models:
- output_retries=5: PydanticAI feeds validation errors back to the model
- @output_validator: semantic checks with feedback messages
- PromptedOutput fallback: if a model ignores tool definitions entirely
  (UnexpectedModelBehavior), retry with schema injected into the system
  prompt so the model produces raw JSON instead of tool calls

Document extraction: title, projects, people.
Chunk extraction: topics (2-3 tags), people, has_decision (bool), has_action_item (bool).
  Boolean questions are much more reliable for local models than multi-way classification.

Requires: pydantic-ai (installed as part of andamentum).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from .metadata_models import (
    ChunkLLMFields,
    ChunkMetadataFields,
    DocumentLLMFields,
    DocumentMetadataFields,
)

logger = logging.getLogger(__name__)

#: Return type of an extraction (DocumentMetadataFields / ChunkMetadataFields).
OutT = TypeVar("OutT")

_DOC_SYSTEM_PROMPT = (
    "You are a metadata extraction agent for a personal knowledge base. "
    "Extract structured metadata from the document provided. "
    "Only extract what is explicitly stated. Be specific with project names and people. "
    "The title must be a one-line summary of max 10 words."
)

_CHUNK_SYSTEM_PROMPT = (
    "You are a metadata extraction agent for a personal knowledge base. "
    "Extract structured metadata from the text chunk provided. "
    "Only extract what is explicitly stated. "
    'Be specific with topics — prefer "MAP-Elites selection" over "optimization". '
    "For has_decision: true only if the text contains an explicit decision or commitment. "
    "For has_action_item: true only if the text contains an explicit to-do or next step."
)

_OUTPUT_RETRIES = 5
_RETRIES = 3


def _build_doc_agent(model: str):  # type: ignore[no-untyped-def]
    """Build a PydanticAI agent for document metadata extraction."""
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent = Agent(
        model,
        system_prompt=_DOC_SYSTEM_PROMPT,
        output_type=DocumentLLMFields,
        retries={"tools": _RETRIES, "output": _OUTPUT_RETRIES},
    )

    @agent.output_validator
    async def validate_doc_output(
        ctx: RunContext[None], output: DocumentLLMFields
    ) -> DocumentLLMFields:
        issues: list[str] = []

        if output.title and len(output.title.split()) > 15:
            issues.append(
                f"Title has {len(output.title.split())} words — shorten to max 10 words."
            )

        if not output.title:
            issues.append(
                "Title is empty — provide a one-line summary of the document."
            )

        if issues:
            raise ModelRetry("\n".join(issues))

        return output

    return agent


def _build_chunk_agent(model: str):  # type: ignore[no-untyped-def]
    """Build a PydanticAI agent for chunk metadata extraction."""
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent = Agent(
        model,
        system_prompt=_CHUNK_SYSTEM_PROMPT,
        output_type=ChunkLLMFields,
        retries={"tools": _RETRIES, "output": _OUTPUT_RETRIES},
    )

    @agent.output_validator
    async def validate_chunk_output(
        ctx: RunContext[None], output: ChunkLLMFields
    ) -> ChunkLLMFields:
        issues: list[str] = []

        if len(output.topics) > 5:
            issues.append(
                f"Too many topics ({len(output.topics)}) — provide 2-3 specific tags."
            )

        if issues:
            raise ModelRetry("\n".join(issues))

        return output

    return agent


async def _extract_with_fallback(
    *,
    model: str,
    input_text: str,
    build_agent: Callable[[str], Any],
    system_prompt: str,
    llm_output_type: type,
    project: Callable[[Any], OutT],
    default_factory: Callable[[], OutT],
    label: str,
) -> OutT:
    """Run a structured-output extraction with the standard local-model fallback.

    Shared by :func:`extract_document_metadata` and :func:`extract_chunk_metadata`
    — the two differ only in prompt, output type, input, and the LLM→public-model
    projection. The control flow is identical:

    1. Run the primary (tool-based) agent and project its output.
    2. If the model ignores tool definitions entirely (``UnexpectedModelBehavior``),
       retry once with :class:`PromptedOutput` (schema injected into the prompt).
    3. On any other failure — or a failed prompted retry — log and return defaults.

    ``ImportError`` (pydantic-ai missing) is re-raised as a clear ``RuntimeError``.
    """
    try:
        agent = build_agent(model)
        result = await agent.run(input_text)
        return project(result.output)
    except ImportError:
        raise RuntimeError(
            "pydantic-ai not installed. Install with: pip install andamentum"
        )
    except Exception as first_error:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        if not isinstance(first_error, UnexpectedModelBehavior):
            logger.warning(f"{label} metadata extraction failed: {first_error}")
            return default_factory()

        logger.info(
            f"{label} metadata: tool-based output failed, retrying with prompted output"
        )

        try:
            from pydantic_ai import Agent, PromptedOutput

            prompted_agent = Agent(
                model,
                system_prompt=system_prompt,
                output_type=PromptedOutput(llm_output_type),
                retries={"tools": _RETRIES, "output": _OUTPUT_RETRIES},
            )
            result = await prompted_agent.run(input_text)
            return project(result.output)
        except Exception as e:
            logger.warning(
                f"{label} metadata extraction failed after prompted retry: {e}"
            )
            return default_factory()


async def extract_document_metadata(
    content: str,
    model: str | None = None,
    max_content_chars: int = 3000,
) -> DocumentMetadataFields:
    """Extract LLM fields for a document using PydanticAI structured output.

    Extracts: title, projects, people.
    Deterministic fields (source, created_at) are left at defaults.

    Args:
        content: Full document text.
        model: PydanticAI model string (e.g., "anthropic:claude-haiku-4-5").
            If None, returns model with defaults only (no LLM extraction).
        max_content_chars: Max characters of content to send to LLM.

    Returns:
        DocumentMetadataFields with LLM-extracted fields filled where possible.
        On failure after all retries, returns model with all defaults.
    """
    if model is None:
        return DocumentMetadataFields()

    return await _extract_with_fallback(
        model=model,
        input_text=content[:max_content_chars],
        build_agent=_build_doc_agent,
        system_prompt=_DOC_SYSTEM_PROMPT,
        llm_output_type=DocumentLLMFields,
        project=lambda f: DocumentMetadataFields(
            title=f.title,
            projects=f.projects,
            people=f.people,
        ),
        default_factory=DocumentMetadataFields,
        label="Document",
    )


async def extract_chunk_metadata(
    chunk_text: str,
    model: str | None = None,
) -> ChunkMetadataFields:
    """Extract LLM fields for a chunk using PydanticAI structured output.

    Extracts: topics (2-3 tags), people, has_decision (bool), has_action_item (bool).
    Boolean questions are more reliable for local models than multi-way classification.
    Deterministic fields (parent_doc_id, section_path, chunk_index) are left at defaults.

    Args:
        chunk_text: The chunk text to extract metadata from.
        model: PydanticAI model string. If None, returns model with defaults only.

    Returns:
        ChunkMetadataFields with LLM-extracted fields filled where possible.
        On failure after all retries, returns model with all defaults.
    """
    if model is None:
        return ChunkMetadataFields()

    return await _extract_with_fallback(
        model=model,
        input_text=chunk_text,
        build_agent=_build_chunk_agent,
        system_prompt=_CHUNK_SYSTEM_PROMPT,
        llm_output_type=ChunkLLMFields,
        project=lambda f: ChunkMetadataFields(
            topics=f.topics,
            people=f.people,
            has_decision=f.has_decision,
            has_action_item=f.has_action_item,
        ),
        default_factory=ChunkMetadataFields,
        label="Chunk",
    )
