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

Document extraction: title, doc_type (5 values), projects, people.
Chunk extraction: topics (2-3 tags), people, has_decision (bool), has_action_item (bool).
  Boolean questions are much more reliable for local models than multi-way classification.

Requires: pip install andamentum[llm]  (pydantic-ai)
"""

from __future__ import annotations

import logging

from .metadata_models import (
    ChunkLLMFields,
    ChunkMetadataFields,
    DocumentLLMFields,
    DocumentMetadataFields,
)

logger = logging.getLogger(__name__)

_DOC_SYSTEM_PROMPT = (
    "You are a metadata extraction agent for a personal knowledge base. "
    "Extract structured metadata from the document provided. "
    "Only extract what is explicitly stated. Be specific with project names and people. "
    "The title must be a one-line summary of max 10 words. "
    "doc_type must be one of: reference (papers/articles), plan (grants/proposals), "
    "log (meetings/progress), correspondence (emails/messages), note (thoughts/ideas)."
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
        retries=_RETRIES,
        output_retries=_OUTPUT_RETRIES,
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
        retries=_RETRIES,
        output_retries=_OUTPUT_RETRIES,
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


async def extract_document_metadata(
    content: str,
    model: str | None = None,
    max_content_chars: int = 3000,
) -> DocumentMetadataFields:
    """Extract LLM fields for a document using PydanticAI structured output.

    Extracts: title, doc_type (5 values), projects, people.
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

    try:
        agent = _build_doc_agent(model)
        result = await agent.run(content[:max_content_chars])
        llm_fields = result.output

        return DocumentMetadataFields(
            title=llm_fields.title,
            doc_type=llm_fields.doc_type,
            projects=llm_fields.projects,
            people=llm_fields.people,
        )
    except ImportError:
        raise RuntimeError(
            "pydantic-ai not installed. Install with: pip install andamentum[llm]"
        )
    except Exception as first_error:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        if not isinstance(first_error, UnexpectedModelBehavior):
            logger.warning(f"Document metadata extraction failed: {first_error}")
            return DocumentMetadataFields()

        logger.info(
            "Document metadata: tool-based output failed, retrying with prompted output"
        )

        try:
            from pydantic_ai import Agent, PromptedOutput

            prompted_agent = Agent(
                model,
                system_prompt=_DOC_SYSTEM_PROMPT,
                output_type=PromptedOutput(DocumentLLMFields),
                retries=_RETRIES,
                output_retries=_OUTPUT_RETRIES,
            )
            result = await prompted_agent.run(content[:max_content_chars])
            llm_fields = result.output

            return DocumentMetadataFields(
                title=llm_fields.title,
                doc_type=llm_fields.doc_type,
                projects=llm_fields.projects,
                people=llm_fields.people,
            )
        except Exception as e:
            logger.warning(
                f"Document metadata extraction failed after prompted retry: {e}"
            )
            return DocumentMetadataFields()


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

    try:
        agent = _build_chunk_agent(model)
        result = await agent.run(chunk_text)
        llm_fields = result.output

        return ChunkMetadataFields(
            topics=llm_fields.topics,
            people=llm_fields.people,
            has_decision=llm_fields.has_decision,
            has_action_item=llm_fields.has_action_item,
        )
    except ImportError:
        raise RuntimeError(
            "pydantic-ai not installed. Install with: pip install andamentum[llm]"
        )
    except Exception as first_error:
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        if not isinstance(first_error, UnexpectedModelBehavior):
            logger.warning(f"Chunk metadata extraction failed: {first_error}")
            return ChunkMetadataFields()

        logger.info(
            "Chunk metadata: tool-based output failed, retrying with prompted output"
        )

        try:
            from pydantic_ai import Agent, PromptedOutput

            prompted_agent = Agent(
                model,
                system_prompt=_CHUNK_SYSTEM_PROMPT,
                output_type=PromptedOutput(ChunkLLMFields),
                retries=_RETRIES,
                output_retries=_OUTPUT_RETRIES,
            )
            result = await prompted_agent.run(chunk_text)
            llm_fields = result.output

            return ChunkMetadataFields(
                topics=llm_fields.topics,
                people=llm_fields.people,
                has_decision=llm_fields.has_decision,
                has_action_item=llm_fields.has_action_item,
            )
        except Exception as e:
            logger.warning(
                f"Chunk metadata extraction failed after prompted retry: {e}"
            )
            return ChunkMetadataFields()
