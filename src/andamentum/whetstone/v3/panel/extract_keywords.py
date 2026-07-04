"""Discipline extraction — the panel's one document-classification call."""

from __future__ import annotations

import logging
from typing import cast

from andamentum.core.agents import build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..model import Section
from .agents import EXTRACT_KEYWORDS_DEFN, KeywordExtractionOutput
from .document_view import build_document_view

logger = logging.getLogger("andamentum.whetstone.v3.panel")


async def extract_keywords(
    source: str, sections: list[Section], *, model: str
) -> list[str]:
    """Identify 3-5 review-relevant academic disciplines for the document."""
    agent = build_pydantic_ai_agent(EXTRACT_KEYWORDS_DEFN, resolve_model(model))
    result = await agent.run(
        "Identify 3-5 relevant academic disciplines for reviewing:\n\n"
        f"{build_document_view(source, sections)}"
    )
    from .._metrics import bump_from_result

    bump_from_result(result)
    disciplines = cast(KeywordExtractionOutput, result.output).disciplines
    logger.info(
        "[panel] extracted %d discipline(s): %s",
        len(disciplines),
        ", ".join(disciplines),
    )
    return disciplines
