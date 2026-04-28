"""Keyword-extraction agent for panel mode.

Single LLM call. Reads the document map / markdown and emits 3-5
academic disciplines that would be most relevant for reviewing the
work. Each discipline becomes one fictional expert in the next phase.

The prompt is lifted from v1's ``multi_expert._KEYWORD_PROMPT`` with
the v2 hygiene pass: drop the v1 "AnalysisAgent / NOT EditingAgent"
boilerplate, keep the substantive guidance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

EXTRACT_KEYWORDS_PROMPT = """You are an expert in academic classification and interdisciplinary research.

Your task is to analyze the provided document and identify 3-5 academic disciplines that would be most relevant for reviewing this work.

# Guidelines

1. Breadth and diversity. Select disciplines that cover different aspects of the work:
     • Primary discipline (most directly related)
     • Secondary disciplines (complementary perspectives)
     • Consider interdisciplinary angles

2. Specificity. Be specific rather than generic:
     • Good: "Computational Neuroscience", "Machine Learning",
       "Cognitive Psychology"
     • Too broad: "Science", "Technology", "Research"

3. Academic conventions. Use standard academic discipline names.

4. Relevance ranking. Order from most to least relevant.

5. Realistic scope. Think about what types of experts would actually
   review this:
     • Who would be on a PhD committee for this topic?
     • Which departments would have relevant expertise?

# Output

Return a KeywordExtractionOutput with 3-5 academic disciplines that
would provide the most valuable and diverse perspectives for reviewing
this work."""


class KeywordExtractionOutput(BaseModel):
    """Flat output for the keyword_extractor agent."""

    disciplines: list[str] = Field(
        description=(
            "3-5 academic disciplines, ordered most-to-least relevant. "
            "Use specific discipline names (e.g. 'Computational "
            "Neuroscience'), not generic ones (e.g. 'Science')."
        )
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="extract_keywords",
        prompt=EXTRACT_KEYWORDS_PROMPT,
        output_model=KeywordExtractionOutput,
        retries=2,
        output_retries=2,
    )


EXTRACT_KEYWORDS_AGENT = _build()
