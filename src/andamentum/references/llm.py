"""The two model calls, each small enough for a local model.

Both are used ONLY when the evidence is mixed (see :mod:`.decide`). Each prompt
carries a bounded sample — never the whole document — and each answer is one
flat field: a style name, or a list of ids. The model never says which entry a
citation points to; code does that, and can be checked.

*model* is always an explicit argument; there is no default.
"""

from __future__ import annotations

import re
from typing import Literal, cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

# Bounds on what one call sees. A local model with a small context window
# handles these comfortably: ~1.5k tokens for the style call, ~1k per batch.
CONTEXT_CHARS = 60  # text on each side of a candidate
SAMPLES_PER_KIND = 6
ENTRIES_SHOWN = 3
ENTRY_CHARS = 150
BATCH_SIZE = 12


class StyleAnswer(BaseModel):
    style: Literal[
        "numeric_brackets",
        "numeric_superscript",
        "numeric_parentheses",
        "author_year",
        "undecided",
    ] = Field(description="How this document cites its references.")
    reason: str = Field(default="", description="One short sentence.")


class CitationIds(BaseModel):
    citation_ids: list[int] = Field(
        default_factory=list,
        description="The ids of the marked items that ARE citations of the reference list.",
    )


_STYLE_PROMPT = """You decide how a document cites its reference list.

You see: how many candidates of each kind were found, a few examples of each
with the text around them (the candidate is between ⟦ and ⟧), and the first
entries of the reference list.

Answer with ONE style:
- numeric_brackets: citations are numbers in square brackets, like [3] or [4, 5].
- numeric_superscript: citations are raised numbers after a word, like disease³.
- numeric_parentheses: citations are numbers in parentheses, like (3).
- author_year: citations name an author and a year, like (Smith et al., 2020).
- undecided: none of these clearly fits.

Numbers can also be footnote marks, exponents, units, affiliations, list items
or equation numbers. Judge by what the examples look like and by whether the
reference list is numbered."""

_CANDIDATE_PROMPT = """You decide which marked items in a document are citations
of its reference list. Each item is shown with the text around it; the item is
between ⟦ and ⟧.

The document cites its references in this style: {style}.
{list_line}

A marked item is a citation when it points to an entry in the reference list.
It is NOT a citation when it is an exponent or unit (m², 10⁵), a footnote mark,
an author affiliation mark, a chemical or gene name, a label with a number
(exon2, day3, chr11, trial1), a list or equation number, or part of a date or
quantity. Citations usually close a clause, right before punctuation.

Return the ids of the items that ARE citations. Return an empty list if none
are."""

STYLE_WORDS = {
    "numeric_brackets": "numbers in square brackets, like [3]",
    "numeric_superscript": "raised (superscript) numbers after a word",
    "numeric_parentheses": "numbers in parentheses, like (3)",
    "author_year": "an author and a year, like (Smith et al., 2020)",
}


def in_context(text: str, start: int, end: int) -> str:
    """The candidate with the text around it, on one line.

    Whitespace is collapsed but NOT stripped at the edges: whether a space
    separates the mark from the next word ("exon⟦2⟧ were" against
    "work⟦2⟧.") is part of the evidence, and an earlier version that dropped
    it showed every label glued to the following word.
    """
    before = text[max(0, start - CONTEXT_CHARS) : start]
    after = text[end : end + CONTEXT_CHARS]

    def flat(chunk: str) -> str:
        return re.sub(r"\s+", " ", chunk)

    return f"…{flat(before)}⟦{text[start:end]}⟧{flat(after)}…"


async def ask_style(prompt: str, *, model: str) -> StyleAnswer:
    agent = build_pydantic_ai_agent(
        AgentDefinition(name="references_style", prompt=_STYLE_PROMPT, output_model=StyleAnswer, retries=2, output_retries=3),
        resolve_model(model),
    )
    result = await agent.run(prompt)
    return cast(StyleAnswer, result.output)


async def ask_citations(
    items: str, *, style: str, model: str, list_line: str = ""
) -> CitationIds:
    agent = build_pydantic_ai_agent(
        AgentDefinition(
            name="references_candidates",
            prompt=_CANDIDATE_PROMPT.format(
                style=STYLE_WORDS.get(style, style), list_line=list_line
            ),
            output_model=CitationIds,
            retries=2,
            output_retries=3,
        ),
        resolve_model(model),
    )
    result = await agent.run(items)
    return cast(CitationIds, result.output)
