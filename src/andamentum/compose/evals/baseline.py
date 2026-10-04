"""The single-call baseline arm: one model call writes the whole document."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from .._call import call_model
from ..genres import Genre
from ..log import ActionLog
from ..schemas import Fact
from ..write import MARKER_RULES, facts_block

SOURCE_CHARS_PER_SOURCE = 20000

BASELINE_INSTRUCTIONS = (
    """\
You write a complete document in one pass from a task and its sources. Return a \
title and an ordered list of sections, each with a title and a body. Section \
bodies are plain markdown paragraphs, "- " bullets, "1. " numbered lists or pipe \
tables, with no headings inside a body. Follow the genre's section skeleton, \
length and style notes. Use only what the sources support; where a specific is \
needed and the sources do not give it, write [TO FILL: what is needed].

"""
    + MARKER_RULES
)


class BaselineSection(BaseModel):
    title: str
    body: str


class BaselineDocument(BaseModel):
    title: str
    sections: list[BaselineSection]


async def single_call(
    task: str,
    sources_text: dict[str, str],
    genre: Genre,
    *,
    model: str,
    facts: Sequence[Fact] = (),
    log: ActionLog | None = None,
) -> BaselineDocument:
    """Write the whole document with one call (the evaluation baseline).

    *facts* are the same extracted, quote-checked facts the compose arm drafts
    from, listed with the same marker rules, so both arms cite alike and the
    grounding check judges them on equal terms.
    """
    skeleton = "\n".join(
        f"- {s.title} ({s.words[0]}-{s.words[1]} words): {s.purpose}"
        for s in genre.sections
    )
    sources = (
        "\n\n".join(
            f"--- {name}\n{text[:SOURCE_CHARS_PER_SOURCE]}"
            for name, text in sources_text.items()
        )
        or "(no sources)"
    )
    message = (
        f"TASK:\n{task}\n\n"
        f"GENRE: {genre.name}: {genre.description}\n"
        f"Total length: {genre.word_range[0]}-{genre.word_range[1]} words.\n"
        f"Section skeleton:\n{skeleton}\n"
        f"Style: {genre.style_notes}\n\n"
        f"{facts_block(list(facts))}\n\n"
        f"SOURCES:\n{sources}"
    )
    return await call_model(
        model=model,
        instructions=BASELINE_INSTRUCTIONS,
        output_type=BaselineDocument,
        user_message=message,
        node="baseline",
        log=log,
    )
