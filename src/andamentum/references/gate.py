"""Decision-model replacement for the open-candidate language-model call.

The rules in :mod:`.decide` settle most citation candidates; the ones they
leave open go to a language model. With a decision model, each open candidate
is put to it as one yes/no question instead, and ``p >= 0.5`` means citation.
"""

from __future__ import annotations

from andamentum.core.decisions import DecisionModel, Question

from . import llm
from .models import Candidate, Judged, ReferenceEntry
from .resolve import numbered

QUESTION_TEXT = (
    "Is the item marked between ⟦ and ⟧ a citation of an entry in the "
    "reference list? Answer yes when it points to a listed source, "
    "including a repeated citation of a source cited earlier. Answer no "
    "for an exponent or unit, a label or name with a number (day3, exon2), "
    "a list or equation number, a count or statistic, an author "
    "affiliation mark, or part of a date or quantity."
)


def _points_to(candidate: Candidate, entries: list[ReferenceEntry]) -> str:
    listed = numbered(entries)
    lines = []
    for n in candidate.numbers:
        if n in listed:
            lines.append(f"{n}. {listed[n][0].text[: llm.ENTRY_CHARS]}")
        else:
            lines.append(f"{n}. (the list has no entry with this number)")
    return "\n".join(lines)


async def judge_open(
    open_: list[tuple[Candidate, str]],
    *,
    text: str,
    entries: list[ReferenceEntry],
    style: str,
    list_line: str,
    decision_model: DecisionModel,
) -> list[Judged]:
    """Ask the decision model about every open candidate; one answer each."""
    state = {
        "citation_style": llm.STYLE_WORDS.get(style, style),
        "reference_list": list_line,
        "candidates": {
            str(c.id): {
                "marked_item_in_context": llm.in_context(text, c.start, c.end),
                "entries_it_would_point_to": _points_to(c, entries),
            }
            for c, _ in open_
        },
    }
    questions = {
        str(c.id): Question(
            type="noul",
            instructions=f"Candidate {c.id}. {QUESTION_TEXT}",
        )
        for c, _ in open_
    }
    answers = await decision_model.ask_all(state, questions)
    judged = []
    for c, _ in open_:
        p = answers[str(c.id)].p_true
        assert p is not None
        judged.append(
            Judged(candidate=c, is_citation=p >= 0.5, reason=f"decision model (p={p:.2f})")
        )
    return judged
