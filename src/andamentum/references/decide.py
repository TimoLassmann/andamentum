"""What a document's citations are: first by evidence, then — only when the
evidence is mixed — by a model.

Two decisions, in order:

1. **The style** of the whole document (:func:`decide_style`). Each candidate
   kind is scored by *support*: how many of its candidates point at a real
   entry — a number the list carries, an author-and-year the list has. A
   reference-manager field settles it outright. Otherwise one style must lead
   clearly (at least 3 supported candidates and 3 times the runner-up, or be
   the only kind of candidate there is) and agree with the list's shape
   (numbered list ↔ numeric style). When that
   fails, the model decides from a bounded sample; its answer is accepted only
   if that style has candidates at all.

2. **Each candidate** of that style (:func:`judge_candidates`). Rules settle
   what they can — a superscript after a unit is an exponent, a number larger
   than the list is not a citation, a raised number beside a name in the
   author block is an affiliation. What the rules leave open goes to the
   model in small batches. With no model given, those stay "not judged" and
   are counted, never guessed.
"""

from __future__ import annotations

import logging
import re

from . import llm
from .models import Candidate, Judged, ReferenceEntry, Style, StyleDecision
from .resolve import numbered, resolve_work

logger = logging.getLogger("andamentum.references")

_FAMILY: dict[str, Style] = {
    "brackets": "numeric_brackets",
    "superscript": "numeric_superscript",
    "glued": "numeric_superscript",
    "parentheses": "numeric_parentheses",
    "parenthetical": "author_year",
    "narrative": "author_year",
    "note": "notes",
}
_CITING_FORMS = frozenset({"full", "short", "ibid"})
NOTE_CHARS = 300  # of a note's text, shown to the model
_LEAD, _RATIO = 3, 3.0
# A raised number right after one of these is an exponent or a unit.
_UNIT_BEFORE = re.compile(
    r"(?:\d|\b(?:m|cm|mm|km|nm|µm|um|s|ms|g|kg|mg|µg|l|L|ml|mL|mol|K|×10|x10|10|e))\s?$"
)
_NOT_CITATION_BEFORE = re.compile(
    r"(?:\b(?:Fig|Figs|Figure|Figures|Table|Tables|Eq|Eqs|Equation|Equations|Step|Steps|Section|n|N|p|P|df|Chapter|Supplementary)\.?\s*[=:]?\s*)$"
)


def _supported(candidate: Candidate, entries: list[ReferenceEntry]) -> bool:
    if candidate.source == "note":
        # A note that reads as a citation is evidence for the notes style
        # whether or not a bibliography exists: notes-only documents have none.
        return candidate.note_form in _CITING_FORMS
    if candidate.works:
        return any(resolve_work(w, entries).status != "missing" for w in candidate.works)
    listed = numbered(entries)
    return bool(candidate.numbers) and all(n in listed for n in candidate.numbers)


def support(candidates: list[Candidate], entries: list[ReferenceEntry]) -> dict[Style, int]:
    counts: dict[Style, int] = {}
    for c in candidates:
        style = _FAMILY.get(c.source)
        if style is not None and _supported(c, entries):
            counts[style] = counts.get(style, 0) + 1
    return counts


async def decide_style(
    text: str,
    candidates: list[Candidate],
    entries: list[ReferenceEntry],
    *,
    model: str | None,
    author_block_end: int = 0,
) -> StyleDecision:
    fields = [c for c in candidates if c.source == "field"]
    if fields:
        style = _style_of_fields(fields, text)
        return StyleDecision(
            style=style,
            decided_by="evidence",
            reason=f"{len(fields)} citations are reference-manager fields, which say what they cite.",
            support={style: len(fields)},
        )

    # A candidate a rule already rules out — an exponent, a unit, an
    # affiliation mark, a figure number — is not evidence for any style.
    # Counted, the distractors in one real-looking document (affiliations 1
    # and 2, "10⁵", "m²") made superscripts look like a rival to its nine
    # bracket citations.
    usable = [
        c
        for c in candidates
        if _rule(c, text, entries, author_block_end=author_block_end)[0] is not False
    ]
    counts = support(usable, entries)
    list_numbered = sum(1 for e in entries if e.number is not None) > len(entries) / 2
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if ranked:
        best, lead = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        # A numbered list goes with numeric citations, an unnumbered one with
        # author-year. Anything else (a numbered list cited by author and
        # year exists, but rarely) is left to the model.
        agrees = best == "notes" or (best == "author_year") != list_numbered
        # A short document may have one or two citations. When every
        # candidate it has is of the same kind, there is nothing to weigh.
        alone = all(_FAMILY.get(c.source) == best for c in usable)
        clear = (lead >= _LEAD and lead >= _RATIO * runner_up) or alone
        if clear and agrees:
            return StyleDecision(
                style=best,
                decided_by="evidence",
                reason=(
                    (
                        f"{lead} notes read as citations"
                        if best == "notes"
                        else f"{lead} {_describe(best)} point to entries in the "
                        f"{'numbered' if list_numbered else 'unnumbered'} list"
                    )
                    + (f", against {runner_up} of the next kind" if runner_up else "")
                    + "."
                ),
                support=counts,
            )

    if model is None:
        return StyleDecision(
            style="undecided",
            decided_by="none",
            reason=(
                "The text does not settle how this document cites "
                f"(candidates that point to entries: {_counts_line(counts)}). "
                "Pass a model to let it decide."
            ),
            support=counts,
        )
    prompt = _style_prompt(text, candidates, entries, counts, list_numbered)
    try:
        answer = await llm.ask_style(prompt, model=model)
    except Exception as exc:  # the report says so; nothing is guessed
        logger.warning("[references] style model call failed: %s", exc)
        return StyleDecision(
            style="undecided",
            decided_by="none",
            reason=f"The model call to decide the style failed ({type(exc).__name__}).",
            support=counts,
        )
    # The model may only name a style the evidence supports at all: one with
    # candidates that survive the rules and point at a real entry. Merely
    # having candidates is not enough — in an MLA document the only raised
    # numbers are distractors the rules already set aside (affiliations,
    # "10⁵"), and a local model named "superscript" from them.
    if answer.style == "undecided" or counts.get(answer.style, 0) == 0:
        return StyleDecision(
            style="undecided",
            decided_by="model",
            reason=answer.reason or "The model could not decide the style.",
            support=counts,
        )
    return StyleDecision(style=answer.style, decided_by="model", reason=answer.reason, support=counts)


def _describe(style: str) -> str:
    return {
        "numeric_brackets": "numbers in brackets",
        "numeric_superscript": "superscript numbers",
        "numeric_parentheses": "numbers in parentheses",
        "author_year": "author-year citations",
        "notes": "notes",
    }.get(style, style)


def _counts_line(counts: dict[Style, int]) -> str:
    return ", ".join(f"{_describe(k)} {v}" for k, v in counts.items()) or "none"


def _style_of_fields(fields: list[Candidate], text: str) -> Style:
    if any(f.works for f in fields):
        return "author_year"
    shown = "".join(text[f.start : f.end] for f in fields)
    if "[" in shown:
        return "numeric_brackets"
    if "(" in shown:
        return "numeric_parentheses"
    return "numeric_superscript"


def _style_prompt(
    text: str,
    candidates: list[Candidate],
    entries: list[ReferenceEntry],
    counts: dict[Style, int],
    list_numbered: bool,
) -> str:
    lines = [
        f"Reference list: {len(entries)} entries, {'numbered' if list_numbered else 'not numbered'}.",
        "First entries:",
    ]
    for e in entries[: llm.ENTRIES_SHOWN]:
        prefix = f"{e.number}. " if e.number is not None else ""
        lines.append(f"  {prefix}{e.text[: llm.ENTRY_CHARS]}")
    for source in ("brackets", "superscript", "glued", "parentheses", "parenthetical", "narrative", "note"):
        of_kind = [c for c in candidates if c.source == source]
        if not of_kind:
            continue
        lines.append(
            f"\n{source} ({len(of_kind)} found, {counts.get(_FAMILY[source], 0)} "
            f"of this style point to an entry). Examples:"
        )
        step = max(1, len(of_kind) // llm.SAMPLES_PER_KIND)
        for c in of_kind[::step][: llm.SAMPLES_PER_KIND]:
            lines.append(f"  {_shown(c, text)}")
    return "\n".join(lines)


# ── each candidate ──────────────────────────────────────────────────────────


def _rule(
    candidate: Candidate,
    text: str,
    entries: list[ReferenceEntry],
    *,
    author_block_end: int,
) -> tuple[bool | None, str]:
    """(verdict, reason). A verdict of None means the rules cannot tell."""
    source = candidate.source
    if source == "field":
        return True, "rule: a reference-manager field"
    if source == "note":
        form = candidate.note_form
        if form in _CITING_FORMS:
            return True, f"rule: a note that cites ({'Ibid.' if form == 'ibid' else f'{form} form'})"
        if form == "comment":
            return False, "rule: a note of commentary"
        return None, "a note the rules cannot read"
    if candidate.works:
        # In an author-year document, a well-formed author-year citation is a
        # citation whether or not its author is in the list — one that is not
        # is exactly the error the check exists to report. (The scan already
        # refuses capitalised words that are not names: "Table 2020".)
        if any(resolve_work(w, entries).status != "missing" for w in candidate.works):
            return True, "rule: the author is in the list"
        return True, "rule: an author-year citation"
    before = text[max(0, candidate.start - 25) : candidate.start]
    listed = numbered(entries)
    highest = max(listed) if listed else 0
    if highest and any(n > max(highest, 1) * 3 for n in candidate.numbers):
        return False, "rule: far larger than the list"
    if source == "brackets":
        return True, "rule: a number in brackets"
    if _NOT_CITATION_BEFORE.search(before):
        return False, "rule: a figure, table, equation or statistic number"
    if source == "parentheses":
        line_start = text.rfind("\n", 0, candidate.start) + 1
        if not text[line_start : candidate.start].strip():
            return False, "rule: a list item number"
        if all(n in listed for n in candidate.numbers):
            return True, "rule: a number in parentheses that the list carries"
        if highest and all(1 <= n <= highest for n in candidate.numbers):
            return True, "rule: a number in parentheses, in the list's range"
        return None, "a number in parentheses the list does not carry"
    # superscript or glued
    if _UNIT_BEFORE.search(before):
        return False, "rule: an exponent or unit"
    if candidate.start < author_block_end and re.search(r"[A-Z][a-zA-Z'’\-]*\s?$|,\s?$", before):
        return False, "rule: an affiliation mark in the author block"
    if not all(n in listed for n in candidate.numbers):
        # Inside the list's own range, after a word: a citation of an entry
        # the list is missing — the error to report. Outside it, the rules
        # cannot tell a citation from a footnote mark.
        in_range = all(1 <= n <= highest for n in candidate.numbers)
        after_word = re.search(r"(?:[a-z]{2}|[.,;:)\]’”\"'])\s?$", before)
        if in_range and after_word:
            return True, "rule: a number in the list's range, after a word"
        return None, "a number the list does not carry"
    if source == "superscript" and re.search(r"(?:[a-z]{2}|[.,;:)\]’”\"'])\s?$", before):
        return True, "rule: a raised number after a word or punctuation"
    return None, "the rules cannot tell"


def _shown(candidate: Candidate, text: str) -> str:
    """What the model sees of one candidate: a note's own text, or the
    candidate marked in its context."""
    if candidate.source == "note":
        body = " ".join(candidate.note_text.split())[:NOTE_CHARS]
        return f"a note reading: ⟦{body}⟧"
    return llm.in_context(text, candidate.start, candidate.end)


def _list_line(entries: list[ReferenceEntry]) -> str:
    listed = numbered(entries)
    if listed:
        return f"The reference list is numbered {min(listed)} to {max(listed)}."
    return f"The reference list has {len(entries)} entries, not numbered."


async def judge_candidates(
    text: str,
    candidates: list[Candidate],
    entries: list[ReferenceEntry],
    *,
    style: Style,
    model: str | None,
    author_block_end: int = 0,
) -> list[Judged]:
    judged: list[Judged] = []
    open_: list[tuple[Candidate, str]] = []
    # The highest number cited so far, in document order. A list numbered in
    # order of first citation introduces 1, 2, 3 ... one at a time, so a
    # glued or raised digit after a word that is exactly the next new number
    # is a citation by that evidence alone. Measured: gpt-5.4-nano rejected
    # two such digits in every run on a flat (formatting-lost) Nature-style
    # text; this rule reads all of them.
    highest = 0
    listed = numbered(entries)
    for c in candidates:
        if c.source != "field" and _FAMILY.get(c.source) != style:
            judged.append(Judged(candidate=c, is_citation=False, reason=f"rule: not the document's style ({_describe(style)})"))
            continue
        verdict, reason = _rule(c, text, entries, author_block_end=author_block_end)
        if verdict is None and c.numbers and c.source in ("glued", "superscript"):
            before = text[max(0, c.start - 25) : c.start]
            after_word = re.search(r"(?:[a-z]{2}|[.,;:)\]’”\"'])\s?$", before)
            if after_word and min(c.numbers) == highest + 1 and all(n in listed for n in c.numbers):
                verdict, reason = True, "rule: the next number in citation order"
            # A repeated citation ("earlier work2.") is NOT settled by a rule:
            # punctuation cannot tell it from a label closing a clause
            # ("counted at day3."), and telling them apart by a list of label
            # words would be a domain rule. It goes to the model.
        if verdict is None:
            open_.append((c, reason))
        else:
            judged.append(Judged(candidate=c, is_citation=verdict, reason=reason))
            if verdict and c.numbers:
                highest = max(highest, max(c.numbers))

    if open_ and model is None:
        judged += [
            Judged(candidate=c, is_citation=False, reason=f"not judged: {why}; pass a model to decide")
            for c, why in open_
        ]
    elif open_ and model is not None:
        for i in range(0, len(open_), llm.BATCH_SIZE):
            batch = open_[i : i + llm.BATCH_SIZE]
            items = "\n".join(f"{c.id}: {_shown(c, text)}" for c, _ in batch)
            try:
                answer = await llm.ask_citations(
                    items, style=style, model=model, list_line=_list_line(entries)
                )
                yes = set(answer.citation_ids)
                judged += [Judged(candidate=c, is_citation=c.id in yes, reason="model") for c, _ in batch]
            except Exception as exc:
                logger.warning("[references] candidate model call failed: %s", exc)
                judged += [
                    Judged(candidate=c, is_citation=False, reason=f"not judged: the model call failed ({type(exc).__name__})")
                    for c, _ in batch
                ]
    judged.sort(key=lambda j: j.candidate.start)
    return judged
