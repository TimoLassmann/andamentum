"""Adapter: proofread.ProofreadResult → list[whetstone.Finding].

Every per-flag finding from ``andamentum.proofread`` is mapped to a
whetstone ``Finding`` with a context-padded anchor quote, so it flows
through the same rendering chain as LLM-emitted findings and appears
as a Word comment in the .docx output.

Anchoring strategy
------------------
Proofread returns char spans into the full markdown text (e.g.
``Span(start=1234, end=1238)`` for a weasel word). Whetstone's ``Quote``
type wants offsets within a *section*, plus the verbatim text. Two
moving parts handled here:

  1. **Section lookup** — find which ``SectionRef`` contains the
     proofread span; rebase the offsets onto the section.
  2. **Anchor uniqueness** — the docx renderer matches by first
     occurrence of ``text_pattern``. Anchoring on a 4-character weasel
     word would land the comment on the first ``"very"`` in the
     document, not the one proofread flagged. The adapter expands each
     span to ~80 chars of word-aligned surrounding context, which is
     long enough to be effectively unique within a section.

Categories
----------
All proofread findings map to whetstone's ``minor`` severity, ``high``
confidence (deterministic), and ``consider`` priority. The category tag
is namespaced under ``style:`` so downstream consumers can filter:

  ``style:weasel``, ``style:passive``, ``style:duplicate_word``,
  ``style:weak_opener``.

Readability scores are intentionally NOT converted into a finding:
they are a document-level summary without a meaningful anchor span,
and the docx renderer skips findings with no quotes. Callers that
want readability numbers should read ``result.metrics`` directly (the
adapter exposes them in a return-tuple alongside the findings list)
or call ``proofread.analyze()`` separately.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..schemas import Finding, Quote

if TYPE_CHECKING:
    from andamentum.proofread import ProofreadResult

    from .types import SectionRef

logger = logging.getLogger("andamentum.whetstone")


# How much surrounding context to include in the anchor quote (in chars).
# 80 is long enough that a word-aligned window is effectively unique
# within a typical section, short enough that the comment lands precisely.
_ANCHOR_CONTEXT_PADDING: int = 80


def _expand_to_word_boundaries(
    text: str, start: int, end: int, padding: int = _ANCHOR_CONTEXT_PADDING
) -> tuple[int, int]:
    """Expand a span by *padding* chars on each side, snapping to whitespace.

    Returns word-aligned (start, end) such that ``text[start:end]`` is a
    self-contained chunk of context — no broken words at the edges.
    """
    new_start = max(0, start - padding)
    new_end = min(len(text), end + padding)
    while new_start > 0 and not text[new_start - 1].isspace():
        new_start -= 1
    while new_end < len(text) and not text[new_end].isspace():
        new_end += 1
    return new_start, new_end


def _locate_in_section(
    sections: "list[SectionRef]", markdown_offset: int
) -> "SectionRef | None":
    """Return the SectionRef whose char range contains *markdown_offset*."""
    for sec in sections:
        if sec.char_start <= markdown_offset < sec.char_end:
            return sec
    return None


def _build_quote(
    *,
    markdown: str,
    sections: "list[SectionRef]",
    flag_start: int,
    flag_end: int,
) -> "Quote | None":
    """Convert a proofread flag's markdown char range into a whetstone Quote.

    Returns None if the flag can't be anchored to any section (e.g. the
    flag lives in markdown outside the chunked sections).
    """
    section = _locate_in_section(sections, flag_start)
    if section is None:
        return None

    # Expand the flag span to surrounding context, aligned to word
    # boundaries — this is the anchor text the docx renderer matches.
    anchor_start, anchor_end = _expand_to_word_boundaries(
        markdown, flag_start, flag_end
    )
    # Clamp to section bounds — the quote MUST live entirely within the section.
    anchor_start = max(anchor_start, section.char_start)
    anchor_end = min(anchor_end, section.char_end)

    in_section_start = anchor_start - section.char_start
    in_section_end = anchor_end - section.char_start
    anchor_text = markdown[anchor_start:anchor_end]

    if not anchor_text.strip():
        return None

    return Quote(
        section_id=section.id,
        char_start=in_section_start,
        char_end=in_section_end,
        text=anchor_text,
    )


def proofread_to_findings(
    *,
    result: "ProofreadResult",
    markdown: str,
    sections: "list[SectionRef]",
) -> list[Finding]:
    """Convert each per-flag proofread finding into a whetstone Finding.

    Readability scores are NOT included — see module docstring.

    Parameters
    ----------
    result:
        The ``ProofreadResult`` from ``andamentum.proofread.analyze()``.
    markdown:
        The full markdown text proofread was run on. Required to rebuild
        word-aligned anchor context from char offsets.
    sections:
        The chunked sections (whetstone's SectionRef list) used to map
        markdown-level offsets into section-local offsets.

    Returns
    -------
    list[Finding]
        One Finding per anchorable per-flag proofread issue. Flags that
        can't be located in any section are silently skipped (rare; only
        happens if proofread saw text the chunker didn't).
    """
    findings: list[Finding] = []

    for w in result.weasel_words:
        quote = _build_quote(
            markdown=markdown,
            sections=sections,
            flag_start=w.span.start,
            flag_end=w.span.end,
        )
        if quote is None:
            continue
        findings.append(
            Finding(
                title=f"Weasel word: '{w.word}'",
                severity="minor",
                confidence="high",
                rationale=(
                    f"'{w.word}' is on the weasel-word list (Matt Might): "
                    f"hedging language that weakens a claim without adding "
                    f"information. Either commit to the claim or remove the "
                    f"qualifier."
                ),
                quotes=[quote],
                sections_involved=[quote.section_id],
                source="deterministic",
                category="style:weasel",
                priority="consider",
            )
        )

    for p in result.passive_voice:
        quote = _build_quote(
            markdown=markdown,
            sections=sections,
            flag_start=p.span.start,
            flag_end=p.span.end,
        )
        if quote is None:
            continue
        findings.append(
            Finding(
                title="Passive voice",
                severity="minor",
                confidence="high",
                rationale=(
                    f"Detected passive-voice construction ('{p.matched_text}'). "
                    f"Active voice usually reads more clearly. The heuristic "
                    f"(be-verb + past participle) is approximate; ignore if "
                    f"passive is genuinely the right choice here."
                ),
                quotes=[quote],
                sections_involved=[quote.section_id],
                source="deterministic",
                category="style:passive",
                priority="consider",
            )
        )

    for d in result.duplicate_words:
        quote = _build_quote(
            markdown=markdown,
            sections=sections,
            flag_start=d.span.start,
            flag_end=d.span.end,
        )
        if quote is None:
            continue
        findings.append(
            Finding(
                title=f"Duplicate word: '{d.word} {d.word}'",
                severity="minor",
                confidence="high",
                rationale=(
                    f"Adjacent repetition of '{d.word}'. Most duplicate-word "
                    f"sequences are typos; common idiomatic doublings "
                    f"('had had', 'that that') are excluded from this check."
                ),
                quotes=[quote],
                sections_involved=[quote.section_id],
                source="deterministic",
                category="style:duplicate_word",
                priority="consider",
            )
        )

    for o in result.weak_openers:
        quote = _build_quote(
            markdown=markdown,
            sections=sections,
            flag_start=o.span.start,
            flag_end=o.span.end,
        )
        if quote is None:
            continue
        findings.append(
            Finding(
                title=f"Weak sentence opener: '{o.matched_text}'",
                severity="minor",
                confidence="high",
                rationale=(
                    f"'{o.matched_text}' is a vacuous opener that delays the "
                    f"sentence's actual content. Consider leading with the "
                    f"subject of the claim directly."
                ),
                quotes=[quote],
                sections_involved=[quote.section_id],
                source="deterministic",
                category="style:weak_opener",
                priority="consider",
            )
        )

    return findings
