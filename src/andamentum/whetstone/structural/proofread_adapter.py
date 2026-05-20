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
  2. **Anchor narrowing — stepwise.** The docx renderer matches by
     first occurrence of ``text_pattern``, and Word's comment range
     spans the entire matched substring. A 4-char anchor like
     ``"very"`` would land on the first occurrence in the section, not
     the flagged one; a 200-char padded anchor lands precisely on the
     right occurrence but visually attaches the comment balloon at the
     end of the range — often a sentence or two past the issue. The
     adapter walks a three-step ladder:

       1. **Enclosing sentence** (preferred). Scan backwards / forwards
          for sentence-terminating punctuation followed by whitespace
          and a capital letter. If the resulting sentence appears
          exactly once in the section's text, use it as the anchor.
       2. **Enclosing paragraph** (fallback). When the same sentence
          repeats in the section (rare — e.g. a list item duplicated),
          step up to the paragraph (split on blank lines). If unique,
          use it.
       3. **Word-aligned padded window** (last resort). Only when both
          the enclosing sentence and the enclosing paragraph are
          duplicated within the section. ~80 chars of word-aligned
          context — wider than ideal but guarantees uniqueness in
          pathological cases (e.g. a paragraph repeated verbatim).

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


# Width of the last-resort padded fallback (chars on each side of the flag).
# Only used when both the enclosing sentence AND enclosing paragraph happen to
# be duplicated within the section — vanishingly rare in real text.
_FALLBACK_PADDING: int = 80


def _is_sentence_break_after(text: str, i: int) -> bool:
    """True when ``text[i]`` ends a sentence.

    Heuristic: sentence-ending punctuation (``. ! ?``) followed by whitespace
    and then a letter (uppercase) or open-quote/paren. Tolerates abbreviations
    like ``Dr.``, ``e.g.,``, ``Fig. 3``: those don't satisfy the
    capital-letter-after-whitespace condition. End-of-text always counts.
    """
    if i >= len(text) or text[i] not in ".!?":
        return False
    if i + 1 >= len(text):
        return True  # End of section
    if not text[i + 1].isspace():
        return False  # Not followed by whitespace (e.g. "3.14")
    j = i + 2
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text):
        return True  # Trailing whitespace then end
    # Real sentence boundaries usually start with a capital letter (or an
    # opening quote/parenthesis that wraps a capital). Lowercase or digit
    # follow-on is an abbreviation, a numbered fragment, or a list item.
    return text[j].isupper() or text[j] in "\"'(["


def _enclosing_sentence(text: str, offset: int) -> tuple[int, int]:
    """Return ``[start, end)`` of the sentence in *text* that contains *offset*.

    Falls back to the whole text when no sentence boundary is detected.
    """
    # Scan backwards for the start of the current sentence.
    start = 0
    for i in range(min(offset, len(text)) - 1, 0, -1):
        if _is_sentence_break_after(text, i):
            start = i + 1
            while start < len(text) and text[start].isspace():
                start += 1
            break

    # Scan forwards for the end of the current sentence.
    end = len(text)
    for i in range(max(offset, 0), len(text)):
        if _is_sentence_break_after(text, i):
            end = i + 1
            break

    return start, end


def _enclosing_paragraph(text: str, offset: int) -> tuple[int, int]:
    """Return ``[start, end)`` of the paragraph (blank-line-delimited) at *offset*."""
    # Backward: previous blank line, or start of text.
    prev = text.rfind("\n\n", 0, max(offset, 0))
    start = 0 if prev == -1 else prev + 2

    # Forward: next blank line, or end of text.
    nxt = text.find("\n\n", offset)
    end = len(text) if nxt == -1 else nxt

    return start, end


def _expand_to_word_boundaries(
    text: str, start: int, end: int, padding: int = _FALLBACK_PADDING
) -> tuple[int, int]:
    """Word-aligned padded window. Last-resort fallback when both sentence
    and paragraph anchors are non-unique in the section."""
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
    sections: "list[SectionRef]",
    flag_start: int,
    flag_end: int,
) -> "Quote | None":
    """Convert a proofread flag's markdown char range into a whetstone Quote.

    Walks the three-step anchor ladder (sentence → paragraph → padded
    fallback) and picks the narrowest variant that's unique within the
    enclosing section. Returns None if the flag can't be located in any
    section (rare — only when proofread saw text the chunker didn't).
    """
    section = _locate_in_section(sections, flag_start)
    if section is None:
        return None

    section_text = section.text
    in_section_flag_start = flag_start - section.char_start
    in_section_flag_end = flag_end - section.char_start

    def _maybe_anchor(start: int, end: int) -> "Quote | None":
        """Build a Quote if the [start, end) slice is non-empty and unique."""
        candidate = section_text[start:end]
        if not candidate.strip():
            return None
        if section_text.count(candidate) != 1:
            return None
        return Quote(
            section_id=section.id,
            char_start=start,
            char_end=end,
            text=candidate,
        )

    # ── Step 1: enclosing sentence (preferred) ────────────────────────────
    sent_start, sent_end = _enclosing_sentence(section_text, in_section_flag_start)
    quote = _maybe_anchor(sent_start, sent_end)
    if quote is not None:
        return quote

    # ── Step 2: enclosing paragraph (fallback) ────────────────────────────
    para_start, para_end = _enclosing_paragraph(section_text, in_section_flag_start)
    quote = _maybe_anchor(para_start, para_end)
    if quote is not None:
        return quote

    # ── Step 3: padded word-aligned window (last resort) ──────────────────
    pad_start, pad_end = _expand_to_word_boundaries(
        section_text, in_section_flag_start, in_section_flag_end
    )
    anchor_text = section_text[pad_start:pad_end]
    if not anchor_text.strip():
        return None
    return Quote(
        section_id=section.id,
        char_start=pad_start,
        char_end=pad_end,
        text=anchor_text,
    )


def proofread_to_findings(
    *,
    result: "ProofreadResult",
    sections: "list[SectionRef]",
) -> list[Finding]:
    """Convert each per-flag proofread finding into a whetstone Finding.

    Readability scores are NOT included — see module docstring.

    Parameters
    ----------
    result:
        The ``ProofreadResult`` from ``andamentum.proofread.analyze()``.
    sections:
        The chunked sections (whetstone's SectionRef list). Provides both
        the section the flag belongs to (via char-range lookup) and the
        section's own text — which is what the anchor ladder operates on.

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
