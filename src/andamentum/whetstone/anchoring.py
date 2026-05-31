"""Verbatim-quote anchoring against source section text.

The single discipline that prevents the reflection–investigation loop
from drifting away from the manuscript: every quote — whether emitted by
a lens or by an investigator — is checked against actual section text
before it enters the findings pool. Quotes that don't appear verbatim
in the source are dropped.

Implementation delegates to ``andamentum.chunker.validation.find_anchor``,
which provides tiered matching:

  • exact substring,
  • whitespace-normalised match (handles minor formatting drift), then
  • fuzzy match via rapidfuzz (handles minor punctuation differences).

The returned ``Quote.text`` is always the slice of the *source* at the
matched offsets — never the model's submitted text. So even when fuzzy
matching kicks in, the persisted quote is always exactly what the
manuscript says.
"""

from __future__ import annotations

from andamentum.chunker.validation import find_anchor

from .schemas import Quote


def anchor_quote(quote_text: str, section_text: str, section_id: str) -> Quote | None:
    """Return a verified ``Quote`` if ``quote_text`` is in ``section_text``.

    Returns ``None`` when:
      • ``quote_text`` is empty or whitespace-only;
      • the chunker's tiered matcher can't find the span anywhere in
        ``section_text`` (i.e. the model fabricated the quote).

    On success the returned ``Quote.text`` is the source text at the
    matched offsets, not the input ``quote_text``.
    """
    if not quote_text or not quote_text.strip():
        return None
    match = find_anchor(quote_text, section_text, search_from=0)
    if match is None:
        return None
    return Quote(
        section_id=section_id,
        char_start=match.start,
        char_end=match.end,
        text=section_text[match.start : match.end],
    )
