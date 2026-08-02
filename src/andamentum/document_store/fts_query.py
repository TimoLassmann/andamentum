"""FTS5 query preparation — make raw user queries syntax-safe and identifier-aware.

Background
----------
The FTS5 tables are tokenized ``porter unicode61``. That is the right choice
for prose. The problem is not the *tokenizer* — it is the *query string*.

A raw query containing notation such as ``c.1234G>A`` (an HGVS variant),
``NM_000256.3`` (an accession) or ``10.1038/nature12373`` (a DOI) is not valid
FTS5 query syntax: ``.``, ``:``, ``-`` and friends are operators/separators, so
``documents_fts MATCH 'c.1234G>A'`` raises ``fts5: syntax error near "."``. In
the search pipeline that exception is swallowed and the keyword signal silently
drops out, leaving only embeddings — which cannot separate ``c.1234G>A`` from
``c.1234G>T`` because their vectors are near-identical.

The fix is to phrase-quote any token that is not a plain alphanumeric bareword.
Inside an FTS5 phrase the same ``porter unicode61`` tokenizer runs, so
``"c.1234G>A"`` becomes the ordered token sequence ``[c, 1234g, a]`` and matches
*only* documents carrying that exact sequence — ``c.1234G>T`` (``[c, 1234g, t]``)
does not match. Discrimination is recovered on the existing index, with no new
table, no re-tokenization and no reindex.

Deliberate FTS5 power-queries (explicit quotes, boolean operators, wildcards)
are passed through untouched — this only rescues the raw-text case that today
either crashes or under-discriminates.
"""

from __future__ import annotations

_BOOL_OPERATORS = (" AND ", " OR ", " NOT ", " NEAR ")


def prepare_fts_query(query: str) -> str:
    """Return an FTS5-syntax-safe MATCH string for a raw user query.

    - Clean prose (all-alphanumeric barewords) is returned unchanged, so keyword
      ranking on ordinary queries is identical to before.
    - Deliberate FTS5 power-queries (containing ``"``, ``*``, or an **uppercase**
      boolean operator) are returned unchanged — the caller opted into FTS5
      syntax. Uppercase is the test because FTS5 only honours ``AND``/``OR``/
      ``NOT``/``NEAR`` in upper case; lowercase ``and`` is a search term, and
      treating it as an operator would leave ordinary prose unescaped.
    - Otherwise each whitespace token that is not a plain alphanumeric bareword
      is wrapped as a phrase (``"c.1234G>A"``). This is both syntax-safe and
      makes punctuated identifiers match as atomic ordered token sequences.
    - Pure-punctuation tokens (no alphanumeric character) are dropped, since an
      empty phrase is not useful and can be rejected by FTS5.

    Examples:
        >>> prepare_fts_query("how does mitosis divide")
        'how does mitosis divide'
        >>> prepare_fts_query("c.1234G>A")
        '"c.1234G>A"'
        >>> prepare_fts_query("pathogenic variant c.1234G>A in BRCA1")
        'pathogenic variant "c.1234G>A" in BRCA1'
        >>> prepare_fts_query('"neural networks"')  # power-query, untouched
        '"neural networks"'
    """
    stripped = query.strip()
    if not stripped:
        return stripped

    # Respect deliberate FTS5 syntax — don't second-guess a power-user query.
    if '"' in stripped or "*" in stripped:
        return stripped
    # FTS5's boolean keywords are CASE-SENSITIVE — only an uppercase AND/OR/NOT/
    # NEAR is an operator; lowercase "and" is an ordinary search term. Matching
    # case-insensitively here would classify ordinary prose ("...random words and
    # asking the model...") as a deliberate power-query and return it unescaped,
    # so a hyphenated word in that same query would reach FTS5 raw and be parsed
    # as a column filter -> OperationalError("no such column: training").
    padded = f" {stripped} "
    if any(op in padded for op in _BOOL_OPERATORS):
        return stripped

    terms: list[str] = []
    for token in stripped.split():
        if token.isalnum():
            terms.append(token)
        elif any(ch.isalnum() for ch in token):
            # Punctuated token (identifier, hyphenated word, DOI, …). Phrase-quote
            # it so FTS5 treats it as an atomic ordered token sequence.
            terms.append('"' + token + '"')
        # else: pure punctuation — drop it.

    return " ".join(terms) if terms else stripped
