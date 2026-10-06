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

import re

#: FTS5's binary operators. They are case-sensitive and need a term on both
#: sides; ``NEAR`` without parentheses is an ordinary word, so it is not here.
_BOOL_OPERATORS = ("AND", "OR", "NOT")
_TOKEN = re.compile(r'"[^"]*"|[^\s"]+')


def prepare_fts_query(query: str) -> str:
    """Return an FTS5-syntax-safe MATCH string for a raw user query.

    The query is split into tokens and only valid FTS5 syntax is kept:

    - Clean prose (all-alphanumeric barewords) is returned unchanged, so keyword
      ranking on ordinary queries is identical to before.
    - A balanced ``"phrase"`` stays a phrase. Unbalanced quote marks are
      dropped, since FTS5 rejects them (``unterminated string``).
    - An uppercase ``AND`` / ``OR`` / ``NOT`` stays an operator when it stands
      between two terms, and is dropped otherwise. FTS5 only honours uppercase
      operators; lowercase ``and`` is an ordinary search term.
    - ``word*`` stays a prefix query.
    - Any other token that is not a plain alphanumeric bareword is wrapped as a
      phrase (``"c.1234G>A"``). This is both syntax-safe and makes punctuated
      identifiers match as atomic ordered token sequences.
    - Pure-punctuation tokens (no alphanumeric character) are dropped, since an
      empty phrase is not useful and can be rejected by FTS5.

    Deliberate FTS5 syntax used to pass through untouched as a whole. That
    failed on ordinary questions: a quoted title in a question ending in ``?``,
    or a hyphenated word next to ``AND`` (FTS5 reads ``co-founder`` as column
    ``co`` minus ``founder``), reached FTS5 raw and was rejected. Each piece of
    deliberate syntax is now kept on its own while the rest is made safe.

    Examples:
        >>> prepare_fts_query("how does mitosis divide")
        'how does mitosis divide'
        >>> prepare_fts_query("c.1234G>A")
        '"c.1234G>A"'
        >>> prepare_fts_query("pathogenic variant c.1234G>A in BRCA1")
        'pathogenic variant "c.1234G>A" in BRCA1'
        >>> prepare_fts_query('"neural networks"')
        '"neural networks"'
        >>> prepare_fts_query('Who starred in "Friends"?')
        'Who starred in "Friends"'
    """
    stripped = query.strip()
    if not stripped:
        return stripped
    text = stripped if stripped.count('"') % 2 == 0 else stripped.replace('"', " ")

    terms: list[str] = []
    for token in _TOKEN.findall(text):
        if token.startswith('"'):
            # A balanced phrase; FTS5 tokenizes inside it, so any character
            # but a quote is safe there.
            if any(ch.isalnum() for ch in token):
                terms.append(token)
        elif token in _BOOL_OPERATORS:
            if terms and terms[-1] not in _BOOL_OPERATORS:
                terms.append(token)
        elif token.isalnum() or (token.endswith("*") and token[:-1].isalnum()):
            terms.append(token)
        elif any(ch.isalnum() for ch in token):
            # Punctuated token (identifier, hyphenated word, DOI, ...). Phrase-quote
            # it so FTS5 treats it as an atomic ordered token sequence.
            terms.append('"' + token + '"')
        # else: pure punctuation; drop it.
    while terms and terms[-1] in _BOOL_OPERATORS:
        terms.pop()

    return " ".join(terms) if terms else stripped
