"""The public chunk-level unit: :class:`ChunkHit` and its stable identifier.

:func:`~.search.search_unified` fuses four signals at the *document* level,
so a :class:`~.public.SearchResult` answers "which documents", never "which
passage". A caller that wants passages — to rerank them, to quote one, or to
record which passage a claim came from — needs a chunk it can name now and
resolve again later. That is what this module adds.

**The identifier is a coordinate, not a content hash.** It is
``"{doc_id}:{chunk_index}"``, derived on read rather than stored, and it
guarantees exactly this much:

- It survives re-embedding. :func:`~.public.reembed_document` rewrites the
  vectors and leaves each chunk's index within its document alone.
- It survives a process restart, a backup, and a copy of the database file,
  because it is built from values that are themselves persisted.
- It does **not** survive re-chunking. Ingest the same document again with
  different chunk boundaries and ``"abc123:3"`` names a different passage.
  A caller that must detect that compares :attr:`ChunkHit.content`; nothing
  in this module can detect it for them.

The chunks table's own ``INTEGER PRIMARY KEY`` is deliberately not the public
id. :func:`~.chunks.delete_chunks_for_document` plus a re-insert gives
unmodified text a new AUTOINCREMENT row id, so the row id is stable across
strictly less than the derived coordinate is — it would be a worse promise
wearing a more official-looking name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_SEPARATOR = ":"


def chunk_id_for(*, doc_id: str, chunk_index: int) -> str:
    """Build the public chunk id for a chunk at ``chunk_index`` of ``doc_id``.

    Both arguments are keyword-only: they are a ``str`` and an ``int`` that
    a caller reads off the same row, and the store's convention is that
    confusable values passed onward go behind a ``*``.
    """
    return f"{doc_id}{_SEPARATOR}{chunk_index}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Split a public chunk id back into ``(doc_id, chunk_index)``.

    Splits on the LAST separator, so a document id that itself contains a
    colon round-trips. Raises :class:`ValueError` on anything that is not a
    chunk id this module produced — a malformed id is a caller bug, and
    returning a plausible-looking guess would push the failure into a query
    that then quietly matches nothing.
    """
    doc_id, separator, index_text = chunk_id.rpartition(_SEPARATOR)
    if not separator or not doc_id:
        raise ValueError(
            f"Not a chunk id: {chunk_id!r}. Expected '<doc_id>:<chunk_index>'."
        )
    try:
        chunk_index = int(index_text)
    except ValueError:
        raise ValueError(
            f"Not a chunk id: {chunk_id!r}. Chunk index {index_text!r} is not an integer."
        ) from None
    return doc_id, chunk_index


@dataclass
class ChunkHit:
    """One chunk of one document, with the id needed to come back to it.

    ``score`` is ``None`` whenever the chunk was not scored in the context
    that produced it — :func:`~.public.get_chunks` enumerates a document and
    ranks nothing, so every hit it returns carries ``None`` rather than a
    ``0.0`` that would read as "scored, and scored badly". When it is set it
    is the producing signal's own score and carries that signal's meaning;
    it is not comparable across queries. A cross-query-comparable number is
    what :func:`~.reranking.rerank` is for.
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    score: float | None = None
    token_count: int | None = None
    metadata: dict = field(default_factory=dict)
