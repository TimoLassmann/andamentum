"""Full-text search using SQLite FTS5."""

from __future__ import annotations
from typing import List, Optional
from pathlib import Path

from .connection import get_connection, DEFAULT_DB_PATH
from .rag.search import SearchResult


def fts_search(
    query: str,
    limit: int = 10,
    db_path: Optional[Path] = None
) -> List[SearchResult]:
    """Full-text search using SQLite FTS5.

    Fast keyword search on document titles and content using Porter stemming.
    Returns results ordered by FTS5 BM25 relevance ranking.

    Args:
        query: Search query (supports FTS5 syntax: "phrase", AND, OR, NOT, *)
        limit: Maximum number of results
        db_path: Path to database file (uses default if None)

    Returns:
        List of SearchResult objects, ordered by relevance

    Examples:
        >>> # Simple keyword search
        >>> results = fts_search("machine learning")

        >>> # Phrase search
        >>> results = fts_search('"neural networks"')

        >>> # Boolean operators
        >>> results = fts_search("python AND (pandas OR numpy)")

        >>> # Prefix wildcard
        >>> results = fts_search("transform*")
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # FTS5 search with BM25 ranking
        cursor.execute("""
            SELECT
                d.doc_uuid,
                d.file_path,
                d.dc_title,
                d.dc_format,
                d.dc_creator,
                d.dc_subject,
                d.markdown_content,
                fts.rank
            FROM documents_fts fts
            JOIN documents d ON fts.rowid = d.id
            WHERE documents_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()

        # Convert to SearchResult objects
        results = []
        for row in rows:
            doc_id, file_path, dc_title, dc_format, dc_creator, dc_subject, markdown_content, rank = row

            # Extract snippet around first match
            snippet = _extract_snippet(markdown_content or "", query, max_length=500)

            # Convert rank (negative value) to positive similarity score (0-1)
            # FTS5 rank is negative, more negative = better match
            # Normalize to 0-1 range (1 = best match)
            similarity_score = min(1.0, abs(rank) / 10.0)  # Heuristic normalization

            results.append(SearchResult(
                content=snippet,
                file_path=file_path,
                doc_id=doc_id,
                similarity_score=similarity_score,
                match_type="keyword",
                start_char=0,
                end_char=len(snippet),
                token_count=None,
                dc_title=dc_title,
                dc_format=dc_format,
                dc_creator=dc_creator,
                dc_subject=dc_subject.split(',') if dc_subject else None,
                metadata={"fts_rank": rank}
            ))

        return results


def _extract_snippet(content: str, query: str, max_length: int = 500) -> str:
    """Extract snippet around first occurrence of query terms.

    Args:
        content: Full document content
        query: Search query
        max_length: Maximum snippet length

    Returns:
        Snippet with query context
    """
    if not content:
        return ""

    # Simple tokenization - split query into terms
    query_terms = [term.strip('"').lower() for term in query.split() if term.strip('"').lower() not in {'and', 'or', 'not'}]

    if not query_terms:
        # No valid query terms, return beginning
        return content[:max_length] + ("..." if len(content) > max_length else "")

    # Find first occurrence of any query term
    content_lower = content.lower()
    first_match_pos = -1

    for term in query_terms:
        pos = content_lower.find(term.rstrip('*'))  # Handle wildcard
        if pos != -1 and (first_match_pos == -1 or pos < first_match_pos):
            first_match_pos = pos

    if first_match_pos == -1:
        # No match found (shouldn't happen with FTS5), return beginning
        return content[:max_length] + ("..." if len(content) > max_length else "")

    # Extract snippet centered on match
    snippet_start = max(0, first_match_pos - max_length // 3)
    snippet_end = min(len(content), snippet_start + max_length)

    snippet = content[snippet_start:snippet_end]

    # Add ellipsis if truncated
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(content):
        snippet = snippet + "..."

    return snippet
