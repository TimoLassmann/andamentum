"""Passage extraction — locate relevant sections in raw web page text.

Given a page's raw text and a set of "pointers" (quotes, key points,
findings from upstream analysis), this module finds the high-signal
regions where annotations converge — analogous to genome annotation
where multiple evidence tracks highlight the same locus.

Architecture: Layer 1 (framework-agnostic, async for embedding fallback only)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .embeddings import _MAX_EMBED_CHARS, _OVERLAP_CHARS, _chunk_text, embed_texts


# ── Data Types ────────────────────────────────────────────────────────────


@dataclass
class Pointer:
    """A single annotation from upstream analysis."""

    text: str
    kind: str  # "key_excerpt", "key_point", "evidence_item", or "key_finding"
    page_url: str = ""
    match_method: str = ""  # set during location: "string", "embedding", or "" if unlocated
    match_similarity: float = 0.0  # cosine similarity for embedding matches


@dataclass
class PageData:
    """A web page with its content and analysis annotations."""

    url: str
    title: str
    content: str
    key_excerpts: list[str]  # verbatim quotes from the page
    key_points: list[str]  # AI-identified important points
    relevance_score: float = 0.5


@dataclass
class LocatedPassage:
    """A passage extracted from source text with its converging annotations."""

    text: str  # Extracted passage (original source text)
    page_url: str
    page_title: str
    annotations: list[str]  # What the analysis said about this region
    annotation_kinds: list[str]  # Kind of each annotation
    match_methods: list[str]  # How each annotation was located ("string" or "embedding")
    match_similarities: list[float]  # Cosine similarity for each (1.0 for string matches)
    chunk_indices: list[int]  # Which chunks this passage spans
    annotation_count: int = 0  # How many pointers converged here


# ── Internal Helpers ──────────────────────────────────────────────────────

_STRIDE = _MAX_EMBED_CHARS - _OVERLAP_CHARS  # 1800


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def _find_pointer_in_chunks(pointer_text: str, chunks: list[str]) -> int | None:
    """Find which chunk contains the pointer text via string matching.

    Pass 1: exact substring match after whitespace normalisation (case-insensitive).
    Pass 2: fuzzy match via SequenceMatcher, accept if ratio >= 0.3.

    Returns chunk index or None.
    """
    # Strip surrounding quote marks (ASCII and smart quotes) — deep-research
    # wraps verbatim excerpts in quotes that don't appear in the raw page text.
    cleaned = pointer_text.strip().strip("\"'\u201c\u201d\u2018\u2019")
    norm_pointer = _normalize_whitespace(cleaned).lower()
    if not norm_pointer:
        return None

    # Pass 1: exact substring
    for i, chunk in enumerate(chunks):
        if norm_pointer in _normalize_whitespace(chunk).lower():
            return i

    # Pass 2: fuzzy match
    best_idx: int | None = None
    best_ratio = 0.0
    for i, chunk in enumerate(chunks):
        ratio = difflib.SequenceMatcher(None, norm_pointer, _normalize_whitespace(chunk).lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio >= 0.3:
        return best_idx
    return None


from .similarity import cosine_similarity as _cosine_similarity  # Canonical implementation


async def _locate_pointers(
    pointers: list[Pointer],
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    similarity_threshold: float = 0.3,
    embedding_model: str | None = None,
) -> list[tuple[int, Pointer]]:
    """Locate each pointer in the chunks.

    Strategy:
    1. Try string matching first (_find_pointer_in_chunks).
    2. For unmatched pointers, embed them and find chunk with highest
       cosine similarity (must exceed similarity_threshold).

    Returns list of (chunk_index, pointer) pairs.
    """
    located: list[tuple[int, Pointer]] = []
    need_embedding: list[Pointer] = []

    for pointer in pointers:
        idx = _find_pointer_in_chunks(pointer.text, chunks)
        if idx is not None:
            pointer.match_method = "string"
            pointer.match_similarity = 1.0
            located.append((idx, pointer))
        else:
            need_embedding.append(pointer)

    if need_embedding and chunk_embeddings:
        if not embedding_model:
            raise RuntimeError("embedding_model is required for pointer embedding. Pass embedding_model= to _locate_pointers().")
        pointer_texts = [p.text for p in need_embedding]
        pointer_embeddings = await embed_texts(pointer_texts, model=embedding_model)

        for pointer, p_emb in zip(need_embedding, pointer_embeddings):
            best_idx = -1
            best_sim = -1.0
            for i, c_emb in enumerate(chunk_embeddings):
                sim = _cosine_similarity(p_emb, c_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_sim >= similarity_threshold:
                pointer.match_method = "embedding"
                pointer.match_similarity = best_sim
                located.append((best_idx, pointer))

    return located


def _merge_annotations(
    located: list[tuple[int, Pointer]],
    adjacency: int = 0,
) -> list[list[tuple[int, Pointer]]]:
    """Group pointers whose chunk indices are within adjacency (single-linkage).

    Sort by chunk index, then walk through: extend current group if within
    adjacency of the last item, otherwise start a new group.
    """
    if not located:
        return []

    sorted_located = sorted(located, key=lambda x: x[0])
    groups: list[list[tuple[int, Pointer]]] = [[sorted_located[0]]]

    for item in sorted_located[1:]:
        last_idx = groups[-1][-1][0]
        if item[0] - last_idx <= adjacency:
            groups[-1].append(item)
        else:
            groups.append([item])

    return groups


_SENTENCE_ENDS = re.compile(r"[.!?]\s")
_SOFT_EXTEND = 300


def _extract_passage_text(
    raw_text: str,
    chunk_indices: list[int],
    stride: int = _STRIDE,
) -> str:
    """Extract passage text for the given chunks with soft sentence completion.

    No padding — the passage is exactly the chunk(s) where pointers landed.
    At each edge, walks up to 300 characters into the adjacent text looking
    for a sentence boundary (. or ! or ? followed by whitespace).  If found,
    extends to complete the sentence.  If not found, takes the 300 characters
    as-is.  Maximum extension is bounded at 300 chars per side.
    """
    if not raw_text or not chunk_indices:
        return ""

    min_idx = min(chunk_indices)
    max_idx = max(chunk_indices)

    start_char = max(0, min_idx * stride)
    end_char = min((max_idx + 1) * stride + _OVERLAP_CHARS, len(raw_text))

    if start_char >= len(raw_text):
        return raw_text

    # Soft completion at start: walk backward up to 300 chars
    if start_char > 0:
        look_start = max(0, start_char - _SOFT_EXTEND)
        prefix = raw_text[look_start:start_char]
        matches = list(_SENTENCE_ENDS.finditer(prefix))
        if matches:
            start_char = look_start + matches[-1].end()
        else:
            start_char = look_start

    # Soft completion at end: walk forward up to 300 chars
    if end_char < len(raw_text):
        look_end = min(len(raw_text), end_char + _SOFT_EXTEND)
        suffix = raw_text[end_char:look_end]
        match = _SENTENCE_ENDS.search(suffix)
        if match:
            end_char = end_char + match.end()
        else:
            end_char = look_end

    return raw_text[start_char:end_char].strip()


# ── Top-Level API ─────────────────────────────────────────────────────────


async def extract_passages(
    pages: list[PageData],
    cross_page_findings: list[str] | None = None,
    cross_page_finding_embeddings: list[list[float]] | None = None,
    chunk_embeddings_by_url: dict[str, list[list[float]]] | None = None,
    embedding_model: str | None = None,
) -> list[LocatedPassage]:
    """Extract located passages from pages using pointer annotations.

    For each page:
    1. Collect all pointers (key_excerpts + key_points).
    2. If cross_page_findings provided and chunk embeddings available,
       check each finding against each page via embedding similarity
       (threshold 0.4), add matching findings as pointers.
    3. Locate all pointers via _locate_pointers.
    4. Merge overlapping annotations via _merge_annotations.
    5. Extract passage text for each group via _extract_passage_text.

    Returns list of LocatedPassage.
    """
    results: list[LocatedPassage] = []

    for page in pages:
        if not page.content.strip():
            continue

        chunks = _chunk_text(page.content)
        chunk_embs = (chunk_embeddings_by_url or {}).get(page.url, [])

        # Collect pointers from page annotations
        page_pointers: list[Pointer] = []
        for excerpt in page.key_excerpts:
            page_pointers.append(Pointer(text=excerpt, kind="key_excerpt", page_url=page.url))
        for point in page.key_points:
            page_pointers.append(Pointer(text=point, kind="key_point", page_url=page.url))

        # Cross-page findings: check each finding against this page
        if cross_page_findings and cross_page_finding_embeddings and chunk_embs:
            for finding, f_emb in zip(cross_page_findings, cross_page_finding_embeddings):
                best_sim = 0.0
                for c_emb in chunk_embs:
                    sim = _cosine_similarity(f_emb, c_emb)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim >= 0.4:
                    page_pointers.append(Pointer(text=finding, kind="key_finding", page_url=page.url))

        if not page_pointers:
            continue

        located = await _locate_pointers(page_pointers, chunks, chunk_embs, embedding_model=embedding_model)
        if not located:
            continue

        groups = _merge_annotations(located)

        for group in groups:
            chunk_indices = sorted(set(idx for idx, _ in group))
            passage_text = _extract_passage_text(page.content, chunk_indices)
            annotations = [p.text for _, p in group]
            annotation_kinds = [p.kind for _, p in group]
            match_methods = [p.match_method for _, p in group]
            match_similarities = [p.match_similarity for _, p in group]
            results.append(
                LocatedPassage(
                    text=passage_text,
                    page_url=page.url,
                    page_title=page.title,
                    annotations=annotations,
                    annotation_kinds=annotation_kinds,
                    match_methods=match_methods,
                    match_similarities=match_similarities,
                    chunk_indices=chunk_indices,
                    annotation_count=len(group),
                )
            )

    return results
