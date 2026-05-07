"""Stage 2 — semantic split for oversized sections.

Used only when stage 1 (structural split) leaves a section larger than
``target_max``. Splits at points where consecutive paragraph embeddings
show the largest cosine drops, picking enough cut points to bring all
resulting pieces under ``target_max``.

Returns a list of (start, end) char-offset spans into the original source,
plus per-cut metadata (cosine drop, percentile rank) so the optional LLM
judge stage can reconsider grey-zone cuts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .embeddings import EmbeddingFn, cosine_similarity

# Paragraph break: blank line OR start of source. Unicode-friendly enough
# for trafilatura/docling output (which uses \n\n between paragraphs).
_PARA_RE = re.compile(r"\n\s*\n+")

# Sentence boundary: sentence-final punctuation followed by whitespace and a
# capital letter or opening paren/bracket/quote. Pragmatic — misses cases like
# "et al." or "Dr. Smith" but is only used as a fallback when a paragraph
# already exceeds the embedder's input budget. The split point lands AFTER the
# whitespace (i.e. immediately before the next sentence's first non-space
# character) so every char of the original paragraph survives in some span.
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


@dataclass
class Paragraph:
    start: int  # absolute offset of first char of paragraph
    end: int  # absolute offset of last char + 1 (exclusive)
    text: str


@dataclass
class CutCandidate:
    """A potential split point between two adjacent paragraphs."""

    after_para_idx: int  # cut goes AFTER paragraphs[idx]
    cut_offset: int  # absolute char offset where the cut should land
    drop: float  # 1 - cosine_similarity (higher = more semantic shift)
    percentile: float  # 0–1, this drop's rank among ALL drops in this section


def _sentence_offsets(text: str) -> list[tuple[int, int]]:
    """Partition `text` into (start, end) sentence offsets.

    Sentences include any trailing whitespace up to the next sentence's first
    character. The returned spans cover `text` exactly: no gaps, no overlap.
    If no sentence boundaries are found, returns a single span over `text`.
    """
    boundaries = [0]
    for m in _SENT_END_RE.finditer(text):
        boundaries.append(m.end())
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _split_oversized_paragraph(paragraph: Paragraph, budget: int) -> list[Paragraph]:
    """Subdivide a paragraph that exceeds the embedder's input budget.

    Strategy: greedy-pack sentences (no overlap — overlap inflates cosine
    similarities and breaks downstream cut selection) into ≤budget sub-spans.
    Pathological single sentences longer than the budget fall through to a
    no-overlap char-window split. Sub-paragraph spans partition the parent
    exactly: the union is byte-identical to ``paragraph.text``.
    """
    if len(paragraph.text) <= budget:
        return [paragraph]

    sentences = _sentence_offsets(paragraph.text)

    packed: list[tuple[int, int]] = []
    cur_s, cur_e = sentences[0]
    for s, e in sentences[1:]:
        if e - cur_s <= budget:
            cur_e = e
        else:
            packed.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    packed.append((cur_s, cur_e))

    final: list[tuple[int, int]] = []
    for s, e in packed:
        if e - s <= budget:
            final.append((s, e))
            continue
        cursor = s
        while cursor < e:
            stop = min(cursor + budget, e)
            final.append((cursor, stop))
            cursor = stop

    return [
        Paragraph(
            start=paragraph.start + s,
            end=paragraph.start + e,
            text=paragraph.text[s:e],
        )
        for s, e in final
    ]


def find_paragraphs(
    text: str,
    base_offset: int = 0,
    *,
    budget: int | None = None,
) -> list[Paragraph]:
    """Split `text` into paragraphs at blank-line boundaries.

    `base_offset` is added to each paragraph's start/end so spans can be
    expressed in absolute source-document offsets.

    If `budget` is given, paragraphs longer than `budget` chars are further
    subdivided at sentence boundaries (with a no-overlap char-window
    fallback for pathological single sentences). The post-condition is
    ``all(len(p.text) <= budget for p in result)`` whenever a budget is
    supplied. Span partitioning is preserved across the subdivision.
    """
    paras: list[Paragraph] = []
    cursor = 0
    for m in _PARA_RE.finditer(text):
        end = m.start()
        if end > cursor:
            chunk = text[cursor:end]
            if chunk.strip():
                paras.append(
                    Paragraph(
                        start=base_offset + cursor,
                        end=base_offset + end,
                        text=chunk,
                    )
                )
        cursor = m.end()
    if cursor < len(text):
        chunk = text[cursor:]
        if chunk.strip():
            paras.append(
                Paragraph(
                    start=base_offset + cursor,
                    end=base_offset + len(text),
                    text=chunk,
                )
            )

    if budget is None:
        return paras

    out: list[Paragraph] = []
    for p in paras:
        out.extend(_split_oversized_paragraph(p, budget))
    return out


async def semantic_split_section(
    *,
    source: str,
    section_start: int,
    section_end: int,
    target_max: int,
    target_min: int,
    embedding_fn: EmbeddingFn,
    embed_input_budget: int,
) -> tuple[list[tuple[int, int]], list[CutCandidate]]:
    """Split source[section_start:section_end] at paragraph boundaries.

    Embeds each paragraph, computes cosine drops between consecutive
    embeddings, picks the largest drops as cut points until every resulting
    piece is ≤ target_max chars (or no more cuts can help).

    `embed_input_budget` is the embedder's per-input char limit. Paragraphs
    longer than this are subdivided at sentence boundaries before embedding —
    Ollama returns 500 on inputs that overflow the model's context, and the
    splitter avoids that path entirely. The subdivision also gives stage 2
    cut-candidates inside what would otherwise be an atomic monster paragraph.

    Returns:
      - list of (start, end) char-offset spans
      - list of all CutCandidates that were considered (sorted by drop, desc)
        — the orchestrator uses these to feed the optional LLM judge stage
    """
    section_text = source[section_start:section_end]
    paras = find_paragraphs(
        section_text, base_offset=section_start, budget=embed_input_budget
    )

    if len(paras) <= 1:
        # No paragraph structure to split on — return the section as-is.
        # The orchestrator may need a more aggressive fallback (sentence
        # split) but for v1 we just emit oversized; the editor lets users
        # fix manually.
        return [(section_start, section_end)], []

    # Embed every paragraph
    embeddings = await embedding_fn([p.text for p in paras])

    # Compute drops between every pair of consecutive paragraphs
    drops: list[float] = []
    for i in range(len(paras) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        drops.append(1.0 - sim)

    # Build candidate cuts (sorted desc by drop)
    sorted_drops = sorted(drops)
    n_drops = len(drops)
    candidates: list[CutCandidate] = []
    for i, d in enumerate(drops):
        # Percentile = fraction of drops that are <= this one (0..1)
        pct = (sorted_drops.index(d) + 1) / n_drops
        candidates.append(
            CutCandidate(
                after_para_idx=i,
                cut_offset=paras[i].end,
                drop=d,
                percentile=pct,
            )
        )

    # Greedy: pick the largest-drop cut that ACTUALLY helps (i.e. one of the
    # current pieces is over target_max), until everything is in budget OR
    # no remaining cut helps.
    chosen: set[int] = set()  # paragraph indices after which we cut
    sorted_candidates = sorted(candidates, key=lambda c: -c.drop)

    def _spans_with_cuts() -> list[tuple[int, int]]:
        """Build current spans given `chosen` cut indices.

        Spans are CONTIGUOUS — the cut at `p.end` becomes both the end of
        the previous span and the start of the next span. Inter-paragraph
        whitespace ends up at the start of the following span (rather than
        being dropped) so a downstream judge-merge can splice spans back
        together by `prev_end == next_start`.
        """
        spans: list[tuple[int, int]] = []
        run_start = section_start
        for i, p in enumerate(paras):
            if i in chosen:
                spans.append((run_start, p.end))
                run_start = p.end
        spans.append((run_start, section_end))
        return spans

    def _any_over_budget(spans: list[tuple[int, int]]) -> bool:
        return any((e - s) > target_max for s, e in spans)

    for cand in sorted_candidates:
        spans = _spans_with_cuts()
        if not _any_over_budget(spans):
            break
        # Would this cut help any over-budget span?
        # A cut at after_para_idx splits a span containing it.
        cut_offset = cand.cut_offset
        helps = False
        for s, e in spans:
            if s < cut_offset < e and (e - s) > target_max:
                helps = True
                break
        if helps:
            chosen.add(cand.after_para_idx)

    spans = _spans_with_cuts()
    # Drop empty spans (shouldn't happen but defensive)
    spans = [(s, e) for s, e in spans if e > s]

    # Drop tiny under-min trailing spans by absorbing into previous if possible
    # (we don't merge across topics — just trim degenerate cuts that produce
    # almost-empty trailing pieces).
    if len(spans) >= 2 and (spans[-1][1] - spans[-1][0]) < target_min // 2:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()

    return spans, sorted_candidates


def grey_zone_cuts(
    candidates: list[CutCandidate],
    *,
    chosen_offsets: set[int],
    low_pct: float = 0.60,
    high_pct: float = 0.90,
) -> list[CutCandidate]:
    """Return cuts whose percentile rank falls in the grey zone.

    These are candidates the LLM judge should reconsider — they're "decent"
    cuts (above 60th pct) but not "obvious" (below 90th pct). Both chosen
    and unchosen grey-zone cuts may be reconsidered.
    """
    return [
        c
        for c in candidates
        if low_pct <= c.percentile <= high_pct and c.cut_offset in chosen_offsets
    ]
