"""Tests for the v2 store (``document_store.v2``): sections, legs, fusion, rerank.

Synthetic documents and deterministic fake vectors only; no model is loaded.
The reranker is replaced by a stub that scores by a keyword, which is enough to
check that reranking reorders hits and replaces their score.
"""

from __future__ import annotations

import pytest

from andamentum.document_store import reranking
from andamentum.document_store.v2 import SectionStore, identifiers, rrf, split_sections

DIM = 8

DOC = """Preface without a heading.

# Apollo handbook

The Apollo widget is assembled in Brightwater.

## Release v2.1

Release v2.1 fixed the SaltLog import of NM_000256.3 records.

```
# a comment inside a code fence, not a heading
```

## Maintenance

Jane Doe maintains the widget.
"""


def _vec(text: str) -> list[float]:
    # Two features that make "widget" and "release" texts separable.
    t = text.lower()
    return [float("widget" in t), float("release" in t), 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


async def _embed(texts: list[str]) -> list[list[float]]:
    return [_vec(t) for t in texts]


# ── sections ──────────────────────────────────────────────────────────────


def test_sections_follow_headings_and_keep_line_ranges():
    sections = split_sections(DOC, max_chars=2000)
    assert [(s.heading, s.level) for s in sections] == [
        ("", 0),
        ("Apollo handbook", 1),
        ("Release v2.1", 2),
        ("Maintenance", 2),
    ]
    lines = DOC.split("\n")
    for s in sections:
        assert s.text == "\n".join(lines[s.line_from - 1 : s.line_to])
    release = sections[2]
    assert "# a comment inside a code fence" in release.text  # fence is not a heading


def test_long_section_is_cut_at_paragraphs_and_keeps_its_heading_line():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
    text = f"# Long part\n\n{body}"
    pieces = split_sections(text, max_chars=600)
    assert len(pieces) > 1
    assert pieces[0].text.startswith("# Long part\n\nParagraph 0")
    assert all(p.heading == "Long part" for p in pieces)
    assert all(len(p.text) <= 600 for p in pieces)
    # Pieces tile the section: consecutive, no line lost.
    for a, b in zip(pieces, pieces[1:]):
        assert b.line_from == a.line_to + 2  # one blank line between paragraphs


def test_identifiers_finds_quoted_phrases_and_id_like_tokens():
    q = 'Did "Apollo Rising" ship v2.1 with NM_000256.3 in 1994?'
    assert identifiers(q) == ["Apollo Rising", "v2.1", "NM_000256.3", "1994"]
    assert identifiers("what is the widget for") == []


def test_rrf_sums_reciprocal_ranks_and_counts_a_repeated_ranking_twice():
    once = dict(rrf([[1, 2], [2, 3]]))
    assert once[2] > once[1] > once[3]
    twice = dict(rrf([[1, 2], [3], [3]]))
    assert twice[3] > twice[1]


# ── the store ─────────────────────────────────────────────────────────────


@pytest.fixture()
async def store(tmp_path):
    s = SectionStore(tmp_path / "v2.db", embedding_dim=DIM)
    s.create()
    await s.add_document(path="apollo.md", title="Apollo", text=DOC, embed_fn=_embed)
    await s.add_document(
        path="hill.md",
        title="Quiet Hill",
        text="# Quiet Hill\n\nQuiet Hill has sheep and no widget.",
        embed_fn=_embed,
    )
    return s


async def test_bm25_leg_matches_any_query_word(store):
    hits = store._hits([(sid, 0.0) for sid in store.bm25("who maintains sheep")])
    assert {h.path for h in hits} == {"apollo.md", "hill.md"}


async def test_exact_leg_matches_identifiers_literally(store):
    [sid] = store.exact("which release imported NM_000256.3")
    [hit] = store._hits([(sid, 0.0)])
    assert hit.heading == "Release v2.1"
    assert hit.citation == "apollo.md:7-13"


async def test_vector_leg_returns_nearest_sections(store):
    first = store._hits([(store.nearest(_vec("release notes"), k=1)[0], 0.0)])[0]
    assert first.heading == "Release v2.1"


async def test_round0_fuses_legs_and_options_change_the_ranking(store):
    q = "Who maintains the widget in Brightwater?"
    plain = store.round0(q, query_embedding=_vec(q))
    assert plain and all(h.score > 0 for h in plain)
    lexical = store.round0(q, query_embedding=None)
    weighted = store.round0(
        q, query_embedding=_vec(q), use_exact=False, vector_weight=2
    )
    assert {h.section_id for h in lexical} <= {h.section_id for h in plain}
    assert [h.score for h in weighted] != [h.score for h in plain]


async def test_vector_index_and_embed_fn_must_agree(tmp_path):
    lexical = SectionStore(tmp_path / "lex.db", embedding_dim=None)
    lexical.create()
    with pytest.raises(ValueError, match="exactly when"):
        await lexical.add_document(path="a", title="a", text="a", embed_fn=_embed)
    with pytest.raises(ValueError, match="no vector index"):
        lexical.nearest([0.0] * DIM)


async def test_rerank_reorders_hits_and_replaces_the_score(store, monkeypatch):
    async def fake_rerank(query, texts, *, reranker_model):
        assert reranker_model == "stub-reranker"
        return [1.0 if "sheep" in t else 0.0 for t in texts]

    monkeypatch.setattr(reranking, "rerank", fake_rerank)
    q = "widget"
    hits = store.round0(q, query_embedding=_vec(q))
    reranked = await store.rerank("sheep", hits, reranker_model="stub-reranker")
    assert reranked[0].path == "hill.md"
    assert reranked[0].score == 1.0
    assert {h.section_id for h in reranked} == {h.section_id for h in hits}
