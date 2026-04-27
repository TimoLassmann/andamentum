"""Tests for stage 2 — semantic split via paragraph embeddings."""

from andamentum.chunker.semantic_split import (
    find_paragraphs,
    semantic_split_section,
)


def test_find_paragraphs_splits_on_blank_lines():
    text = "Para 1.\n\nPara 2.\n\n\nPara 3.\n"
    paras = find_paragraphs(text, base_offset=100)
    assert [p.text.strip() for p in paras] == ["Para 1.", "Para 2.", "Para 3."]
    # Offsets are absolute (base_offset added)
    assert paras[0].start == 100
    assert paras[1].start > paras[0].end


def test_find_paragraphs_handles_no_blank_lines():
    text = "Just one paragraph with no breaks."
    paras = find_paragraphs(text)
    assert len(paras) == 1
    assert paras[0].text == text


async def test_semantic_split_returns_one_span_when_under_budget():
    src = "Short section content."
    spans, candidates = await semantic_split_section(
        source=src,
        section_start=0,
        section_end=len(src),
        target_max=10_000,
        target_min=2_000,
        embedding_fn=lambda texts: _fake_embed_unrelated(texts),  # type: ignore[arg-type]
    )
    assert spans == [(0, len(src))]
    assert candidates == []


async def test_semantic_split_cuts_at_largest_drop_when_over_budget():
    """Two paragraphs with very different embeddings should yield a cut between them."""
    # Two distinct topical paragraphs
    p1 = "Topic A. " * 200  # ~1800 chars
    p2 = "Topic B. " * 200  # ~1800 chars
    src = p1 + "\n\n" + p2

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        # First paragraph all-A → vector [1,0]; second all-B → [0,1]
        out = []
        for t in texts:
            if "Topic A" in t:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    spans, candidates = await semantic_split_section(
        source=src,
        section_start=0,
        section_end=len(src),
        target_max=2_500,  # smaller than the 3.6k combined length
        target_min=500,
        embedding_fn=fake_embed,
    )
    assert len(spans) == 2  # cut between A and B
    assert spans[0][0] == 0
    assert spans[-1][1] == len(src)
    # The candidate cut had a high drop (~1.0)
    assert candidates[0].drop > 0.9


async def test_semantic_split_skips_cut_when_no_paragraph_break():
    """No paragraph boundaries → can't split, returns whole section."""
    src = "x" * 5_000  # one big block, no \n\n
    spans, candidates = await semantic_split_section(
        source=src,
        section_start=0,
        section_end=len(src),
        target_max=2_000,
        target_min=500,
        embedding_fn=lambda texts: _fake_embed_unrelated(texts),  # type: ignore[arg-type]
    )
    assert spans == [(0, len(src))]
    assert candidates == []


async def _fake_embed_unrelated(texts: list[str]) -> list[list[float]]:
    """Returns orthogonal-ish vectors so all drops are similar."""
    return [[float(i + 1), 0.0] for i in range(len(texts))]
