"""Tests for stage 2 — semantic split via paragraph embeddings."""

from andamentum.chunker.semantic_split import (
    _split_oversized_paragraph,
    Paragraph,
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
        embed_input_budget=10_000,
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
        embed_input_budget=10_000,
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
        embed_input_budget=10_000,
    )
    assert spans == [(0, len(src))]
    assert candidates == []


async def _fake_embed_unrelated(texts: list[str]) -> list[list[float]]:
    """Returns orthogonal-ish vectors so all drops are similar."""
    return [[float(i + 1), 0.0] for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Oversized-paragraph subdivision (the regression that prompted this fix:
# Ollama returns 500 when an embed input exceeds the model's context window).
# ---------------------------------------------------------------------------


def test_split_oversized_paragraph_packs_at_sentence_boundaries():
    sentences = ["Sentence number {}. ".format(i) for i in range(50)]
    text = "".join(sentences)  # ~950 chars
    p = Paragraph(start=0, end=len(text), text=text)
    parts = _split_oversized_paragraph(p, budget=200)

    assert len(parts) >= 2
    assert all(len(part.text) <= 200 for part in parts)
    # Spans partition the parent exactly — byte-identical reassembly.
    assert "".join(part.text for part in parts) == text
    # Spans are contiguous and absolute.
    assert parts[0].start == 0
    assert parts[-1].end == len(text)
    for a, b in zip(parts, parts[1:]):
        assert a.end == b.start


def test_split_oversized_paragraph_falls_back_to_char_window():
    """A single sentence longer than the budget — char-window with no overlap."""
    text = "x" * 5_000  # one giant blob, no sentence punctuation
    p = Paragraph(start=100, end=100 + len(text), text=text)
    parts = _split_oversized_paragraph(p, budget=1_000)

    assert len(parts) == 5
    assert all(len(part.text) <= 1_000 for part in parts)
    # Spans partition + are contiguous and absolute (offset=100).
    assert "".join(part.text for part in parts) == text
    assert parts[0].start == 100
    assert parts[-1].end == 100 + len(text)
    for a, b in zip(parts, parts[1:]):
        assert a.end == b.start


def test_split_oversized_paragraph_returns_self_when_under_budget():
    text = "Short."
    p = Paragraph(start=0, end=len(text), text=text)
    assert _split_oversized_paragraph(p, budget=1_000) == [p]


def test_find_paragraphs_with_budget_subdivides_long_blocks():
    """One short paragraph + one over-budget paragraph → only the long one is split."""
    short = "Short paragraph."
    long_text = ". ".join(f"Sentence {i}" for i in range(40)) + "."  # ~430 chars
    text = short + "\n\n" + long_text
    paras = find_paragraphs(text, budget=200)

    assert len(paras) >= 3  # 1 short + at least 2 sub-paras of long
    assert all(len(p.text) <= 200 for p in paras)


async def test_semantic_split_no_500_with_oversized_paragraph():
    """Regression: a paragraph larger than embed_input_budget must NOT be
    sent to the embedder verbatim. Before the fix, Ollama returned 500."""
    one_giant_para = "a" * 5_000  # no blank lines, no sentences
    src = "Intro.\n\n" + one_giant_para

    # Fake embedder that asserts every input fits the advertised budget.
    seen_lengths: list[int] = []

    async def strict_embed(texts: list[str]) -> list[list[float]]:
        for t in texts:
            seen_lengths.append(len(t))
            assert len(t) <= 1_000, f"embed input {len(t)} > budget"
        return [[float(i), 0.0] for i in range(len(texts))]

    spans, _candidates = await semantic_split_section(
        source=src,
        section_start=0,
        section_end=len(src),
        target_max=2_500,
        target_min=500,
        embedding_fn=strict_embed,
        embed_input_budget=1_000,
    )
    # Confirms the embedder was actually called with the subdivided pieces.
    assert seen_lengths
    assert max(seen_lengths) <= 1_000
    # And the section was split.
    assert len(spans) >= 2
