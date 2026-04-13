"""Tests for passage extraction module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ..passage_extraction import (
    LocatedPassage,
    PageData,
    Pointer,
    _cosine_similarity,
    _extract_passage_text,
    _find_pointer_in_chunks,
    _locate_pointers,
    _merge_annotations,
    _normalize_whitespace,
    extract_passages,
)


# ── _normalize_whitespace ─────────────────────────────────────────────────


class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert _normalize_whitespace("hello   world") == "hello world"

    def test_newlines(self):
        assert _normalize_whitespace("hello\nworld") == "hello world"

    def test_tabs(self):
        assert _normalize_whitespace("hello\t\tworld") == "hello world"

    def test_mixed_whitespace(self):
        assert _normalize_whitespace("  hello \n\t world  ") == "hello world"

    def test_empty_string(self):
        assert _normalize_whitespace("") == ""

    def test_only_whitespace(self):
        assert _normalize_whitespace("   \n\t  ") == ""

    def test_no_change_needed(self):
        assert _normalize_whitespace("hello world") == "hello world"


# ── _find_pointer_in_chunks ──────────────────────────────────────────────


class TestFindPointerInChunks:
    def test_exact_match(self):
        chunks = ["The quick brown fox", "jumps over the lazy dog"]
        assert _find_pointer_in_chunks("quick brown fox", chunks) == 0

    def test_case_insensitive(self):
        chunks = ["The Quick Brown Fox", "jumps over the lazy dog"]
        assert _find_pointer_in_chunks("quick brown fox", chunks) == 0

    def test_whitespace_differences(self):
        chunks = ["The  quick\nbrown   fox", "jumps over the lazy dog"]
        assert _find_pointer_in_chunks("quick brown fox", chunks) == 0

    def test_no_match_returns_none(self):
        chunks = ["The quick brown fox", "jumps over the lazy dog"]
        # Completely unrelated text — should fall below fuzzy threshold
        assert _find_pointer_in_chunks("zzzzzzzzzzzzzzz xyz 123 qqqqqq", chunks) is None

    def test_second_chunk(self):
        chunks = ["alpha beta gamma", "delta epsilon zeta"]
        assert _find_pointer_in_chunks("epsilon zeta", chunks) == 1

    def test_fuzzy_partial_match(self):
        chunks = [
            "The method uses randomized controlled trials for evaluation",
            "Completely unrelated content about cooking recipes",
        ]
        # Close but not exact — should fuzzy match to chunk 0
        result = _find_pointer_in_chunks(
            "The method uses randomized controlled trials for assessment", chunks
        )
        assert result == 0

    def test_empty_pointer(self):
        chunks = ["hello world"]
        assert _find_pointer_in_chunks("", chunks) is None


# ── _cosine_similarity ───────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # dot = 32, |a| = sqrt(14), |b| = sqrt(77)
        expected = 32.0 / (14**0.5 * 77**0.5)
        assert abs(_cosine_similarity(a, b) - expected) < 1e-9


# ── _locate_pointers ────────────────────────────────────────────────────


class TestLocatePointers:
    @pytest.mark.asyncio
    async def test_string_match_takes_priority(self):
        """String-matched pointers should not trigger embed_texts."""
        chunks = ["The quick brown fox", "jumps over the lazy dog"]
        pointers = [Pointer(text="quick brown fox", kind="key_excerpt")]

        with patch(
            "andamentum.epistemic.passage_extraction.embed_texts",
            new_callable=AsyncMock,
        ) as mock_embed:
            result = await _locate_pointers(pointers, chunks, chunk_embeddings=[])
            mock_embed.assert_not_called()

        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1].text == "quick brown fox"

    @pytest.mark.asyncio
    async def test_embedding_fallback(self):
        """Pointers that can't string-match should fall back to embeddings."""
        chunks = ["alpha beta gamma"]
        # Pointer that won't string-match
        pointers = [Pointer(text="zzz completely unrelated zzz", kind="key_point")]
        chunk_embs = [[1.0, 0.0, 0.0]]

        mock_embed = AsyncMock(return_value=[[0.9, 0.1, 0.0]])  # similar to chunk 0

        with patch("andamentum.epistemic.passage_extraction.embed_texts", mock_embed):
            result = await _locate_pointers(
                pointers,
                chunks,
                chunk_embeddings=chunk_embs,
                embedding_model="test-model",
            )

        mock_embed.assert_called_once()
        assert len(result) == 1
        assert result[0][0] == 0

    @pytest.mark.asyncio
    async def test_unmatchable_pointer_dropped(self):
        """Pointers below similarity threshold should be dropped."""
        chunks = ["alpha beta gamma"]
        pointers = [Pointer(text="zzz totally different zzz", kind="key_point")]
        chunk_embs = [[1.0, 0.0, 0.0]]

        # Return a very different embedding — near-orthogonal
        mock_embed = AsyncMock(return_value=[[0.0, 0.0, 1.0]])

        with patch("andamentum.epistemic.passage_extraction.embed_texts", mock_embed):
            result = await _locate_pointers(
                pointers,
                chunks,
                chunk_embeddings=chunk_embs,
                similarity_threshold=0.5,
                embedding_model="test-model",
            )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_embedding_when_no_chunk_embeddings(self):
        """If chunk_embeddings is empty, unmatched pointers are simply dropped."""
        chunks = ["alpha beta gamma"]
        pointers = [Pointer(text="zzz no match zzz", kind="key_point")]

        with patch(
            "andamentum.epistemic.passage_extraction.embed_texts",
            new_callable=AsyncMock,
        ) as mock_embed:
            result = await _locate_pointers(pointers, chunks, chunk_embeddings=[])
            mock_embed.assert_not_called()

        assert len(result) == 0


# ── _merge_annotations ──────────────────────────────────────────────────


class TestMergeAnnotations:
    def test_same_chunk_merges(self):
        p1 = Pointer(text="a", kind="key_excerpt")
        p2 = Pointer(text="b", kind="key_point")
        located = [(0, p1), (0, p2)]
        groups = _merge_annotations(located)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_adjacent_chunks_merge(self):
        p1 = Pointer(text="a", kind="key_excerpt")
        p2 = Pointer(text="b", kind="key_point")
        located = [(0, p1), (1, p2)]
        groups = _merge_annotations(located, adjacency=1)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_distant_chunks_separate(self):
        p1 = Pointer(text="a", kind="key_excerpt")
        p2 = Pointer(text="b", kind="key_point")
        located = [(0, p1), (5, p2)]
        groups = _merge_annotations(located, adjacency=1)
        assert len(groups) == 2

    def test_chain_merging(self):
        """Indices 3, 4, 5 should all merge into one group."""
        pointers = [
            (3, Pointer(text="a", kind="key_excerpt")),
            (4, Pointer(text="b", kind="key_point")),
            (5, Pointer(text="c", kind="key_excerpt")),
        ]
        groups = _merge_annotations(pointers, adjacency=1)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_empty_input(self):
        assert _merge_annotations([]) == []

    def test_unsorted_input(self):
        """Input should be sorted internally."""
        p1 = Pointer(text="a", kind="key_excerpt")
        p2 = Pointer(text="b", kind="key_point")
        p3 = Pointer(text="c", kind="key_excerpt")
        located = [(5, p3), (0, p1), (1, p2)]
        groups = _merge_annotations(located, adjacency=1)
        assert len(groups) == 2
        # First group: indices 0, 1
        assert groups[0][0][0] == 0
        assert groups[0][1][0] == 1
        # Second group: index 5
        assert groups[1][0][0] == 5

    def test_custom_adjacency(self):
        p1 = Pointer(text="a", kind="key_excerpt")
        p2 = Pointer(text="b", kind="key_point")
        located = [(0, p1), (3, p2)]
        # adjacency=3 should merge them
        groups = _merge_annotations(located, adjacency=3)
        assert len(groups) == 1


# ── _extract_passage_text ───────────────────────────────────────────────


class TestExtractPassageText:
    def test_single_chunk(self):
        # Create text long enough to span multiple chunks (stride = 1800)
        raw_text = "A" * 10000
        passage = _extract_passage_text(raw_text, [2])
        assert len(passage) > 0
        # Chunk 2 = chars 3600-5600 + up to 300 soft extend each side
        assert len(passage) <= 2000 + 600 + 50  # chunk + max extend + margin

    def test_short_text_returns_whole_text(self):
        raw_text = "Hello world, short text."
        passage = _extract_passage_text(raw_text, [0])
        assert "Hello world" in passage

    def test_soft_sentence_completion(self):
        # Sentence boundary just before and after the chunk
        raw_text = "First sentence. " + "X" * 2000 + " Last part. Trailing text here."
        passage = _extract_passage_text(raw_text, [0])
        # Should extend forward to include "Last part."
        assert "Last part." in passage

    def test_no_sentence_boundary_takes_300_chars(self):
        # No periods anywhere — should extend by 300 chars at the end
        # Use chunk 1 (starts at 1800) so both edges can extend
        raw_text = "A" * 4500
        passage = _extract_passage_text(raw_text, [1])
        # Chunk 1 = chars 1800-3800, extend back 300 to 1500, forward 300 to 4100
        assert len(passage) >= 2000
        assert len(passage) <= 2000 + 600 + 50

    def test_empty_text(self):
        assert _extract_passage_text("", [0]) == ""

    def test_empty_indices(self):
        assert _extract_passage_text("hello", []) == ""

    def test_index_beyond_text(self):
        raw_text = "short"
        passage = _extract_passage_text(raw_text, [100])
        assert passage == raw_text


# ── extract_passages (top-level) ────────────────────────────────────────


class TestExtractPassages:
    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        """Pointers that match should produce passages."""
        content = "The study found significant results. " * 50
        page = PageData(
            url="https://example.com",
            title="Example",
            content=content,
            key_excerpts=["significant results"],
            key_points=[],
        )
        results = await extract_passages([page])
        assert len(results) >= 1
        assert isinstance(results[0], LocatedPassage)
        assert results[0].page_url == "https://example.com"
        assert results[0].page_title == "Example"
        assert results[0].annotation_count >= 1
        assert "key_excerpt" in results[0].annotation_kinds

    @pytest.mark.asyncio
    async def test_empty_pages_returns_empty(self):
        page = PageData(
            url="https://example.com",
            title="Empty",
            content="",
            key_excerpts=["something"],
            key_points=[],
        )
        results = await extract_passages([page])
        assert results == []

    @pytest.mark.asyncio
    async def test_no_pointers_returns_empty(self):
        page = PageData(
            url="https://example.com",
            title="No Pointers",
            content="Some content here.",
            key_excerpts=[],
            key_points=[],
        )
        results = await extract_passages([page])
        assert results == []

    @pytest.mark.asyncio
    async def test_two_pointers_same_region_merge(self):
        """Two pointers pointing at the same text region should merge into one passage."""
        content = "Alpha beta gamma delta. " * 10 + "The key result is important. " * 5
        page = PageData(
            url="https://example.com",
            title="Merge Test",
            content=content,
            key_excerpts=["key result is important"],
            key_points=["key result is important"],
        )
        results = await extract_passages([page])
        # Both pointers hit the same region → should produce one passage
        assert len(results) == 1
        assert results[0].annotation_count == 2

    @pytest.mark.asyncio
    async def test_cross_page_findings_with_embeddings(self):
        """Cross-page findings should be added as pointers when similar to page chunks."""
        content = "Research methodology is important. " * 20
        page = PageData(
            url="https://example.com",
            title="Research",
            content=content,
            key_excerpts=["Research methodology"],
            key_points=[],
        )
        # Finding embedding very similar to chunk embedding
        finding_embs = [[1.0, 0.0, 0.0]]
        chunk_embs_by_url = {"https://example.com": [[0.95, 0.1, 0.0]]}

        # The finding pointer won't substring-match, so _locate_pointers will
        # fall back to embedding.  Mock embed_texts to return a vector close
        # to the single chunk embedding so the finding lands on chunk 0.
        mock_embed = AsyncMock(return_value=[[0.9, 0.1, 0.0]])
        with patch("andamentum.epistemic.passage_extraction.embed_texts", mock_embed):
            results = await extract_passages(
                [page],
                cross_page_findings=["Methodology is crucial"],
                cross_page_finding_embeddings=finding_embs,
                chunk_embeddings_by_url=chunk_embs_by_url,
                embedding_model="test-model",
            )
        assert len(results) >= 1
        # Should have both the excerpt and the finding
        all_kinds = results[0].annotation_kinds
        assert "key_excerpt" in all_kinds
        assert "key_finding" in all_kinds

    @pytest.mark.asyncio
    async def test_cross_page_findings_below_threshold_skipped(self):
        """Cross-page findings below similarity threshold should not be added."""
        content = "Some unrelated content. " * 20
        page = PageData(
            url="https://example.com",
            title="Unrelated",
            content=content,
            key_excerpts=["unrelated content"],
            key_points=[],
        )
        # Finding embedding orthogonal to chunk embedding → low similarity
        finding_embs = [[0.0, 0.0, 1.0]]
        chunk_embs = {"https://example.com": [[1.0, 0.0, 0.0]]}

        results = await extract_passages(
            [page],
            cross_page_findings=["Totally different topic"],
            cross_page_finding_embeddings=finding_embs,
            chunk_embeddings_by_url=chunk_embs,
        )
        # The excerpt still matches, but no finding should be added
        assert len(results) >= 1
        for result in results:
            assert "key_finding" not in result.annotation_kinds

    @pytest.mark.asyncio
    async def test_multiple_pages(self):
        """Each page should be processed independently."""
        page1 = PageData(
            url="https://a.com",
            title="Page A",
            content="Alpha content here. " * 20,
            key_excerpts=["Alpha content"],
            key_points=[],
        )
        page2 = PageData(
            url="https://b.com",
            title="Page B",
            content="Beta content here. " * 20,
            key_excerpts=["Beta content"],
            key_points=[],
        )
        results = await extract_passages([page1, page2])
        urls = {r.page_url for r in results}
        assert "https://a.com" in urls
        assert "https://b.com" in urls
