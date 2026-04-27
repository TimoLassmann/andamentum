"""End-to-end tests for the structural-first extract_units pipeline."""

from andamentum.chunker.extractor import extract_units
from andamentum.chunker.types import ChunkingResult


async def test_extract_units_splits_at_section_headings():
    """Top-level `## ` headings should produce one unit per section."""
    src = (
        "## Introduction\n\n"
        + "Intro paragraph. " * 100
        + "\n\n## Methods\n\n"
        + "Methods paragraph. " * 100
        + "\n\n## Results\n\n"
        + "Results paragraph. " * 100
    )
    result = await extract_units(
        src,
        target_min_chars=500,
        target_max_chars=10_000,
    )
    assert isinstance(result, ChunkingResult)
    titles = [u.title for u in result.units]
    assert titles == ["Introduction", "Methods", "Results"]
    # Each unit's text is byte-identical to a source span
    for u in result.units:
        assert src[u.source_start : u.source_end] == u.text
    # Coverage should be ≈ 100% (no orphan content between sections)
    assert result.coverage > 0.95


async def test_extract_units_keeps_preamble_as_its_own_unit():
    """Text before the first heading should become a unit, not a gap."""
    src = "Title and abstract before any heading.\n\n## Section A\n\nbody"
    result = await extract_units(src, target_min_chars=10, target_max_chars=10_000)
    # Two units: preamble + Section A
    assert len(result.units) == 2
    assert result.units[0].source_start == 0
    assert "Title and abstract" in result.units[0].text


async def test_extract_units_handles_no_headings():
    """A source with no markdown headings still emits one unit (the whole doc)."""
    src = "Plain prose with no headings. " * 50
    result = await extract_units(src, target_min_chars=100, target_max_chars=10_000)
    assert len(result.units) == 1
    assert result.units[0].text == src.rstrip() or result.units[0].text == src


async def test_extract_units_empty_source_returns_empty_result():
    result = await extract_units("")
    assert result.units == []
    assert result.gaps == []
    assert result.total_chars == 0


async def test_extract_units_calls_judge_for_grey_zone_cuts():
    """When a judge is supplied AND a section gets semantic-split, the judge is consulted."""
    # Two big paragraphs with very different topical signals — semantic
    # split should produce a cut between them in the grey zone (only one cut,
    # so percentile = 1.0 → above 0.9 → NOT in grey zone by default).
    # Use a wider grey zone so the test exercises the path.
    p1 = "Discussion of topic A " * 200
    p2 = "Discussion of topic B " * 200
    src = "## Big section\n\n" + p1 + "\n\n" + p2

    async def fake_embed(texts):
        return [[1.0, 0.0] if "topic A" in t else [0.0, 1.0] for t in texts]

    judge_calls = {"n": 0}

    async def fake_judge(*, instructions, user_message, output_type, validators):
        judge_calls["n"] += 1
        # Tell the chunker the cut is a real boundary — keep it.
        from andamentum.chunker.judge import JudgeVerdict

        return JudgeVerdict(decision="keep", reason="distinct topics")

    result = await extract_units(
        src,
        target_min_chars=500,
        target_max_chars=2_000,  # forces semantic split
        embedding_fn=fake_embed,
        judge_executor=fake_judge,
        judge_low_pct=0.0,  # widen so any cut is judged
        judge_high_pct=1.0,
    )
    assert judge_calls["n"] >= 1
    assert result.model_calls == judge_calls["n"]


async def test_extract_units_judge_merge_removes_cut():
    """If the judge says merge, the corresponding cut is removed and units coalesce."""
    p1 = "Filler text " * 200
    p2 = "More filler " * 200
    src = "## Section\n\n" + p1 + "\n\n" + p2

    async def fake_embed(texts):
        return [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i, _ in enumerate(texts)]

    async def merge_judge(*, instructions, user_message, output_type, validators):
        from andamentum.chunker.judge import JudgeVerdict

        return JudgeVerdict(decision="merge", reason="same paragraph really")

    no_judge = await extract_units(
        src,
        target_min_chars=500,
        target_max_chars=2_000,
        embedding_fn=fake_embed,
    )
    with_judge = await extract_units(
        src,
        target_min_chars=500,
        target_max_chars=2_000,
        embedding_fn=fake_embed,
        judge_executor=merge_judge,
        judge_low_pct=0.0,
        judge_high_pct=1.0,
    )
    # Without judge: 2 units (cut applied). With merge-judge: 1 unit.
    assert len(no_judge.units) >= 2
    assert len(with_judge.units) == 1


async def test_extract_units_accepts_legacy_kwargs():
    """Old callers (window_size, lookahead, primary_executor) shouldn't break."""
    src = "## A\n\n" + "x " * 100 + "\n\n## B\n\n" + "y " * 100
    result = await extract_units(
        src,
        primary_executor=None,
        window_size=10_000,
        lookahead=4_000,
        extension_chars=5_000,
        max_iterations=3,
        domain="academic",
    )
    assert len(result.units) == 2
