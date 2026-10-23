"""Tests for the deterministic sectioniser."""

from __future__ import annotations

from andamentum.whetstone.pipeline.sectionize import sectionize


def test_no_headings_yields_one_section_covering_source() -> None:
    src = "Just some prose with no headings at all."
    secs = sectionize(src)
    assert len(secs) == 1
    assert secs[0].text == src
    assert (secs[0].start, secs[0].end) == (0, len(src))


def test_section_text_matches_offsets() -> None:
    src = "# Intro\n\nWe study X.\n\n## Methods\n\nWe did Y in detail here.\n"
    for s in sectionize(src, target_max=20):  # small budget → force splitting
        assert s.text == src[s.start : s.end]
        assert s.text.strip()


def test_size_banding_splits_a_large_section() -> None:
    body = "Sentence number {} with some filler content. ".format
    big = "# Top\n\n" + "".join(body(i) for i in range(50))
    big += "\n\n## Sub\n\n" + "".join(body(i) for i in range(50))
    one = sectionize(big, target_max=100_000)  # huge budget → no split
    many = sectionize(big, target_max=300)  # tiny budget → splits
    assert len(many) > len(one)


def test_preamble_before_first_heading_is_kept() -> None:
    src = "Title block and authors.\n\n# Introduction\n\nBody."
    secs = sectionize(src, target_max=10)
    assert any("Title block" in s.text for s in secs)


# ── a parent heading with no text of its own ──────────────────────────


def _results_with_subsections() -> str:
    filler = "Sentence number {} with some filler content. ".format
    return (
        "# Introduction\n\nWe study X.\n\n"
        "# Results\n\n"
        "## Benchmark tasks\n\n" + "".join(filler(i) for i in range(10)) + "\n\n"
        "## Scores\n\n" + "".join(filler(i) for i in range(10)) + "\n"
    )


def test_a_heading_only_parent_joins_its_first_subsection() -> None:
    """A "# Results" heading straight into "## Benchmark tasks" is not an empty Results
    section: no section holds only the heading, and the subsection after it
    opens with it and names it as a parent."""
    src = _results_with_subsections()
    secs = sectionize(src, target_max=300)

    assert [s.title for s in secs] == ["Introduction", "Benchmark tasks", "Scores"]
    tasks = secs[1]
    assert tasks.parents == ["Results"]
    assert tasks.text.startswith("# Results\n\n## Benchmark tasks")
    assert secs[2].parents == []
    for s in secs:
        assert s.text == src[s.start : s.end]


def test_nested_heading_only_parents_are_all_named() -> None:
    filler = "Sentence number {} with some filler content. ".format
    src = (
        "# Results\n\n## Part one\n\n"
        "### Detail\n\n" + "".join(filler(i) for i in range(10)) + "\n\n"
        "### More\n\n" + "".join(filler(i) for i in range(10)) + "\n"
    )
    secs = sectionize(src, target_max=200)
    assert secs[0].title == "Detail"
    assert secs[0].parents == ["Results", "Part one"]
    assert secs[0].start == 0


def test_an_empty_section_before_a_sibling_stays() -> None:
    """A heading followed by a heading at its own level has no subsections to
    join: that section is empty, and the review should be able to say so."""
    filler = "Sentence number {} with some filler content. ".format
    src = "# Results\n\n# Discussion\n\n" + "".join(filler(i) for i in range(10)) + "\n"
    secs = sectionize(src, target_max=100)
    assert [s.title for s in secs] == ["Results", "Discussion"]
    assert all(s.parents == [] for s in secs)
