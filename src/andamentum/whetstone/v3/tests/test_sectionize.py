"""Tests for the deterministic sectioniser."""

from __future__ import annotations

from andamentum.whetstone.v3.sectionize import sectionize


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
