"""Tests for stage 1 — markdown header parsing + recursive section split."""

from andamentum.chunker.structural import (
    Section,
    build_section_tree,
    find_headings,
    section_iter_leaves,
    split_section_recursively,
)


def test_find_headings_picks_up_atx_levels():
    src = "# Title\n\n## Section A\n\nbody\n\n### Sub A1\n\nbody\n\n## Section B\n"
    h = find_headings(src)
    assert [x.level for x in h] == [1, 2, 3, 2]
    assert [x.title for x in h] == ["Title", "Section A", "Sub A1", "Section B"]


def test_find_headings_ignores_hashes_in_prose():
    src = "Some text with #notatitle in the middle.\n\nMore text."
    assert find_headings(src) == []


def test_build_section_tree_nests_subsections():
    src = "## A\n\nA-body\n\n### A1\n\nA1-body\n\n### A2\n\nA2-body\n\n## B\n\nB-body\n"
    sections = build_section_tree(src, find_headings(src))
    assert [s.title for s in sections] == ["A", "B"]
    a, b = sections
    assert [c.title for c in a.children] == ["A1", "A2"]
    assert b.children == []
    # A's span includes its children's text
    assert a.end == src.index("## B")
    assert b.end == len(src)


def test_section_iter_leaves_returns_deepest_only():
    src = "## A\n\n### A1\n\nx\n\n### A2\n\ny\n"
    sections = build_section_tree(src, find_headings(src))
    leaves = section_iter_leaves(sections[0])
    assert [l.title for l in leaves] == ["A1", "A2"]


def test_split_section_recursively_keeps_small_sections_intact():
    sec = Section(start=0, end=500, level=2, title="A")
    out = split_section_recursively(sec, target_max=10_000)
    assert out == [sec]


def test_split_section_recursively_descends_into_children_when_too_big():
    src = "## A\n\n" + "x" * 6_000 + "\n\n### A1\n\n" + "y" * 4_000
    sections = build_section_tree(src, find_headings(src))
    parent = sections[0]
    pieces = split_section_recursively(parent, target_max=5_000)
    # Should have an "intro" piece (## A header + 6k of x's) plus the A1 child
    titles = [p.title for p in pieces]
    assert "A1" in titles
    # Total span coverage should equal the parent's span
    total = sum(p.end - p.start for p in pieces)
    assert total == parent.length
