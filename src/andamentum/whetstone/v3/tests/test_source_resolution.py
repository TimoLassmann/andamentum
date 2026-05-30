"""Source resolution for run_review_v3 / run_panel_v3.

The legacy v3 behaviour was: if ``Path(source).exists()`` was False,
silently treat ``source`` as raw markdown content. That bit users on
2026-05-26 — a typo'd path was reviewed as if it were the manuscript,
the LLM happily classified it ("general — placeholder content"), and
the docx renderer crashed downstream on the *real* harvest. Now we
loud-fail when a string looks like a path attempt but the path
doesn't exist.
"""

from __future__ import annotations

import pytest

from andamentum.whetstone.v3.graph import (
    _harvest_or_treat_as_markdown,
    _looks_like_filesystem_path,
)


# ── _looks_like_filesystem_path heuristic ───────────────────────────────────


@pytest.mark.parametrize(
    "candidate",
    [
        "/tmp/draft.md",
        "/Users/researcher/draft.md",
        "./draft.md",
        "../draft.md",
        "~/draft.md",
        "draft.md",
        "paper.pdf",
        "report.docx",
        "slides.pptx",
        "manuscript.tex",
    ],
)
def test_recognises_path_attempts(candidate: str) -> None:
    assert _looks_like_filesystem_path(candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        "# Title\n\nThis is markdown content with newlines.",
        "Just a plain sentence with no path-like markers.",
        "This is a longer block of text that clearly is not a path. "
        "It rambles on for many words without hitting any extension or "
        "slash that would suggest the caller meant a filesystem reference.",
        "",
    ],
)
def test_does_not_recognise_content_as_path(candidate: str) -> None:
    assert not _looks_like_filesystem_path(candidate)


# ── _harvest_or_treat_as_markdown behaviour ─────────────────────────────────


async def test_existing_file_is_harvested(tmp_path) -> None:
    """An existing path is read via harvest.extract."""
    src = tmp_path / "draft.md"
    src.write_text("# Title\n\nBody text.\n")
    out = await _harvest_or_treat_as_markdown(str(src))
    assert "Title" in out
    assert "Body text" in out


async def test_missing_path_raises_file_not_found() -> None:
    """A path-shaped string that doesn't exist must raise — silently
    treating it as raw markdown is the 2026-05-26 bug."""
    with pytest.raises(FileNotFoundError, match="not found"):
        await _harvest_or_treat_as_markdown("/tmp/this-file-does-not-exist-xyz.md")


async def test_raw_markdown_with_newlines_passes_through() -> None:
    """Multi-line content is not a path attempt — it's a manuscript
    the caller passed directly. Return it verbatim."""
    md = "# Title\n\nSome content here.\n"
    out = await _harvest_or_treat_as_markdown(md)
    assert out == md


async def test_plain_prose_no_path_markers_passes_through() -> None:
    """A bare sentence with no slashes / no extension is treated as
    content, not a missing-file error."""
    md = "This is just a sentence of prose"
    out = await _harvest_or_treat_as_markdown(md)
    assert out == md
