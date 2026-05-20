"""Tests for the tiered provenance watermark.

Two layers:
- Invisible metadata: always on regardless of visible_watermark.
- Visible banner: ON by default for review-report renderers, can be
  suppressed via visible_watermark=False (e.g. for --apply-patches
  output).
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from andamentum.whetstone._watermark import (
    BANNER_TITLE,
    DISCLAIMER_SHORT,
    metadata_markdown_comment,
    stamp_docx_core_properties,
)
from andamentum.whetstone.renderers.html import render_html
from andamentum.whetstone.renderers.markdown import render_markdown
from andamentum.whetstone.schemas import ReviewResult


class TestMarkdownWatermark:
    def test_invisible_metadata_always_present(self) -> None:
        md = render_markdown(
            ReviewResult(summary="x"),
            model="openai:gpt-5.4-nano",
            visible_watermark=False,
        )
        assert "<!-- andamentum-whetstone" in md
        assert "ai-generated: true" in md
        assert "openai:gpt-5.4-nano" in md

    def test_visible_banner_on_by_default(self) -> None:
        md = render_markdown(ReviewResult(summary="x"))
        assert BANNER_TITLE in md
        assert DISCLAIMER_SHORT in md

    def test_visible_banner_off_when_flag_false(self) -> None:
        md = render_markdown(
            ReviewResult(summary="x"), visible_watermark=False
        )
        assert BANNER_TITLE not in md
        # But the invisible metadata still appears.
        assert "<!-- andamentum-whetstone" in md

    def test_clean_document_still_says_so(self) -> None:
        md = render_markdown(ReviewResult())
        assert "looks clean" in md.lower()


class TestHtmlWatermark:
    def test_visible_banner_on_by_default(self) -> None:
        html = render_html(ReviewResult(summary="x"))
        assert BANNER_TITLE in html

    def test_visible_banner_off_when_flag_false(self) -> None:
        html = render_html(ReviewResult(summary="x"), visible_watermark=False)
        assert BANNER_TITLE not in html

    def test_clean_document_still_says_so(self) -> None:
        html = render_html(ReviewResult())
        assert "looks clean" in html.lower()


class TestDocxMetadataStamp:
    def test_stamp_writes_contributor_keyword_and_description(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "test.docx"
        DocxDocument().save(str(path))

        stamp_docx_core_properties(path, model="anthropic:claude-haiku-4-5")

        doc = DocxDocument(str(path))
        assert "andamentum-whetstone" in (doc.core_properties.author or "")
        assert "andamentum:ai-generated" in (doc.core_properties.keywords or "")
        assert "andamentum-whetstone" in (doc.core_properties.comments or "")
        assert "anthropic:claude-haiku-4-5" in (doc.core_properties.author or "")

    def test_stamp_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "test.docx"
        DocxDocument().save(str(path))

        stamp_docx_core_properties(path, model="openai:gpt-5.4-nano")
        first_author = DocxDocument(str(path)).core_properties.author

        stamp_docx_core_properties(path, model="openai:gpt-5.4-nano")
        second_author = DocxDocument(str(path)).core_properties.author

        # Second call must not append a duplicate entry.
        assert first_author == second_author

    def test_stamp_preserves_existing_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "test.docx"
        doc = DocxDocument()
        doc.core_properties.author = "Dr Smith"
        doc.core_properties.keywords = "manuscript, draft"
        doc.save(str(path))

        stamp_docx_core_properties(path, model="ollama:llama3")

        doc = DocxDocument(str(path))
        assert "Dr Smith" in (doc.core_properties.author or "")
        assert "andamentum-whetstone" in (doc.core_properties.author or "")
        assert "manuscript, draft" in (doc.core_properties.keywords or "")
        assert "andamentum:ai-generated" in (doc.core_properties.keywords or "")

    def test_stamp_on_missing_file_does_not_raise(self, tmp_path: Path) -> None:
        # Best-effort: failures must not propagate
        stamp_docx_core_properties(tmp_path / "nonexistent.docx", model=None)


def test_markdown_metadata_comment_format() -> None:
    comment = metadata_markdown_comment(model="ollama:llama3")
    assert comment.startswith("<!--")
    assert comment.endswith("-->")
    assert "ai-generated: true" in comment
    assert "ollama:llama3" in comment
