"""Tests for andamentum.typeset.renderer."""

from __future__ import annotations

from andamentum.typeset.renderer import render


class TestProse:
    def test_renders_markdown_to_html(self) -> None:
        html = render([{"kind": "prose", "content": "Hello **world**"}])
        assert "<strong>world</strong>" in html

    def test_renders_heading_inside_prose(self) -> None:
        html = render([{"kind": "prose", "content": "## Section\n\nBody text."}])
        assert "<h2" in html
        assert "Body text" in html

    def test_plain_string_treated_as_prose(self) -> None:
        html = render("# Hello\n\nWorld.")
        assert "Hello" in html
        assert "World" in html

    def test_prose_with_heading_field(self) -> None:
        html = render([{"kind": "prose", "content": "Body.", "heading": "Section Title"}])
        assert "Section Title" in html
        assert "Body" in html


class TestHeading:
    def test_heading_renders_h1(self) -> None:
        html = render([{"kind": "heading", "content": "My Report"}])
        assert "<h1>" in html
        assert "My Report" in html
        assert "typeset-heading" in html

    def test_heading_with_subtitle(self) -> None:
        html = render([{"kind": "heading", "content": "Title", "subtitle": "A subtitle"}])
        assert "typeset-subtitle" in html
        assert "A subtitle" in html

    def test_heading_with_meta_dict(self) -> None:
        html = render(
            [
                {
                    "kind": "heading",
                    "content": "Title",
                    "meta": {"date": "2026-04-16", "model": "gemma4"},
                }
            ]
        )
        assert "typeset-meta" in html
        assert "2026-04-16" in html

    def test_heading_with_meta_string(self) -> None:
        html = render([{"kind": "heading", "content": "Title", "meta": "Some meta text"}])
        assert "Some meta text" in html


class TestRenderShell:
    def test_output_is_complete_html(self) -> None:
        html = render([{"kind": "prose", "content": "X."}])
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "typeset-document" in html

    def test_title_auto_detected_from_heading(self) -> None:
        html = render(
            [
                {"kind": "heading", "content": "Auto Title"},
                {"kind": "prose", "content": "Body."},
            ]
        )
        assert "<title>Auto Title</title>" in html

    def test_title_defaults_to_document(self) -> None:
        html = render([{"kind": "prose", "content": "No heading."}])
        assert "<title>Document</title>" in html

    def test_custom_css_overrides_style(self) -> None:
        html = render(
            [{"kind": "prose", "content": "X."}], custom_css="body { color: red; }"
        )
        assert "color: red" in html
