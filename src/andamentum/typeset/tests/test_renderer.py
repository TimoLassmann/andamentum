"""Tests for andamentum.typeset.renderer."""

from __future__ import annotations

from andamentum.typeset import render, render_to_file


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


class TestCallout:
    def test_callout_renders_content(self) -> None:
        html = render([{"kind": "callout", "content": "Important finding."}])
        assert "Important finding" in html
        assert "typeset-callout" in html

    def test_callout_with_tone(self) -> None:
        html = render([{"kind": "callout", "content": "Watch out!", "tone": "warning"}])
        assert "tone-warning" in html

    def test_callout_default_no_tone_on_element(self) -> None:
        html = render([{"kind": "callout", "content": "Note."}])
        assert "typeset-callout" in html
        # The callout element itself should not have a tone class.
        # (CSS may contain .tone-* rules but the element shouldn't.)
        assert 'class="typeset-callout"' in html


class TestItems:
    def test_pairs_variant_renders(self) -> None:
        doc = [{"kind": "items", "entries": [
            {"label": "Question", "body": "Answer here."},
        ]}]
        html = render(doc)
        assert "Question" in html
        assert "Answer here" in html
        assert "typeset-items" in html

    def test_right_variant(self) -> None:
        doc = [{"kind": "items", "variant": "right", "entries": [
            {"label": "2024", "body": "PhD, Bioinformatics."},
        ]}]
        html = render(doc)
        assert "variant-right" in html
        assert "2024" in html

    def test_left_variant(self) -> None:
        doc = [{"kind": "items", "variant": "left", "entries": [
            {"label": "2024", "body": "Keynote at ISMB."},
        ]}]
        html = render(doc)
        assert "variant-left" in html

    def test_multiple_entries(self) -> None:
        doc = [{"kind": "items", "entries": [
            {"label": "A", "body": "First"},
            {"label": "B", "body": "Second"},
        ]}]
        html = render(doc)
        assert "First" in html
        assert "Second" in html

    def test_items_with_heading(self) -> None:
        doc = [{"kind": "items", "heading": "Key Findings", "entries": [
            {"label": "Q", "body": "A"},
        ]}]
        html = render(doc)
        assert "Key Findings" in html

    def test_body_rendered_as_markdown(self) -> None:
        doc = [{"kind": "items", "entries": [
            {"label": "Note", "body": "Has **bold** text."},
        ]}]
        html = render(doc)
        assert "<strong>bold</strong>" in html


class TestAside:
    def test_aside_with_markdown_content(self) -> None:
        html = render([{"kind": "aside", "content": "Small print here."}])
        assert "Small print" in html
        assert "typeset-aside" in html

    def test_aside_with_groups_dict(self) -> None:
        html = render([{"kind": "aside", "groups": {
            "Stats": {"Evidence": 37, "Claims": 2},
            "Meta": {"Model": "gemma4:26b"},
        }}])
        assert "Evidence" in html
        assert "37" in html
        assert "gemma4:26b" in html
        assert "typeset-sidebar" in html


class TestCard:
    def test_card_renders_content(self) -> None:
        html = render([{"kind": "card", "content": "Metformin reduces CV mortality."}])
        assert "Metformin" in html
        assert "typeset-card" in html

    def test_card_with_badge(self) -> None:
        html = render([{"kind": "card", "content": "Claim.", "badge": "supported"}])
        assert "supported" in html
        assert "typeset-badge" in html

    def test_card_with_details(self) -> None:
        html = render([{"kind": "card", "content": "Claim.", "details": "Scope: T2D patients."}])
        assert "<details" in html
        assert "Scope" in html

    def test_card_with_source(self) -> None:
        html = render([{"kind": "card", "content": "X.", "source": "https://example.com"}])
        assert "https://example.com" in html

    def test_card_with_refs(self) -> None:
        html = render([{"kind": "card", "content": "X.", "refs": ["e1", "e2"]}])
        assert "e1" in html


class TestReference:
    def test_reference_renders_content(self) -> None:
        html = render([{"kind": "reference", "content": "A study found..."}])
        assert "A study" in html
        assert "typeset-reference" in html

    def test_reference_with_source_link(self) -> None:
        html = render([{"kind": "reference", "content": "X.", "source": "https://example.com"}])
        assert "href" in html
        assert "https://example.com" in html

    def test_reference_with_badge(self) -> None:
        html = render([{"kind": "reference", "content": "X.", "badge": "supports"}])
        assert "supports" in html

    def test_reference_with_number(self) -> None:
        html = render([{"kind": "reference", "content": "X.", "number": 3}])
        assert "3" in html

    def test_reference_grouping(self) -> None:
        doc = [
            {"kind": "reference", "content": "Paper A.", "group": "2024", "number": 1},
            {"kind": "reference", "content": "Paper B.", "group": "2024", "number": 2},
            {"kind": "reference", "content": "Paper C.", "group": "2023", "number": 3},
        ]
        html = render(doc)
        assert "2024" in html
        assert "2023" in html
        assert "Paper A" in html
        assert "Paper C" in html
        assert "typeset-reference-group" in html

    def test_reference_source_label_overrides_url(self) -> None:
        html = render([{
            "kind": "reference",
            "content": "X.",
            "source": "https://example.com/very/long/path",
            "source_label": "example.com",
        }])
        assert 'href="https://example.com/very/long/path"' in html
        assert ">example.com<" in html

    def test_reference_source_label_defaults_to_url(self) -> None:
        html = render([{
            "kind": "reference",
            "content": "X.",
            "source": "https://example.com",
        }])
        assert ">https://example.com<" in html


class TestAnchorIds:
    def test_heading_id(self) -> None:
        html = render([{"kind": "heading", "content": "T", "id": "top"}])
        assert 'id="top"' in html

    def test_prose_id_on_h2_when_heading_set(self) -> None:
        html = render([{
            "kind": "prose",
            "content": "Body.",
            "heading": "Sources",
            "id": "sources",
        }])
        assert '<h2 id="sources">' in html

    def test_prose_id_on_section_when_no_heading(self) -> None:
        html = render([{"kind": "prose", "content": "Body.", "id": "intro"}])
        assert 'class="typeset-prose" id="intro"' in html

    def test_card_id(self) -> None:
        html = render([{"kind": "card", "content": "Claim.", "id": "claim-1"}])
        assert '<div class="typeset-card" id="claim-1">' in html

    def test_card_source_label(self) -> None:
        html = render([{
            "kind": "card",
            "content": "Claim.",
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12028114/",
            "source_label": "PMC12028114",
        }])
        assert 'href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12028114/"' in html
        assert ">PMC12028114<" in html


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


class TestEndToEnd:
    def test_full_document_renders_all_atoms(self) -> None:
        doc = [
            {"kind": "heading", "content": "Test Report", "subtitle": "A subtitle", "meta": {"date": "2026-04-16"}},
            {"kind": "callout", "content": "Key finding here.", "tone": "note"},
            {"kind": "items", "heading": "Key Facts", "entries": [
                {"label": "Q1", "body": "Answer 1."},
                {"label": "Q2", "body": "Answer 2."},
            ]},
            {"kind": "prose", "content": "## Summary\n\nThe evidence shows..."},
            {"kind": "card", "content": "Claim statement.", "badge": "supported", "refs": ["e1"]},
            {"kind": "reference", "content": "Source description.", "number": 1, "source": "https://example.com", "badge": "supports"},
            {"kind": "aside", "groups": {"Stats": {"Items": 42}}},
        ]
        html = render(doc, style="article")

        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "Key finding" in html
        assert "Q1" in html
        assert "Claim statement" in html
        assert "https://example.com" in html
        assert "42" in html

    def test_renders_with_each_style(self) -> None:
        doc: list[dict[str, object]] = [{"kind": "prose", "content": "Hello."}]
        for style in ["article", "cv", "report"]:
            html = render(doc, style=style)
            assert "Hello" in html

    def test_custom_css_overrides_style(self) -> None:
        doc: list[dict[str, object]] = [{"kind": "prose", "content": "X."}]
        html = render(doc, custom_css="body { color: red; }")
        assert "color: red" in html

    def test_plain_markdown_string_input(self) -> None:
        html = render("# Title\n\nParagraph.")
        assert "Title" in html
        assert "Paragraph" in html

    def test_render_to_file(self, tmp_path: object) -> None:
        from pathlib import Path
        assert isinstance(tmp_path, Path)
        doc: list[dict[str, object]] = [{"kind": "prose", "content": "File test."}]
        path = render_to_file(doc, tmp_path / "out.html")
        assert path.exists()
        content = path.read_text()
        assert "File test" in content
