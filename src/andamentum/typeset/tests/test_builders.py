"""Tests for builder functions and Report class."""

from __future__ import annotations

from andamentum.typeset import (
    Report,
    aside,
    callout,
    card,
    heading,
    items,
    prose,
    reference,
    render,
)


class TestBuilderFunctions:
    def test_heading_returns_dict(self) -> None:
        result = heading("Title")
        assert result == {"kind": "heading", "content": "Title"}

    def test_heading_with_kwargs(self) -> None:
        result = heading("Title", subtitle="Sub", meta={"date": "2026"})
        assert result["kind"] == "heading"
        assert result["subtitle"] == "Sub"
        assert result["meta"] == {"date": "2026"}

    def test_prose_returns_dict(self) -> None:
        result = prose("Body text.")
        assert result == {"kind": "prose", "content": "Body text."}

    def test_callout_with_tone(self) -> None:
        result = callout("Warning!", tone="warning")
        assert result["kind"] == "callout"
        assert result["tone"] == "warning"

    def test_items_with_entries(self) -> None:
        result = items(entries=[{"label": "Q", "body": "A"}], variant="right")
        assert result["kind"] == "items"
        assert result["variant"] == "right"
        assert len(result["entries"]) == 1

    def test_aside_with_groups(self) -> None:
        result = aside(groups={"Stats": {"Count": "42"}})
        assert result["kind"] == "aside"
        assert "Stats" in result["groups"]

    def test_card_with_badge(self) -> None:
        result = card("Claim.", badge="supported", refs=["1", "2"])
        assert result["kind"] == "card"
        assert result["badge"] == "supported"
        assert result["refs"] == ["1", "2"]

    def test_reference_with_number(self) -> None:
        result = reference("Source.", number=1, source="https://example.com")
        assert result["kind"] == "reference"
        assert result["number"] == 1

    def test_builder_output_renders(self) -> None:
        html = render([heading("Test"), prose("Body.")])
        assert "Test" in html
        assert "Body" in html


class TestReport:
    def test_empty_report(self) -> None:
        r = Report()
        assert len(r) == 0
        assert repr(r) == "Report(0 atoms, style='article')"

    def test_append_atoms(self) -> None:
        r = Report()
        r.heading("Title")
        r.prose("Body.")
        r.callout("Note.")
        assert len(r) == 3

    def test_method_chaining(self) -> None:
        r = Report()
        result = r.heading("Title").prose("Body.").callout("Note.")
        assert result is r
        assert len(r) == 3

    def test_all_seven_atom_methods(self) -> None:
        r = Report()
        r.heading("H")
        r.prose("P")
        r.callout("C")
        r.items(entries=[{"label": "L", "body": "B"}])
        r.aside(content="A")
        r.card("Card")
        r.reference("Ref")
        assert len(r) == 7
        kinds = [a["kind"] for a in r.atoms]
        assert kinds == [
            "heading",
            "prose",
            "callout",
            "items",
            "aside",
            "card",
            "reference",
        ]

    def test_render_returns_html(self) -> None:
        r = Report()
        r.heading("Test Report")
        r.prose("Content here.")
        html = r.render()
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "Content here" in html

    def test_save_writes_file(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        r = Report()
        r.heading("File Test")
        path = r.save(tmp_path / "out.html")
        assert path.exists()
        assert "File Test" in path.read_text()

    def test_atoms_returns_copy(self) -> None:
        r = Report()
        r.heading("X")
        atoms = r.atoms
        atoms.append({"kind": "prose", "content": "extra"})
        assert len(r) == 1  # original not modified

    def test_style_passed_to_render(self) -> None:
        r = Report(style="cv")
        r.prose("Test.")
        html = r.render()
        assert "Inter" in html  # CV style uses Inter font

    def test_kwargs_forwarded(self) -> None:
        r = Report(style="article", title="Custom Title")
        r.prose("Body.")
        html = r.render()
        assert "<title>Custom Title</title>" in html
