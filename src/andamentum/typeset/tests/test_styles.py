"""Tests for built-in styles."""

import pytest

from andamentum.typeset.styles import STYLES, get_style


class TestStyles:
    def test_three_styles_exist(self):
        assert set(STYLES.keys()) == {"article", "cv", "report"}

    def test_get_style_returns_css_string(self):
        for name in STYLES:
            css = get_style(name)
            assert isinstance(css, str)
            assert len(css) > 500, f"{name} style seems too short"

    def test_unknown_style_raises(self):
        with pytest.raises(KeyError, match="Unknown style"):
            get_style("nonexistent")

    def test_all_atom_classes_in_each_style(self):
        required_classes = [
            "typeset-document", "typeset-heading", "typeset-subtitle",
            "typeset-meta", "typeset-prose", "typeset-callout",
            "typeset-items", "typeset-item", "typeset-item-label",
            "typeset-item-body", "typeset-aside", "typeset-sidebar",
            "typeset-sidebar-group", "typeset-sidebar-title",
            "typeset-sidebar-row", "typeset-sidebar-label",
            "typeset-sidebar-value", "typeset-card", "typeset-card-body",
            "typeset-badge", "typeset-reference", "typeset-ref-number",
            "typeset-ref-content", "typeset-ref-body", "typeset-ref-source",
            "typeset-reference-group", "typeset-ref-group-label",
        ]
        for style_name in STYLES:
            css = get_style(style_name)
            for cls in required_classes:
                assert cls in css, f"{style_name} style missing .{cls}"

    def test_variant_classes_in_each_style(self):
        for style_name in STYLES:
            css = get_style(style_name)
            assert "variant-pairs" in css, f"{style_name} missing .variant-pairs"
            assert "variant-right" in css, f"{style_name} missing .variant-right"
            assert "variant-left" in css, f"{style_name} missing .variant-left"

    def test_tone_classes_in_article(self):
        css = get_style("article")
        for tone in ["info", "warning", "success", "note", "quote"]:
            assert f"tone-{tone}" in css, f"article missing .tone-{tone}"
