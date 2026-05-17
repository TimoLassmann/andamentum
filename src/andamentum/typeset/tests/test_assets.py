"""The packaged components.css must match docs/design/components.css.

The article style is loaded from src/andamentum/typeset/assets/components.css,
which is a packaged copy of the canonical design system at docs/design/
components.css. When the design source is updated, the package copy must be
re-synced. This test fails loudly when the two drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DESIGN_CSS = _REPO_ROOT / "docs" / "design" / "components.css"
_PACKAGE_CSS = Path(__file__).resolve().parents[1] / "assets" / "components.css"


def test_design_source_and_package_copy_are_byte_identical() -> None:
    if not _DESIGN_CSS.exists():
        pytest.skip(
            "docs/design/components.css not present — running outside the repo tree."
        )

    design = _DESIGN_CSS.read_bytes()
    packaged = _PACKAGE_CSS.read_bytes()
    if design != packaged:
        msg = (
            "components.css has drifted between the design source and the "
            "package copy.\n"
            f"  design:   {_DESIGN_CSS}\n"
            f"  package:  {_PACKAGE_CSS}\n"
            "Re-sync with:\n"
            f"  cp {_DESIGN_CSS} {_PACKAGE_CSS}"
        )
        raise AssertionError(msg)


def test_package_components_css_is_loaded_as_article_style() -> None:
    from andamentum.typeset.styles import ARTICLE

    assert ARTICLE == _PACKAGE_CSS.read_text(encoding="utf-8")
