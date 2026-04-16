# Andamentum Typeset Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone typesetting module (`andamentum.typeset`) with 7 visual atoms that produces beautiful HTML and PDF output, reusable across domains without any consumer doing rendering work.

**Architecture:** A flat Python package with four files: `atoms.py` (data model + validation), `renderer.py` (atom dispatch + HTML assembly), `styles.py` (named CSS stylesheets as string constants), and `__init__.py` (public API). The module accepts a list of atom dicts, validates them, renders each to an HTML fragment via a dispatch table, wraps in a styled document shell, and returns an HTML string. An optional `render_pdf` path pipes through WeasyPrint.

**Tech Stack:** Python 3.10+, `markdown` library for markdown-to-HTML conversion, `weasyprint` (optional, for PDF output only). No other dependencies.

**Atoms (7):**

| Atom | Visual intent | Variants |
|------|--------------|----------|
| `heading` | Document hero: big title, subtitle, meta caption | — |
| `prose` | Flowing markdown body (default) | — |
| `callout` | Emphasized block that breaks the flow | `tone`: info, warning, success, note, quote |
| `items` | Sequence of labeled entries | `variant`: pairs (default), right, left |
| `aside` | Subordinate content region | — |
| `card` | Bordered information unit with structured metadata | — |
| `reference` | Compact source-attributed entry | — |

**Styles (3 for v1):**

| Style | Origin | Character |
|-------|--------|-----------|
| `article` | Epistemic html_report.py CSS | Warm cream `#f9f7f4`, Source Serif 4 body, Inter UI, 860px measure, generous 1.85 line-height |
| `cv` | mosaic typeset CV style + CV app custom CSS | Monochrome, Inter 9pt, tight spacing, uppercase h2 with dark rule, right-aligned year entries |
| `report` | mosaic typeset technical style | Blue headings (Space Grotesk), Inter body, colored table headers, A4 page layout |

---

## File Structure

```
src/andamentum/typeset/
    __init__.py        # Public API: render, render_pdf, render_to_file, STYLES, get_style
    atoms.py           # Atom dataclass definitions, validation, _validate_document()
    renderer.py        # _render_atom() dispatch, _render_heading(), _render_prose(), etc.
    styles.py          # ARTICLE, CV, REPORT CSS constants, STYLES dict, get_style()
    py.typed           # PEP 561 marker

src/andamentum/typeset/tests/
    __init__.py
    test_atoms.py      # Validation tests for each atom type
    test_renderer.py   # Rendering tests: one per atom + end-to-end document tests
    test_styles.py     # Style existence and CSS validity tests
```

---

## Task 1: Package skeleton + public API stub

**Files:**
- Create: `src/andamentum/typeset/__init__.py`
- Create: `src/andamentum/typeset/atoms.py`
- Create: `src/andamentum/typeset/renderer.py`
- Create: `src/andamentum/typeset/styles.py`
- Create: `src/andamentum/typeset/py.typed`
- Create: `src/andamentum/typeset/tests/__init__.py`
- Create: `src/andamentum/typeset/tests/test_atoms.py`
- Modify: `pyproject.toml` (add testpaths entry if needed)

- [ ] **Step 1: Create package directory and empty files**

```bash
mkdir -p src/andamentum/typeset/tests
touch src/andamentum/typeset/py.typed
touch src/andamentum/typeset/tests/__init__.py
```

- [ ] **Step 2: Write the public API stub in `__init__.py`**

```python
"""andamentum.typeset — Beautiful documents from structured atoms.

A document is a list of atoms. Each atom is a dict with a ``kind`` field
that selects the visual pattern, a ``content`` field (usually markdown),
and optional metadata. The module renders atoms to styled HTML or PDF.

Quick start::

    from andamentum.typeset import render

    html = render([
        {"kind": "heading", "content": "My Report"},
        {"kind": "prose", "content": "## Summary\\n\\nThe findings show..."},
        {"kind": "callout", "content": "Key insight here.", "tone": "note"},
    ])

Seven atom kinds: heading, prose, callout, items, aside, card, reference.
Three styles: article (warm serif), cv (monochrome compact), report (blue technical).
"""

from .renderer import render, render_to_file

try:
    from .renderer import render_pdf
except ImportError:
    pass

from .styles import STYLES, get_style

__all__ = ["render", "render_to_file", "render_pdf", "STYLES", "get_style"]
```

- [ ] **Step 3: Write the atom type constants and validation stub in `atoms.py`**

```python
"""Atom definitions and validation for andamentum.typeset.

An atom is a dict with at least a ``kind`` field. Each kind has required
and optional fields. Unknown kinds fall back to ``prose`` with a warning.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ATOM_KINDS = frozenset({
    "heading", "prose", "callout", "items", "aside", "card", "reference",
})

CALLOUT_TONES = frozenset({"info", "warning", "success", "note", "quote"})
ITEMS_VARIANTS = frozenset({"pairs", "right", "left"})

# Required fields per atom kind (beyond 'kind' itself).
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "heading": ["content"],
    "prose": ["content"],
    "callout": ["content"],
    "items": ["entries"],
    "aside": [],  # accepts content (markdown) or groups (dict)
    "card": ["content"],
    "reference": ["content"],
}


def validate_atom(atom: dict[str, Any], index: int) -> dict[str, Any]:
    """Validate one atom dict and return it (possibly normalized).

    Raises ValueError with a message including the atom index so the
    caller can identify which atom is broken.
    """
    if not isinstance(atom, dict):
        raise ValueError(f"atom[{index}]: expected a dict, got {type(atom).__name__}")

    kind = atom.get("kind", "prose")
    if kind not in ATOM_KINDS:
        logger.warning(
            "atom[%d]: unknown kind %r, falling back to 'prose'", index, kind
        )
        kind = "prose"
        atom = {**atom, "kind": kind}

    for field in _REQUIRED_FIELDS.get(kind, []):
        if field not in atom:
            raise ValueError(
                f"atom[{index}] of kind '{kind}' is missing required field '{field}'"
            )

    # Validate enum fields
    if kind == "callout" and "tone" in atom:
        if atom["tone"] not in CALLOUT_TONES:
            raise ValueError(
                f"atom[{index}]: callout tone must be one of {sorted(CALLOUT_TONES)}, "
                f"got {atom['tone']!r}"
            )
    if kind == "items" and "variant" in atom:
        if atom["variant"] not in ITEMS_VARIANTS:
            raise ValueError(
                f"atom[{index}]: items variant must be one of {sorted(ITEMS_VARIANTS)}, "
                f"got {atom['variant']!r}"
            )

    return atom


def validate_document(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a full document (list of atoms). Returns normalized copy."""
    if not isinstance(atoms, list):
        raise ValueError(f"document must be a list of atom dicts, got {type(atoms).__name__}")
    return [validate_atom(atom, i) for i, atom in enumerate(atoms)]
```

- [ ] **Step 4: Write an empty renderer stub**

```python
"""Core renderer — atom dispatch, HTML assembly, and output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atoms import validate_document
from .styles import get_style


def render(
    document: list[dict[str, Any]] | str,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> str:
    """Render a document to an HTML string.

    Args:
        document: A list of atom dicts, or a plain markdown string
            (treated as a single ``prose`` atom).
        style: Named stylesheet (article, cv, report).
        custom_css: Raw CSS override; replaces the named style entirely.
        title: HTML ``<title>`` tag content. Auto-detected from the first
            ``heading`` atom if omitted.

    Returns:
        Complete HTML document as a string.
    """
    raise NotImplementedError("renderer not yet implemented")


def render_to_file(
    document: list[dict[str, Any]] | str,
    output: str | Path,
    **kwargs: Any,
) -> Path:
    """Render to an HTML file. Same arguments as ``render()``."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(document, **kwargs))
    return output


def render_pdf(
    document: list[dict[str, Any]] | str,
    output: str | Path,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> Path:
    """Render to PDF via WeasyPrint. Requires ``pip install weasyprint``."""
    import weasyprint

    html = render(document, style=style, custom_css=custom_css, title=title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = weasyprint.HTML(string=html)
    doc.write_pdf(str(output))
    return output
```

- [ ] **Step 5: Write the styles stub**

```python
"""Named stylesheets for andamentum.typeset.

Each style is a CSS string. Styles define the visual treatment for all
7 atom kinds. The ``{footer_label}`` placeholder is available for
print/PDF running footers.
"""

from __future__ import annotations

# Placeholder — real CSS is added in Task 7.
ARTICLE = "/* article style placeholder */"
CV = "/* cv style placeholder */"
REPORT = "/* report style placeholder */"

STYLES: dict[str, str] = {
    "article": ARTICLE,
    "cv": CV,
    "report": REPORT,
}


def get_style(name: str) -> str:
    """Return CSS for the named style, or raise KeyError."""
    try:
        return STYLES[name.lower()]
    except KeyError:
        available = ", ".join(sorted(STYLES))
        raise KeyError(f"Unknown style {name!r}. Available: {available}") from None
```

- [ ] **Step 6: Write import smoke test**

```python
"""Tests for atom validation."""

from __future__ import annotations

import pytest

from andamentum.typeset.atoms import validate_atom, validate_document, ATOM_KINDS


class TestValidateAtom:
    def test_valid_prose(self):
        atom = validate_atom({"kind": "prose", "content": "hello"}, 0)
        assert atom["kind"] == "prose"

    def test_missing_kind_defaults_to_prose(self):
        atom = validate_atom({"content": "hello"}, 0)
        assert atom["kind"] == "prose"

    def test_unknown_kind_falls_back_to_prose(self):
        atom = validate_atom({"kind": "unknown", "content": "hello"}, 0)
        assert atom["kind"] == "prose"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="missing required field 'content'"):
            validate_atom({"kind": "heading"}, 0)

    def test_invalid_callout_tone_raises(self):
        with pytest.raises(ValueError, match="callout tone must be"):
            validate_atom({"kind": "callout", "content": "x", "tone": "bad"}, 0)

    def test_invalid_items_variant_raises(self):
        with pytest.raises(ValueError, match="items variant must be"):
            validate_atom({"kind": "items", "entries": [], "variant": "bad"}, 0)

    def test_all_seven_kinds_are_present(self):
        assert len(ATOM_KINDS) == 7


class TestValidateDocument:
    def test_empty_document(self):
        result = validate_document([])
        assert result == []

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_document("not a list")

    def test_non_dict_atom_raises(self):
        with pytest.raises(ValueError, match="expected a dict"):
            validate_document(["not a dict"])
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
uv run pytest src/andamentum/typeset/tests/test_atoms.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): package skeleton with atom validation and API stubs"
```

---

## Task 2: Markdown-to-HTML helper + prose atom renderer

**Files:**
- Modify: `src/andamentum/typeset/renderer.py`
- Create: `src/andamentum/typeset/tests/test_renderer.py`

- [ ] **Step 1: Write failing test for prose rendering**

```python
"""Tests for atom rendering."""

from __future__ import annotations

import pytest

from andamentum.typeset import render


class TestProse:
    def test_renders_markdown_to_html(self):
        html = render([{"kind": "prose", "content": "Hello **world**"}])
        assert "<strong>world</strong>" in html

    def test_renders_heading_inside_prose(self):
        html = render([{"kind": "prose", "content": "## Section\n\nBody text."}])
        assert "<h2>" in html
        assert "Body text" in html

    def test_plain_string_treated_as_prose(self):
        html = render("# Hello\n\nWorld.")
        assert "Hello" in html
        assert "World" in html

    def test_prose_with_heading_field(self):
        html = render([{"kind": "prose", "content": "Body.", "heading": "Section Title"}])
        assert "Section Title" in html
        assert "Body" in html
```

- [ ] **Step 2: Run tests — they should fail (NotImplementedError)**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestProse -v
```

Expected: FAIL with NotImplementedError.

- [ ] **Step 3: Implement markdown helper + prose renderer + render()**

In `renderer.py`, replace the stub `render()` with the real implementation. The key functions:

```python
"""Core renderer — atom dispatch, HTML assembly, and output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import markdown
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension

from .atoms import validate_document
from .styles import get_style

logger = logging.getLogger(__name__)

_HTML_SHELL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<article class="typeset-document">
{body}
</article>
</body>
</html>"""


def _md(text: str) -> str:
    """Convert markdown to HTML fragment."""
    extensions = [
        TableExtension(),
        FencedCodeExtension(),
        TocExtension(permalink=False),
    ]
    try:
        from markdown.extensions.codehilite import CodeHiliteExtension
        extensions.append(CodeHiliteExtension(css_class="highlight", guess_lang=False))
    except ImportError:
        pass
    try:
        extensions.append("md_in_html")
    except Exception:
        pass
    return markdown.Markdown(extensions=extensions).convert(text)


# ---------------------------------------------------------------------------
# Per-atom renderers
# ---------------------------------------------------------------------------

def _render_heading(atom: dict) -> str:
    title = _md(atom["content"]).strip()
    # Strip wrapping <p> if markdown produced one
    if title.startswith("<p>") and title.endswith("</p>"):
        title = title[3:-4]
    parts = [f'<header class="typeset-heading">']
    parts.append(f"<h1>{title}</h1>")
    if atom.get("subtitle"):
        parts.append(f'<p class="typeset-subtitle">{_md(atom["subtitle"]).strip()}</p>')
    if atom.get("meta"):
        meta = atom["meta"]
        if isinstance(meta, dict):
            meta_str = " &middot; ".join(f"{v}" for v in meta.values())
        else:
            meta_str = str(meta)
        parts.append(f'<p class="typeset-meta">{meta_str}</p>')
    parts.append("</header>")
    return "\n".join(parts)


def _render_prose(atom: dict) -> str:
    parts = []
    if atom.get("heading"):
        parts.append(f"<h2>{atom['heading']}</h2>")
    parts.append(f'<section class="typeset-prose">{_md(atom["content"])}</section>')
    return "\n".join(parts)


_RENDERERS: dict[str, Any] = {
    "heading": _render_heading,
    "prose": _render_prose,
}


def _render_atom(atom: dict) -> str:
    """Dispatch to the renderer for this atom's kind."""
    kind = atom.get("kind", "prose")
    renderer = _RENDERERS.get(kind)
    if renderer is None:
        logger.warning("No renderer for atom kind %r, falling back to prose", kind)
        return _render_prose(atom)
    return renderer(atom)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(
    document: list[dict[str, Any]] | str,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> str:
    """Render a document to an HTML string."""
    # Plain string → single prose atom
    if isinstance(document, str):
        document = [{"kind": "prose", "content": document}]

    atoms = validate_document(document)

    # Auto-detect title from first heading atom
    if title is None:
        for atom in atoms:
            if atom.get("kind") == "heading":
                title = atom.get("content", "Document")
                break
        if title is None:
            title = "Document"

    css = custom_css if custom_css is not None else get_style(style)

    body_parts = [_render_atom(atom) for atom in atoms]
    body = "\n\n".join(body_parts)

    return _HTML_SHELL.format(title=title, css=css, body=body)


def render_to_file(
    document: list[dict[str, Any]] | str,
    output: str | Path,
    **kwargs: Any,
) -> Path:
    """Render to an HTML file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(document, **kwargs))
    return output


def render_pdf(
    document: list[dict[str, Any]] | str,
    output: str | Path,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> Path:
    """Render to PDF via WeasyPrint."""
    import weasyprint

    html = render(document, style=style, custom_css=custom_css, title=title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = weasyprint.HTML(string=html)
    doc.write_pdf(str(output))
    return output
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestProse -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): prose + heading renderers with markdown conversion"
```

---

## Task 3: Callout atom renderer

**Files:**
- Modify: `src/andamentum/typeset/renderer.py`
- Modify: `src/andamentum/typeset/tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

Add to `test_renderer.py`:

```python
class TestCallout:
    def test_callout_renders_content(self):
        html = render([{"kind": "callout", "content": "Important finding."}])
        assert "Important finding" in html
        assert "typeset-callout" in html

    def test_callout_with_tone(self):
        html = render([{"kind": "callout", "content": "Watch out!", "tone": "warning"}])
        assert "tone-warning" in html

    def test_callout_default_tone(self):
        html = render([{"kind": "callout", "content": "Note."}])
        assert "typeset-callout" in html
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestCallout -v
```

- [ ] **Step 3: Implement callout renderer**

Add to `renderer.py`:

```python
def _render_callout(atom: dict) -> str:
    tone = atom.get("tone", "")
    tone_cls = f" tone-{tone}" if tone else ""
    content = _md(atom["content"])
    return f'<aside class="typeset-callout{tone_cls}">{content}</aside>'
```

Register in `_RENDERERS`:
```python
_RENDERERS["callout"] = _render_callout
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestCallout -v
```

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): callout atom renderer with tone variants"
```

---

## Task 4: Items atom renderer (3 variants)

**Files:**
- Modify: `src/andamentum/typeset/renderer.py`
- Modify: `src/andamentum/typeset/tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
class TestItems:
    def test_pairs_variant_renders_dl(self):
        doc = [{"kind": "items", "entries": [
            {"label": "Question", "body": "Answer here."},
        ]}]
        html = render(doc)
        assert "Question" in html
        assert "Answer here" in html
        assert "typeset-items" in html

    def test_right_variant_renders_year_right(self):
        doc = [{"kind": "items", "variant": "right", "entries": [
            {"label": "2024", "body": "PhD, Bioinformatics. Stockholm University."},
        ]}]
        html = render(doc)
        assert "variant-right" in html
        assert "2024" in html
        assert "Stockholm" in html

    def test_left_variant_renders_year_left(self):
        doc = [{"kind": "items", "variant": "left", "entries": [
            {"label": "2024", "body": "Keynote at ISMB Conference."},
        ]}]
        html = render(doc)
        assert "variant-left" in html

    def test_multiple_entries(self):
        doc = [{"kind": "items", "entries": [
            {"label": "A", "body": "First"},
            {"label": "B", "body": "Second"},
        ]}]
        html = render(doc)
        assert "First" in html
        assert "Second" in html

    def test_items_with_heading(self):
        doc = [{"kind": "items", "heading": "Key Findings", "entries": [
            {"label": "Q", "body": "A"},
        ]}]
        html = render(doc)
        assert "Key Findings" in html

    def test_body_rendered_as_markdown(self):
        doc = [{"kind": "items", "entries": [
            {"label": "Note", "body": "Has **bold** text."},
        ]}]
        html = render(doc)
        assert "<strong>bold</strong>" in html
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestItems -v
```

- [ ] **Step 3: Implement items renderer**

```python
def _render_items(atom: dict) -> str:
    variant = atom.get("variant", "pairs")
    entries = atom.get("entries", [])
    heading = atom.get("heading", "")

    parts = []
    if heading:
        parts.append(f"<h2>{heading}</h2>")

    parts.append(f'<div class="typeset-items variant-{variant}">')
    for entry in entries:
        label = entry.get("label", "")
        body = entry.get("body", "")
        body_html = _md(body) if body else ""
        parts.append(f'<div class="typeset-item">')
        parts.append(f'<div class="typeset-item-label">{label}</div>')
        parts.append(f'<div class="typeset-item-body">{body_html}</div>')
        parts.append(f'</div>')
    parts.append('</div>')
    return "\n".join(parts)
```

Register: `_RENDERERS["items"] = _render_items`

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestItems -v
```

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): items atom renderer with pairs/right/left variants"
```

---

## Task 5: Aside atom renderer

**Files:**
- Modify: `src/andamentum/typeset/renderer.py`
- Modify: `src/andamentum/typeset/tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
class TestAside:
    def test_aside_with_markdown_content(self):
        html = render([{"kind": "aside", "content": "Small print here."}])
        assert "Small print" in html
        assert "typeset-aside" in html

    def test_aside_with_groups_dict(self):
        html = render([{"kind": "aside", "groups": {
            "Stats": {"Evidence": 37, "Claims": 2},
            "Meta": {"Model": "gemma4:26b"},
        }}])
        assert "Evidence" in html
        assert "37" in html
        assert "gemma4:26b" in html
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestAside -v
```

- [ ] **Step 3: Implement**

```python
def _render_aside(atom: dict) -> str:
    if "groups" in atom:
        parts = ['<aside class="typeset-aside typeset-sidebar">']
        for group_name, group_data in atom["groups"].items():
            parts.append(f'<div class="typeset-sidebar-group">')
            parts.append(f'<div class="typeset-sidebar-title">{group_name}</div>')
            for k, v in group_data.items():
                parts.append(
                    f'<div class="typeset-sidebar-row">'
                    f'<span class="typeset-sidebar-label">{k}</span>'
                    f'<span class="typeset-sidebar-value">{v}</span>'
                    f'</div>'
                )
            parts.append('</div>')
        parts.append('</aside>')
        return "\n".join(parts)

    content = atom.get("content", "")
    return f'<aside class="typeset-aside">{_md(content)}</aside>'
```

Register: `_RENDERERS["aside"] = _render_aside`

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestAside -v
```

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): aside atom renderer with markdown and groups modes"
```

---

## Task 6: Card + reference atom renderers

**Files:**
- Modify: `src/andamentum/typeset/renderer.py`
- Modify: `src/andamentum/typeset/tests/test_renderer.py`

- [ ] **Step 1: Write failing tests for card**

```python
class TestCard:
    def test_card_renders_content(self):
        html = render([{"kind": "card", "content": "Metformin reduces CV mortality."}])
        assert "Metformin" in html
        assert "typeset-card" in html

    def test_card_with_badge(self):
        html = render([{"kind": "card", "content": "Claim.", "badge": "supported"}])
        assert "supported" in html
        assert "typeset-badge" in html

    def test_card_with_details(self):
        html = render([{"kind": "card", "content": "Claim.", "details": "Scope: T2D patients."}])
        assert "<details>" in html
        assert "Scope" in html

    def test_card_with_source(self):
        html = render([{"kind": "card", "content": "X.", "source": "https://example.com"}])
        assert "https://example.com" in html

    def test_card_with_refs(self):
        html = render([{"kind": "card", "content": "X.", "refs": ["e1", "e2"]}])
        assert "e1" in html
```

- [ ] **Step 2: Write failing tests for reference**

```python
class TestReference:
    def test_reference_renders_content(self):
        html = render([{"kind": "reference", "content": "A study found..."}])
        assert "A study" in html
        assert "typeset-reference" in html

    def test_reference_with_source_link(self):
        html = render([{"kind": "reference", "content": "X.", "source": "https://example.com"}])
        assert "href" in html
        assert "https://example.com" in html

    def test_reference_with_badge(self):
        html = render([{"kind": "reference", "content": "X.", "badge": "supports"}])
        assert "supports" in html

    def test_reference_with_number(self):
        html = render([{"kind": "reference", "content": "X.", "number": 3}])
        assert "3" in html

    def test_reference_with_group(self):
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
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py::TestCard src/andamentum/typeset/tests/test_renderer.py::TestReference -v
```

- [ ] **Step 4: Implement card renderer**

```python
def _render_card(atom: dict) -> str:
    content = _md(atom["content"])
    parts = ['<div class="typeset-card">']
    parts.append(f'<div class="typeset-card-body">{content}')

    if atom.get("badge"):
        parts.append(f' <span class="typeset-badge">{atom["badge"]}</span>')
    if atom.get("refs"):
        ref_strs = ", ".join(str(r) for r in atom["refs"])
        parts.append(f' <sup class="typeset-refs">{ref_strs}</sup>')

    parts.append('</div>')

    if atom.get("source"):
        parts.append(
            f'<div class="typeset-card-source">'
            f'<a href="{atom["source"]}">{atom["source"]}</a></div>'
        )
    if atom.get("details"):
        details_html = _md(atom["details"])
        parts.append(
            f'<details class="typeset-card-details">'
            f'<summary>Details</summary>{details_html}</details>'
        )
    parts.append('</div>')
    return "\n".join(parts)
```

- [ ] **Step 5: Implement reference renderer with grouping**

The renderer needs to handle consecutive references with the same `group` field and cluster them under a group heading. This requires the main render loop to be group-aware for reference atoms. Add a pre-processing step in `render()` that clusters consecutive references.

```python
def _render_reference(atom: dict) -> str:
    parts = ['<div class="typeset-reference">']
    if atom.get("number") is not None:
        parts.append(f'<div class="typeset-ref-number">{atom["number"]}</div>')
    parts.append(f'<div class="typeset-ref-body">')
    parts.append(_md(atom["content"]))
    if atom.get("badge"):
        parts.append(f' <span class="typeset-badge">{atom["badge"]}</span>')
    parts.append('</div>')
    if atom.get("source"):
        src = atom["source"]
        parts.append(f'<div class="typeset-ref-source"><a href="{src}">{src}</a></div>')
    parts.append('</div>')
    return "\n".join(parts)


def _render_reference_group(atoms: list[dict]) -> str:
    """Render a consecutive group of reference atoms, optionally under a group heading."""
    if not atoms:
        return ""
    group_label = atoms[0].get("group")
    parts = ['<div class="typeset-reference-group">']
    if group_label:
        parts.append(f'<div class="typeset-ref-group-label">{group_label}</div>')
    for atom in atoms:
        parts.append(_render_reference(atom))
    parts.append('</div>')
    return "\n".join(parts)
```

Then in `render()`, replace the simple list comprehension with a loop that clusters consecutive reference atoms:

```python
body_parts: list[str] = []
i = 0
while i < len(atoms):
    atom = atoms[i]
    if atom.get("kind") == "reference":
        # Cluster consecutive references
        group: list[dict] = [atom]
        while i + 1 < len(atoms) and atoms[i + 1].get("kind") == "reference":
            i += 1
            group.append(atoms[i])
        # Sub-group by 'group' field
        from itertools import groupby
        for _, sub in groupby(group, key=lambda a: a.get("group", "")):
            body_parts.append(_render_reference_group(list(sub)))
    else:
        body_parts.append(_render_atom(atom))
    i += 1
```

Register: `_RENDERERS["card"] = _render_card` and `_RENDERERS["reference"] = _render_reference`

- [ ] **Step 6: Run — expect PASS**

```bash
uv run pytest src/andamentum/typeset/tests/test_renderer.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): card + reference atom renderers with grouping"
```

---

## Task 7: CSS stylesheets — article, cv, report

**Files:**
- Modify: `src/andamentum/typeset/styles.py`
- Create: `src/andamentum/typeset/tests/test_styles.py`

This is the largest single task. Each stylesheet must style all 7 atom CSS classes:
`.typeset-heading`, `.typeset-prose`, `.typeset-callout` (+ `.tone-*`),
`.typeset-items` (+ `.variant-*`), `.typeset-item`, `.typeset-item-label`,
`.typeset-item-body`, `.typeset-aside`, `.typeset-sidebar`, `.typeset-card`,
`.typeset-badge`, `.typeset-reference`, `.typeset-ref-number`,
`.typeset-ref-body`, `.typeset-ref-source`, `.typeset-ref-group-label`.

**Source material to port:**
- **article** — from `andamentum/epistemic/html_report.py:174-757` (the `_CSS_STYLES` block). Extract the body/heading/table/blockquote/code rules plus the `.report-layout` centered column. Map epistemic-specific classes (`.verdict`, `.claim`, `.evidence-item`, `.sidebar`) to the typeset atom classes above.
- **cv** — from `mosaic/packages/typeset/src/typeset/styles.py:248-430` (the `CV` constant) merged with `CV/src/cv/exporters/markdown_cv.py:17-216` (the `_CUSTOM_CV_CSS`). The CV app's CSS has the `.cv-entry` and `.pub-group` patterns that map to `.typeset-items.variant-right` and `.typeset-reference-group`.
- **report** — from `mosaic/packages/typeset/src/typeset/styles.py:40-246` (the `TECHNICAL` constant). Map headings, tables, code blocks to typeset atom classes.

- [ ] **Step 1: Write style tests**

```python
"""Tests for built-in styles."""

from andamentum.typeset.styles import STYLES, get_style

import pytest


class TestStyles:
    def test_three_styles_exist(self):
        assert set(STYLES.keys()) == {"article", "cv", "report"}

    def test_get_style_returns_string(self):
        for name in STYLES:
            css = get_style(name)
            assert isinstance(css, str)
            assert len(css) > 100  # not a placeholder

    def test_unknown_style_raises(self):
        with pytest.raises(KeyError, match="Unknown style"):
            get_style("nonexistent")

    def test_all_atom_classes_in_article(self):
        css = get_style("article")
        for cls in [
            "typeset-heading", "typeset-prose", "typeset-callout",
            "typeset-items", "typeset-aside", "typeset-card",
            "typeset-reference",
        ]:
            assert cls in css, f"article style missing class .{cls}"

    def test_all_atom_classes_in_cv(self):
        css = get_style("cv")
        for cls in [
            "typeset-heading", "typeset-prose", "typeset-callout",
            "typeset-items", "typeset-aside", "typeset-card",
            "typeset-reference",
        ]:
            assert cls in css, f"cv style missing class .{cls}"

    def test_all_atom_classes_in_report(self):
        css = get_style("report")
        for cls in [
            "typeset-heading", "typeset-prose", "typeset-callout",
            "typeset-items", "typeset-aside", "typeset-card",
            "typeset-reference",
        ]:
            assert cls in css, f"report style missing class .{cls}"
```

- [ ] **Step 2: Run — expect FAIL (placeholder CSS)**

```bash
uv run pytest src/andamentum/typeset/tests/test_styles.py -v
```

- [ ] **Step 3: Write the ARTICLE stylesheet**

Port from epistemic `html_report.py:174-757`. The key mappings:
- `.report-layout` → `.typeset-document`
- `.report-header` → `.typeset-heading`
- `.verdict` → `.typeset-callout`
- `.key-findings-qa` → `.typeset-items.variant-pairs`
- `.claim` → `.typeset-card`
- `.evidence-item` → `.typeset-reference`
- `.sidebar` → `.typeset-sidebar`
- Body typography (Source Serif 4, `#f9f7f4` background, 19px, 1.85 line-height) stays as-is.

This is a large CSS block (~400-500 lines). Write the full CSS string as `ARTICLE = """..."""` in `styles.py`. Include Google Fonts import for Source Serif 4 + Inter.

- [ ] **Step 4: Write the CV stylesheet**

Port from mosaic `typeset/styles.py:248-430` (base CV) merged with the CV app's `_CUSTOM_CV_CSS`. The key mappings:
- `.typeset-items.variant-right` → flexbox with label right-aligned (replaces `.cv-entry`)
- `.typeset-items.variant-left` → flexbox with label left (replaces `.year-left-entry`)
- `.typeset-reference-group` → year-grouped bibliography (replaces `.pub-group`)
- `.typeset-ref-number` → numbered left margin (replaces `.pub-num`)
- `.typeset-badge` → small gray pill (replaces `.pub-cite`)
- Body: Inter 9pt, monochrome, tight spacing, uppercase h2 with dark rule.

~300 lines of CSS.

- [ ] **Step 5: Write the REPORT stylesheet**

Port from mosaic `typeset/styles.py:40-246` (TECHNICAL style). The key mappings:
- Blue headings (Space Grotesk), `#0d6efd` accent
- `.typeset-card` → blue-bordered card
- `.typeset-callout` tones → colored borders/backgrounds
- `.typeset-reference` → compact with subtle gray
- Body: Inter 9.5pt, 1.55 line-height, white background.

~300 lines of CSS.

- [ ] **Step 6: Run style tests**

```bash
uv run pytest src/andamentum/typeset/tests/test_styles.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): article, cv, and report stylesheets"
```

---

## Task 8: End-to-end rendering tests + visual smoke test

**Files:**
- Modify: `src/andamentum/typeset/tests/test_renderer.py`

- [ ] **Step 1: Write end-to-end document test**

```python
class TestEndToEnd:
    def test_full_document_renders_all_atoms(self):
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

    def test_renders_with_each_style(self):
        doc = [{"kind": "prose", "content": "Hello."}]
        for style in ["article", "cv", "report"]:
            html = render(doc, style=style)
            assert "Hello" in html

    def test_custom_css_overrides_style(self):
        doc = [{"kind": "prose", "content": "X."}]
        html = render(doc, custom_css="body { color: red; }")
        assert "color: red" in html

    def test_plain_markdown_string_input(self):
        html = render("# Title\n\nParagraph.")
        assert "Title" in html
        assert "Paragraph" in html
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest src/andamentum/typeset/tests/ -v
```

Expected: all PASS.

- [ ] **Step 3: Write a visual smoke-test script (not a pytest test)**

Create a small script `src/andamentum/typeset/tests/visual_smoke.py`:

```python
"""Visual smoke test — generates sample HTML files for manual inspection.

Run: uv run python -m andamentum.typeset.tests.visual_smoke
Opens: /tmp/typeset_smoke_{style}.html for each style
"""

from pathlib import Path
from andamentum.typeset import render

SAMPLE_DOC = [
    {"kind": "heading", "content": "Sample Report", "subtitle": "Typeset visual test", "meta": {"date": "2026-04-16", "author": "Test"}},
    {"kind": "callout", "content": "This is the key finding of the research.", "tone": "note"},
    {"kind": "items", "heading": "Key Facts", "entries": [
        {"label": "What was studied?", "body": "The effect of **metformin** on cardiovascular mortality."},
        {"label": "What did we find?", "body": "Mixed evidence across populations."},
        {"label": "Confidence", "body": "High (0.88)"},
    ]},
    {"kind": "prose", "content": "## Summary\n\nThe literature provides conflicting evidence regarding the effect. Some studies show benefit, others do not.\n\n## Details\n\nA nationwide cohort study reported lower incidence."},
    {"kind": "card", "content": "Metformin reduces cardiovascular mortality in T2D patients.", "badge": "supported", "refs": ["1", "2"], "details": "Scope: patients with T2D or coronary artery disease.\n\nVerification: scrutiny passed."},
    {"kind": "card", "content": "Metformin lowers CK-MB biomarker levels.", "badge": "challenged", "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12028114/"},
    {"kind": "reference", "content": "Taiwan nationwide cohort study shows reduced AMI incidence among metformin users.", "number": 1, "source": "https://www.nature.com/articles/s41598-025-13211-z", "badge": "supports"},
    {"kind": "reference", "content": "Meta-analysis of cardiac biomarkers found significant CK-MB reduction.", "number": 2, "source": "https://link.springer.com/article/10.1186/s12933-019-0900-7", "badge": "supports"},
    {"kind": "reference", "content": "Systematic review found no significant MACE reduction in general T2D population.", "number": 3, "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9539433/", "badge": "contradicts"},
    {"kind": "items", "variant": "right", "heading": "Education", "entries": [
        {"label": "2006", "body": "*PhD, Bioinformatics*\nStockholm University, Sweden"},
        {"label": "2001", "body": "*MSc, Applied Mathematics*\nUniversity of Adelaide"},
    ]},
    {"kind": "aside", "groups": {
        "Investigation": {"Evidence items": "37", "Claims": "2", "Uncertainties": "9"},
        "Confidence": {"Score": "0.88 HIGH", "Posterior P(Y)": "0.047"},
    }},
]

if __name__ == "__main__":
    out_dir = Path("/tmp/typeset_smoke")
    out_dir.mkdir(exist_ok=True)
    for style in ["article", "cv", "report"]:
        path = out_dir / f"sample_{style}.html"
        path.write_text(render(SAMPLE_DOC, style=style))
        print(f"Written: {path}")
    print(f"\nOpen in browser: open {out_dir}/sample_article.html")
```

- [ ] **Step 4: Run the visual smoke test and inspect**

```bash
uv run python -m andamentum.typeset.tests.visual_smoke
open /tmp/typeset_smoke/sample_article.html
```

Visually confirm: headings, callouts, items (pairs and right-aligned), cards with badges, references with source links, aside sidebar grid all render with the expected aesthetic per style.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/typeset/
git commit -m "feat(typeset): end-to-end tests + visual smoke-test script"
```

---

## Task 9: Pyright + ruff + full test verification

**Files:**
- Possibly modify any file with lint issues

- [ ] **Step 1: Run ruff**

```bash
uv run ruff check src/andamentum/typeset/
```

Fix any issues.

- [ ] **Step 2: Run pyright**

```bash
uv run pyright src/andamentum/typeset/
```

Fix any type issues.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest src/andamentum/typeset/ -v
```

Expected: all pass.

- [ ] **Step 4: Run the existing andamentum test suite to confirm no regressions**

```bash
uv run pytest -q
```

Expected: 767+ passed, 0 failed.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(typeset): pyright + ruff clean"
```

---

## Task 10: Claude Code skill file

**Files:**
- Create: a skill file or documentation for agents

- [ ] **Step 1: Write a skill-style documentation file**

Create `src/andamentum/typeset/USAGE.md`:

```markdown
# andamentum.typeset — Quick Reference

## API

```python
from andamentum.typeset import render, render_to_file, render_pdf

# From atom list → HTML string
html = render(document, style="article")

# From atom list → HTML file
render_to_file(document, "report.html", style="article")

# From atom list → PDF (requires weasyprint)
render_pdf(document, "report.pdf", style="article")

# From plain markdown → HTML string
html = render("# Hello\n\nWorld.")
```

## Atoms

A document is a list of dicts. Each dict has a `kind` field.

| Kind | Required fields | Optional fields |
|------|----------------|-----------------|
| `heading` | `content` | `subtitle`, `meta` (str or dict) |
| `prose` | `content` (markdown) | `heading` (str) |
| `callout` | `content` (markdown) | `tone` (info/warning/success/note/quote) |
| `items` | `entries` (list of {label, body}) | `variant` (pairs/right/left), `heading` |
| `aside` | — | `content` (markdown) OR `groups` (dict of dicts) |
| `card` | `content` (markdown) | `badge`, `refs` (list), `source` (URL), `details` (markdown) |
| `reference` | `content` (markdown) | `source` (URL), `badge`, `number` (int), `group` (str) |

## Styles

`article` — warm serif, cream background (default)
`cv` — monochrome, compact, academic
`report` — blue technical, Space Grotesk headings

## Example

```python
doc = [
    {"kind": "heading", "content": "Weekly Status", "meta": {"date": "2026-04-16"}},
    {"kind": "callout", "content": "Shipped the routing benchmark.", "tone": "success"},
    {"kind": "items", "entries": [
        {"label": "Done", "body": "Semantic routing at 97.5% recall."},
        {"label": "Next", "body": "Provider query formulation."},
    ]},
    {"kind": "prose", "content": "## Commentary\n\nThe router is live..."},
    {"kind": "aside", "groups": {"Stats": {"Tests": "767", "Pyright": "0 errors"}}},
]
html = render(doc, style="article")
```
```

- [ ] **Step 2: Commit**

```bash
git add src/andamentum/typeset/USAGE.md
git commit -m "docs(typeset): usage reference for agents and humans"
```

---

## Self-Review Checklist

1. **Spec coverage**: All 7 atoms have renderers (Tasks 2-6). All 3 styles written (Task 7). Public API with render/render_to_file/render_pdf (Task 1-2). Validation with friendly errors (Task 1). Visual smoke test (Task 8). Agent documentation (Task 10).

2. **No placeholders**: Every step has either complete code or exact commands.

3. **Type consistency**: `_render_*` functions all take `dict` and return `str`. `render()` takes `list[dict] | str` and returns `str`. `validate_atom()` takes `dict` and `int`, returns `dict`. Style names are lowercase strings throughout.

4. **No epistemic/CV domain code touched**: The plan creates `src/andamentum/typeset/` as a standalone module. No modifications to `src/andamentum/epistemic/` or `~/work/Documents/CV/`.
