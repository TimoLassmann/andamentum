# andamentum.typeset — Quick Reference

## API

```python
from andamentum.typeset import render, render_to_file, render_pdf

# From atom list -> HTML string
html = render(document, style="article")

# From atom list -> HTML file
render_to_file(document, "report.html", style="article")

# From atom list -> PDF (requires weasyprint)
render_pdf(document, "report.pdf", style="article")

# From plain markdown -> HTML string
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

- `article` — warm serif, cream background (default). Source Serif 4 + Inter.
- `cv` — monochrome, compact, academic. Inter 9pt.
- `report` — blue technical. Inter + Space Grotesk.

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

## Visual Smoke Test

Generate sample HTML files for all three styles:

```bash
uv run python -m andamentum.typeset.tests.visual_smoke
open /tmp/typeset_smoke/sample_article.html
```
