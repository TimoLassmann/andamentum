# andamentum.typeset

A document typesetting system with 7 visual atoms, 3 named styles, and HTML + PDF output.

A document is a list of dicts. Each dict has a `kind` field that tells the renderer what visual pattern to use. Content is written in markdown. The renderer handles typography, layout, and styling — the author focuses on what to say and which atom to put it in.

## Quick start

```python
from andamentum.typeset import render, render_to_file, render_pdf

# Build a document as a list of atom dicts
doc = [
    {"kind": "heading", "content": "My Report", "meta": {"date": "2026-04-16"}},
    {"kind": "callout", "content": "The key finding of this research."},
    {"kind": "prose", "content": "## Summary\n\nThe evidence shows..."},
]

# Render to HTML string
html = render(doc)

# Render to HTML file
render_to_file(doc, "report.html")

# Render to PDF (requires weasyprint — see PDF section below)
render_pdf(doc, "report.pdf", footer="Draft — April 2026")

# Plain markdown also works (treated as a single prose atom)
html = render("# Hello\n\nWorld.")
```

## The 7 atoms

### `heading` — document title block

The top of the document. Large serif title, optional subtitle, optional meta caption in small gray text. Separated from the body by a thin horizontal rule.

```python
{"kind": "heading",
 "content": "What does metformin do to CV mortality?",
 "subtitle": "A systematic review",
 "meta": {"date": "2026-04-15", "model": "gemma4:26b", "project": "metformin"}}
```

- `content` (required): the document title. Rendered as `<h1>`.
- `subtitle` (optional): rendered below the title in lighter weight.
- `meta` (optional): if a dict, values are joined with ` · `. If a string, used directly. Rendered as a small gray caption line.

Every document should start with exactly one heading atom.

### `prose` — flowing markdown body

The workhorse atom. Most of your document will be prose. Write standard markdown — headings, paragraphs, lists, tables, images, code blocks, blockquotes, links, bold, italic — and the renderer converts it to styled HTML.

```python
{"kind": "prose",
 "content": "## Summary\n\nThe literature provides **conflicting** evidence.\n\n- Point one\n- Point two"}
```

- `content` (required): markdown text. All standard markdown features work.
- `heading` (optional): if provided, a `<h2>` is rendered before the prose body. Useful for section titles when the prose content doesn't start with a heading.

```python
# These two are equivalent:
{"kind": "prose", "heading": "Summary", "content": "The evidence shows..."}
{"kind": "prose", "content": "## Summary\n\nThe evidence shows..."}
```

### `callout` — emphasized block

Breaks the flow to draw attention to something. Use it for the verdict, a key insight, a warning, or a pull quote.

**Without a tone** (default): renders as flowing text at slightly larger size, like a natural opening paragraph. No border, no background — very minimal, matching the epistemic report's verdict style.

```python
{"kind": "callout",
 "content": "The evidence is mixed, showing reduced mortality in some populations but no significant effect in others."}
```

**With a tone**: renders as a subtle box with a thin colored left border.

```python
{"kind": "callout", "content": "All systems operational.", "tone": "success"}
{"kind": "callout", "content": "This data has not been peer-reviewed.", "tone": "warning"}
{"kind": "callout", "content": "ClinicalTrials achieved 100% accuracy.", "tone": "info"}
{"kind": "callout", "content": "Consider the broader implications.", "tone": "note"}
{"kind": "callout", "content": "Science is the belief in the ignorance of experts.", "tone": "quote"}
```

- `content` (required): markdown text.
- `tone` (optional): one of `info`, `warning`, `success`, `note`, `quote`. Each gets a different accent color. Omit for the minimal flowing-text style.

### `items` — labeled entries

A sequence of labeled items. Each entry has a `label` and a `body`. Three visual layouts are available via the `variant` field.

**`variant="pairs"` (default)**: label above, body below. Good for Q&A blocks, glossaries, key-value summaries. Renders in a soft beige box with sans-serif font.

```python
{"kind": "items", "entries": [
    {"label": "What was studied?", "body": "The effect of **metformin** on cardiovascular mortality."},
    {"label": "What did we find?", "body": "Mixed evidence across populations."},
    {"label": "Confidence", "body": "High (0.88)"},
]}
```

**`variant="right"`**: label right-aligned, body on the left. The classic CV entry layout — year on the right, content on the left. Good for education, positions, awards, any chronological list.

```python
{"kind": "items", "variant": "right", "entries": [
    {"label": "2019–present", "body": "**Head, Computational Biology**\nTelethon Kids Institute, Perth"},
    {"label": "2014–2019", "body": "**Senior Research Fellow**\nTelethon Kids Institute, Perth"},
]}
```

**`variant="left"`**: label on the left, body on the right. Good for teaching, presentations, press — entries where the year is a header and the body is the description.

```python
{"kind": "items", "variant": "left", "entries": [
    {"label": "2024", "body": "Deep Learning in Genomics. Australian Bioinformatics Conference."},
    {"label": "2023", "body": "Introduction to Bioinformatics. University of Western Australia."},
]}
```

- `entries` (required): list of dicts, each with `label` (string) and `body` (markdown string).
- `variant` (optional): `"pairs"` (default), `"right"`, or `"left"`.
- `heading` (optional): if provided, a `<h2>` is rendered above the items block.

### `aside` — subordinate content

A visually subordinate region for metadata, author bios, colophons, or side notes. Always appears smaller and lighter than the main content.

**Content mode**: pass a markdown string.

```python
{"kind": "aside", "content": "Generated by andamentum.typeset on 2026-04-16."}
```

**Groups mode** (sidebar): pass a dict of dicts. Rendered as a multi-column metadata grid with tiny uppercase group titles — the same pattern as the epistemic report's sidebar.

```python
{"kind": "aside", "groups": {
    "Investigation": {"Evidence items": "37", "Claims": "2", "Uncertainties": "9"},
    "Confidence": {"Score": "0.88 HIGH", "Posterior P(Y)": "0.047"},
    "Model": {"LLM": "gemma4:26b", "Embeddings": "embeddinggemma"},
}}
```

- `content` (optional): markdown text for simple asides.
- `groups` (optional): dict of `{group_name: {label: value, ...}}` for sidebar grids. Use this mode at the end of a document for metadata.

### `card` — bordered information unit

A bordered block for a distinct assertion, finding, or claim. Has slots for a status badge, citation references, a source URL, and collapsible details. Use it for anything that's a standalone "unit of information" with metadata.

```python
{"kind": "card",
 "content": "Metformin reduces cardiovascular mortality in T2D patients.",
 "badge": "supported",
 "refs": ["1", "2"],
 "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12028114/",
 "details": "**Scope:** Patients with T2D or coronary artery disease.\n\n**Verification:** Scrutiny passed."}
```

- `content` (required): the main statement or finding, in markdown.
- `badge` (optional): a short status label rendered as a small pill. Common values get automatic coloring in the article style:
  - Green: `supported`, `pass`, `approved`
  - Red: `challenged`, `contradicts`, `fail`, `rejected`
  - Gray: anything else (e.g., `"v2.1"`, `"draft"`, `"under investigation"`)
- `refs` (optional): list of citation identifiers (strings or numbers) rendered as superscripts.
- `source` (optional): URL rendered as a small link below the body.
- `details` (optional): markdown text rendered inside a collapsible `<details>` element.

### `reference` — compact source entry

A bibliographic-style entry with a number, body text, source link, and badge. Use it for evidence items, publication lists, footnotes, changelogs, or any numbered reference list.

```python
{"kind": "reference",
 "number": 1,
 "content": "Taiwan nationwide cohort study shows reduced AMI incidence among metformin users.",
 "source": "https://www.nature.com/articles/s41598-025-13211-z",
 "badge": "supports"}
```

- `content` (required): the reference text, in markdown. Bold author names, italic journals, etc. all work.
- `number` (optional): displayed as a left-margin counter (e.g., `1.`, `2.`).
- `source` (optional): URL rendered as a small link below the body.
- `badge` (optional): a status pill (same coloring rules as card badges).
- `group` (optional): when consecutive references share the same `group` value, they are clustered under a group heading. Useful for year-grouped publication lists:

```python
{"kind": "reference", "number": 85, "group": "2025", "content": "**Lassmann T**, et al. ..."},
{"kind": "reference", "number": 84, "group": "2025", "content": "Smith J, **Lassmann T**. ..."},
{"kind": "reference", "number": 83, "group": "2024", "content": "**Lassmann T**, Jones A. ..."},
```

This renders "2025" as a heading above references 85 and 84, then "2024" as a heading above reference 83.

## Styles

Three built-in styles. Pick one with `style="name"`.

### `article` (default)

The epistemic report aesthetic. Warm cream background (`#f9f7f4`), Source Serif 4 body at 19px with 1.85 line-height, Inter for UI elements (badges, sidebar, meta). Centered 860px column. Generous whitespace. Callouts are minimal flowing text by default; toned callouts get subtle left-border accents. Cards separated by hair-thin rules. Sidebar as a 3-column grid in tiny uppercase gray.

Best for: research reports, analysis summaries, epistemic reports, any long-form document meant to be read on screen.

### `cv`

Monochrome academic CV layout. Inter throughout at 9pt, 1.45 line-height. Uppercase bold h2 headings with dark bottom borders. No color accents. Print-first: A4 page size, margins, page numbers, and optional running footer. First page hides the footer.

Best for: academic CVs, biosketches, professional resumes. Optimized for PDF output.

### `report`

Blue technical report. Inter body with Space Grotesk headings in blue (`#0d6efd`). Blue-accented cards and callouts. Colored table headers. Dark code blocks. A4 print layout with page numbers.

Best for: technical reports, benchmark results, pipeline documentation, data analysis outputs.

### Custom CSS

For full control, pass your own CSS string:

```python
html = render(doc, custom_css="body { font-family: Georgia; font-size: 14px; }")
```

This replaces the built-in style entirely. Your CSS must style all the `.typeset-*` classes yourself.

## PDF output

PDF rendering uses WeasyPrint. It's an optional dependency — install it separately.

### Setup (macOS)

```bash
pip install weasyprint
brew install pango libffi    # WeasyPrint needs these system libraries
```

### Usage

```python
from andamentum.typeset import render_pdf

render_pdf(doc, "report.pdf", style="article")
render_pdf(doc, "cv.pdf", style="cv", footer="April 2026")
```

- `footer`: text shown in the running page footer (bottom-left in CV style, varies by style). Empty string by default.
- The CV style hides the footer on page 1 (standard CV convention).
- All styles include `@page` rules for A4, proper margins, page-break control on headings and items, and orphan/widow protection.

### WeasyPrint notes

- WeasyPrint's flex layout doesn't work with `<p>` tags inside flex children. The renderer strips `<p>` wrapping from item and reference bodies to work around this. This is transparent — you don't need to do anything.
- If you get `OSError: cannot load library 'libgobject-2.0-0'`, run `brew install pango`.

## Document patterns

Common document structures built from atoms.

### Research report

```python
doc = [
    {"kind": "heading", "content": "Research question?", "meta": {...}},
    {"kind": "callout", "content": "The verdict in one sentence."},
    {"kind": "items", "entries": [...]},          # Key findings Q&A
    {"kind": "prose", "heading": "Summary", "content": "..."},
    {"kind": "card", "content": "Claim 1.", "badge": "supported", ...},
    {"kind": "card", "content": "Claim 2.", "badge": "challenged", ...},
    {"kind": "reference", "number": 1, "content": "...", "badge": "supports"},
    {"kind": "reference", "number": 2, "content": "...", "badge": "contradicts"},
    {"kind": "aside", "groups": {...}},           # Sidebar metadata
]
```

### Academic CV

```python
doc = [
    {"kind": "heading", "content": "Your Name", "subtitle": "Institution", "meta": "h-index: 32"},
    {"kind": "prose", "heading": "Education", "content": ""},
    {"kind": "items", "variant": "right", "entries": [
        {"label": "2006", "body": "*PhD, Field*\nUniversity"},
    ]},
    {"kind": "prose", "heading": "Publications", "content": ""},
    {"kind": "reference", "number": 1, "group": "2025", "content": "**You**, et al. Title. *Journal*."},
    {"kind": "prose", "heading": "Grants", "content": "| Grant | Year | Amount |\n|---|---|---|\n| ... |"},
    {"kind": "prose", "heading": "Teaching", "content": ""},
    {"kind": "items", "variant": "left", "entries": [
        {"label": "2024", "body": "Course name. Institution."},
    ]},
]
```

### Status update

```python
doc = [
    {"kind": "heading", "content": "Weekly Status", "meta": {"date": "2026-04-16"}},
    {"kind": "callout", "content": "Shipped semantic routing.", "tone": "success"},
    {"kind": "items", "entries": [
        {"label": "Done", "body": "Routing benchmark at 97.5%."},
        {"label": "Next", "body": "Provider query formulation."},
        {"label": "Blocked", "body": "Nothing."},
    ]},
    {"kind": "prose", "content": "## Commentary\n\nDetails here..."},
]
```

### Technical benchmark report

```python
doc = [
    {"kind": "heading", "content": "Benchmark Results", "meta": {...}},
    {"kind": "callout", "content": "97.5% top-3 recall.", "tone": "success"},
    {"kind": "prose", "heading": "Results", "content": "| Metric | Value |\n|---|---|\n| ... |"},
    {"kind": "card", "content": "Key conclusion.", "badge": "approved", "details": "Method: ..."},
    {"kind": "callout", "content": "Next steps: ...", "tone": "warning"},
    {"kind": "aside", "groups": {"Config": {"model": "...", "threshold": "0.15"}}},
]
```

## Choosing the right atom

When building a document, ask yourself:

| I want to... | Use |
|---|---|
| Set the document title and metadata | `heading` |
| Write flowing body text with headings, lists, tables | `prose` |
| Emphasize a key insight, verdict, or warning | `callout` (with optional `tone`) |
| Show a structured Q&A, key-value list, or glossary | `items` (variant `pairs`) |
| Show chronological entries with years (CV-style) | `items` (variant `right` or `left`) |
| Present a finding with status, citations, details | `card` |
| List numbered sources, references, evidence | `reference` (with optional `group`) |
| Show metadata at the bottom (sidebar grid) | `aside` (with `groups`) |
| Add a small note, colophon, or disclaimer | `aside` (with `content`) |

When in doubt, use `prose`. It handles everything markdown can do, which is most content.

## Validation and error handling

The renderer validates every atom before rendering. Errors are specific and actionable:

- **Missing required field**: `ValueError: atom[3] of kind 'heading' is missing required field 'content'`
- **Invalid tone**: `ValueError: atom[1]: callout tone must be one of ['info', 'note', 'quote', 'success', 'warning'], got 'danger'`
- **Invalid variant**: `ValueError: atom[2]: items variant must be one of ['left', 'pairs', 'right'], got 'center'`
- **Unknown kind**: falls back to `prose` with a warning (does not crash). This means you can experiment with new kind names safely.
- **Non-dict atom**: `ValueError: atom[5]: expected a dict, got str`
- **Non-list document**: `ValueError: document must be a list of atom dicts, got str`

## Showcase

Generate sample documents in all three styles (HTML + PDF):

```bash
uv run python -m andamentum.typeset.tests.showcase
```

Output is written to `/tmp/typeset_showcase/` — six files (article.html, article.pdf, cv.html, cv.pdf, report.html, report.pdf). Open them to see what each style looks like with realistic content.

## API reference

### `render(document, *, style="article", custom_css=None, title=None, footer="") -> str`

Render a document to an HTML string.

- `document`: a list of atom dicts, or a plain markdown string (treated as one `prose` atom).
- `style`: `"article"`, `"cv"`, or `"report"`.
- `custom_css`: raw CSS string; replaces the named style entirely.
- `title`: HTML `<title>` content. Auto-detected from the first heading atom if omitted.
- `footer`: text for the running page footer (used by `{footer_label}` in the CSS). Relevant mainly for PDF output.

### `render_to_file(document, output, **kwargs) -> Path`

Same as `render()`, but writes the HTML to a file and returns the path.

### `render_pdf(document, output, *, style="article", custom_css=None, title=None, footer="") -> Path`

Same as `render()`, but produces a PDF via WeasyPrint. Requires `pip install weasyprint` and system libraries (macOS: `brew install pango libffi`).

### `STYLES`

Dict mapping style names to CSS strings: `{"article": ..., "cv": ..., "report": ...}`.

### `get_style(name) -> str`

Returns the CSS string for a named style. Raises `KeyError` for unknown names.
