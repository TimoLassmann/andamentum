# andamentum.typeset

A document typesetting system with 7 visual atoms, 3 named styles, and HTML + PDF output.

## Quick start

Three ways to build a document, from easiest to most flexible:

### Report builder (recommended)

```python
from andamentum.typeset import Report

r = Report(style="article")
r.heading("My Report", meta={"date": "2026-04-16"})
r.callout("The key finding of this research.")
r.prose("## Summary\n\nThe evidence shows...")
r.save("report.html")
r.save_pdf("report.pdf", footer="Draft — April 2026")
```

Each atom type is a method. Call `.save()` or `.save_pdf()` when done. Methods return `self` so you can chain if you like.

### Builder functions

```python
from andamentum.typeset import render, heading, prose, callout

html = render([
    heading("My Report", meta={"date": "2026-04-16"}),
    callout("The key finding of this research."),
    prose("## Summary\n\nThe evidence shows..."),
])
```

Each function returns a plain dict. Compose them into a list and pass to `render()`.

### Raw dicts

```python
from andamentum.typeset import render

html = render([
    {"kind": "heading", "content": "My Report", "meta": {"date": "2026-04-16"}},
    {"kind": "callout", "content": "The key finding of this research."},
    {"kind": "prose", "content": "## Summary\n\nThe evidence shows..."},
])
```

Most flexible — best for AI agents that produce dicts programmatically.

### Plain markdown

```python
from andamentum.typeset import render

html = render("# Hello\n\nWorld.")
```

A plain string is treated as a single `prose` atom. Simplest possible usage.

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

Common document structures using the Report builder.

### Research report

```python
from andamentum.typeset import Report

r = Report(style="article")
r.heading("Research question?", meta={"date": "2026-04-15", "model": "gemma4:26b"})
r.callout("The verdict in one sentence.")
r.items(entries=[
    {"label": "What was studied?", "body": "..."},
    {"label": "What did we find?", "body": "..."},
])
r.prose("The detailed summary...", heading="Summary")
r.card("Claim 1.", badge="supported", refs=["1", "2"])
r.card("Claim 2.", badge="challenged", details="Scope: ...")
r.reference("Source 1.", number=1, source="https://...", badge="supports")
r.reference("Source 2.", number=2, source="https://...", badge="contradicts")
r.aside(groups={"Stats": {"Evidence": "37", "Claims": "2"}})
r.save("report.html")
```

### Academic CV

```python
from andamentum.typeset import Report

cv = Report(style="cv")
cv.heading("Your Name", subtitle="Institution | City", meta="Publications: 85 | h-index: 32")
cv.prose("", heading="Education")
cv.items(entries=[
    {"label": "2006", "body": "*PhD, Field*\nUniversity, Country"},
], variant="right")
cv.prose("", heading="Publications")
cv.reference("**You**, et al. Title. *Journal*.", number=1, group="2025")
cv.prose("| Grant | Year | Amount |\n|---|---|---|\n| ... |", heading="Grants")
cv.prose("", heading="Teaching")
cv.items(entries=[
    {"label": "2024", "body": "Course. Institution."},
], variant="left")
cv.save_pdf("cv.pdf", footer="April 2026")
```

### Status update

```python
from andamentum.typeset import Report

r = Report()
r.heading("Weekly Status", meta={"date": "2026-04-16"})
r.callout("Shipped semantic routing.", tone="success")
r.items(entries=[
    {"label": "Done", "body": "Routing benchmark at 97.5%."},
    {"label": "Next", "body": "Provider query formulation."},
    {"label": "Blocked", "body": "Nothing."},
])
r.prose("## Commentary\n\nDetails here...")
r.save("status.html")
```

### Technical benchmark report

```python
from andamentum.typeset import Report

r = Report(style="report")
r.heading("Benchmark Results", meta={"date": "2026-04-15", "version": "v1.0"})
r.callout("97.5% top-3 recall.", tone="success")
r.prose("| Metric | Value |\n|---|---|\n| Top-3 | 97.5% |", heading="Results")
r.card("Key conclusion.", badge="approved", details="Method: 200 queries...")
r.callout("Next steps: query formulation tuning.", tone="warning")
r.aside(groups={"Config": {"model": "embeddinggemma", "threshold": "0.15"}})
r.save("benchmark.html")
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

### `Report(style="article", **kwargs)`

Fluent document builder. Each atom type is a method:

- `r.heading(content, **kw)` — append a heading atom
- `r.prose(content, **kw)` — append a prose atom
- `r.callout(content, **kw)` — append a callout atom
- `r.items(entries, **kw)` — append an items atom
- `r.aside(**kw)` — append an aside atom
- `r.card(content, **kw)` — append a card atom
- `r.reference(content, **kw)` — append a reference atom

All methods return `self` for optional chaining. Output methods:

- `r.render() -> str` — render to HTML string
- `r.save(path) -> Path` — write HTML file
- `r.save_pdf(path, **kw) -> Path` — write PDF (requires WeasyPrint)
- `r.atoms -> list[dict]` — get the accumulated atom list (read-only copy)
- `len(r)` — number of atoms

### Builder functions

`heading(content, **kw)`, `prose(content, **kw)`, `callout(content, **kw)`, `items(entries, **kw)`, `aside(**kw)`, `card(content, **kw)`, `reference(content, **kw)`

Each returns a plain dict. Compose into a list and pass to `render()`.

### `render(document, *, style="article", custom_css=None, title=None, footer="") -> str`

Render a document (list of atom dicts, or a plain markdown string) to an HTML string.

### `render_to_file(document, output, **kwargs) -> Path`

Same as `render()`, but writes the HTML to a file and returns the path.

### `render_pdf(document, output, *, style="article", custom_css=None, title=None, footer="") -> Path`

Same as `render()`, but produces a PDF via WeasyPrint. Requires `pip install weasyprint` and system libraries (macOS: `brew install pango libffi`).

### `STYLES`

Dict mapping style names to CSS strings: `{"article": ..., "cv": ..., "report": ...}`.

### `get_style(name) -> str`

Returns the CSS string for a named style. Raises `KeyError` for unknown names.
