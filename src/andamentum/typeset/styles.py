"""Built-in CSS stylesheets for andamentum.typeset.

Three styles are provided:
- **article** — the canonical andamentum design system, read verbatim from
  ``assets/components.css`` (the packaged copy of ``docs/design/components.css``).
- **cv** — monochrome compact (Inter + JetBrains Mono), inline below.
- **report** — blue technical (Inter + Space Grotesk + JetBrains Mono), inline below.

The article style is the full andamentum design system: cream paper, serif body,
hairline rules, brown links, earth-tone semantic accents, dark-mode aware via
``<html data-theme="dark">``. It also covers footnotes, margin notes, and the
full ``.am-*`` app-chrome vocabulary — consumers who only need document
rendering can ignore those rules.

When the design system at ``docs/design/components.css`` is updated, copy the
new file into ``src/andamentum/typeset/assets/components.css``. The sync is
enforced by ``tests/test_assets.py``, which fails loudly if the two files drift.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# ARTICLE — the canonical andamentum design system
# ---------------------------------------------------------------------------
#
# Loaded from the packaged components.css. See module docstring for the
# sync convention with docs/design/components.css.

_ASSETS_DIR = Path(__file__).parent / "assets"

ARTICLE: str = (_ASSETS_DIR / "components.css").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CV — monochrome, compact, Inter throughout
# ---------------------------------------------------------------------------

CV: str = """\
@import url('https://fonts.googleapis.com/css2?\
family=Inter:wght@300;400;500;600;700\
&family=JetBrains+Mono:wght@400;500\
&display=swap');

* {
    box-sizing: border-box;
}

/* ── Document wrapper ───────────────────────────────────────────────── */

.typeset-document {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 9pt;
    line-height: 1.45;
    color: #333;
    max-width: 100%;
    padding: 0;
}

/* ── Heading block ──────────────────────────────────────────────────── */

.typeset-heading {
    text-align: center;
    margin-bottom: 0.8em;
    padding-bottom: 0;
    border-bottom: none;
}

.typeset-heading h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0 0 2px;
    letter-spacing: -0.5px;
    text-align: center;
    border: none;
}

.typeset-subtitle {
    text-align: center;
    color: #555;
    font-size: 9pt;
    margin: 0 0 0.4em;
}

.typeset-meta {
    font-size: 9pt;
    color: #555;
    margin: 0 0 0.4em;
    text-align: center;
}

/* ── Prose (markdown body) ──────────────────────────────────────────── */

.typeset-prose {
    margin: 0 0 0.5em;
}

.typeset-prose h2 {
    font-size: 11pt;
    font-weight: 700;
    color: #1a1a1a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 3px;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}

.typeset-prose h3 {
    font-size: 9.5pt;
    font-weight: 600;
    color: #1a1a1a;
    margin-top: 0.6em;
    margin-bottom: 0.15em;
    page-break-after: avoid;
}

.typeset-prose p {
    margin: 0.25em 0;
    orphans: 3;
    widows: 3;
}

.typeset-prose ul,
.typeset-prose ol {
    margin: 0.2em 0;
    padding-left: 1.4em;
}

.typeset-prose li {
    margin-bottom: 0.1em;
}

.typeset-prose table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.3em 0;
    font-size: 8.5pt;
    border-top: 1.5px solid #1a1a1a;
    border-bottom: 1.5px solid #1a1a1a;
    page-break-inside: auto;
}

.typeset-prose thead {
    display: table-header-group;
}

.typeset-prose th {
    background: none;
    color: #333;
    font-weight: 600;
    text-align: left;
    padding: 4px 6px;
    border-bottom: 1px solid #1a1a1a;
}

.typeset-prose td {
    padding: 3px 6px;
    border: none;
    vertical-align: top;
}

.typeset-prose tr {
    page-break-inside: avoid;
}

.typeset-prose code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8pt;
    background-color: #f5f5f5;
    padding: 1px 3px;
    border-radius: 2px;
    color: #333;
}

.typeset-prose pre {
    background-color: #f8f8f8;
    padding: 10px 14px;
    border-radius: 4px;
    border: 1px solid #e8e8e8;
    font-size: 8pt;
    line-height: 1.45;
    page-break-inside: avoid;
}

.typeset-prose pre code {
    background: none;
    padding: 0;
    border-radius: 0;
}

.typeset-prose blockquote {
    border-left: 3px solid #ddd;
    margin: 0.4em 0;
    padding: 0.2em 0.8em;
    color: #555;
    font-style: italic;
}

.typeset-prose a {
    color: #0066cc;
    text-decoration: none;
}

.typeset-prose strong {
    font-weight: 600;
}

.typeset-prose em {
    font-style: italic;
    color: #555;
}

.typeset-prose hr {
    display: none;
}

.typeset-prose img {
    max-width: 100%;
    height: auto;
}

/* ── Callout ────────────────────────────────────────────────────────── */

.typeset-callout {
    margin: 0.5em 0;
    padding: 8px 12px;
    background: #f5f5f5;
    border-left: 3px solid #ddd;
    font-size: 9pt;
    line-height: 1.45;
    color: #333;
}

.typeset-callout.tone-info {
    border-left-color: #999;
}

.typeset-callout.tone-warning {
    border-left-color: #666;
}

.typeset-callout.tone-success {
    border-left-color: #999;
}

.typeset-callout.tone-note {
    border-left-color: #bbb;
}

.typeset-callout.tone-quote {
    border-left-color: #ddd;
    font-style: italic;
    color: #555;
}

/* ── Items ──────────────────────────────────────────────────────────── */

.typeset-items {
    margin: 0 0 0.5em;
    font-size: 9pt;
    line-height: 1.4;
}

.typeset-item {
    margin-bottom: 4pt;
    page-break-inside: avoid;
}

.typeset-item:last-child {
    margin-bottom: 0;
}

/* variant-pairs: label above, value below — tight */
.typeset-items.variant-pairs .typeset-item-label {
    font-weight: 600;
    font-size: 8pt;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 1px;
}

.typeset-items.variant-pairs .typeset-item-body {
    color: #333;
}

/* variant-right: body left, label right-aligned (CV entry layout) */
.item-right {
    display: flex;
}

.item-right .typeset-item-body {
    flex: 1;
    color: #333;
    padding-left: 12px;
}

.item-right .typeset-item-label {
    width: 90px;
    text-align: right;
    flex-shrink: 0;
    color: #555;
    font-size: 9pt;
    padding-top: 1px;
}

/* variant-left: label left, body right (year-left layout) */
.item-left {
    display: flex;
}

.item-left .typeset-item-label {
    width: 55px;
    flex-shrink: 0;
    font-size: 9pt;
    color: #555;
}

.item-left .typeset-item-body {
    flex: 1;
    font-size: 9pt;
    color: #333;
    padding-left: 10px;
}

/* ── Aside (content mode) ───────────────────────────────────────────── */

.typeset-aside {
    margin: 0.4em 0;
    padding: 6px 12px;
    background: #f8f8f8;
    border-left: 2px solid #ddd;
    font-size: 9pt;
    line-height: 1.4;
    color: #555;
}

/* ── Sidebar (groups / grid mode) ───────────────────────────────────── */

.typeset-sidebar {
    margin-top: 1em;
    padding-top: 8px;
    border-top: 1px solid #ddd;
    font-size: 8.5pt;
    line-height: 1.4;
    color: #555;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
}

.typeset-sidebar-group {
    margin-bottom: 0;
}

.typeset-sidebar-title {
    font-size: 7pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #999;
    margin: 0 0 3px;
}

.typeset-sidebar-row {
    display: flex;
    justify-content: space-between;
    padding: 1px 0;
    font-size: 8.5pt;
}

.typeset-sidebar-label {
    color: #999;
}

.typeset-sidebar-value {
    font-weight: 500;
    color: #555;
}

/* ── Card ───────────────────────────────────────────────────────────── */

.typeset-card {
    margin: 4pt 0;
    padding: 0 0 4pt;
    border-bottom: 1px solid #eee;
    page-break-inside: avoid;
}

.typeset-card:last-child {
    border-bottom: none;
}

.typeset-card-body {
    line-height: 1.4;
    color: #333;
}

.typeset-badge {
    display: inline-block;
    font-size: 7pt;
    color: #888;
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 500;
    vertical-align: middle;
}

.typeset-refs {
    font-size: 8pt;
    color: #999;
}

.typeset-refs a {
    color: #999;
    text-decoration: none;
}

.typeset-card-source {
    font-size: 8pt;
    color: #888;
    word-break: break-all;
    margin-top: 2px;
}

.typeset-card-source a {
    color: #888;
    text-decoration: none;
}

.typeset-card-details {
    margin-top: 4px;
}

.typeset-card-details summary {
    font-size: 8pt;
    color: #999;
    cursor: pointer;
}

.typeset-card-details summary:hover {
    color: #333;
}

/* ── Reference ──────────────────────────────────────────────────────── */

.typeset-reference {
    display: flex;
    margin-bottom: 7pt;
    font-size: 9pt;
    line-height: 1.35;
    page-break-inside: avoid;
}

.typeset-reference:last-child {
    margin-bottom: 0;
}

.typeset-ref-number {
    width: 22px;
    flex-shrink: 0;
    color: #999;
    font-size: 8pt;
    text-align: right;
    padding-right: 6px;
    padding-top: 0.5px;
}

.typeset-ref-content {
    flex: 1;
}

.typeset-ref-body {
    color: #333;
    font-size: 9pt;
    line-height: 1.35;
}

.typeset-ref-source {
    display: block;
    margin-top: 1pt;
    font-size: 8.5pt;
    color: #555;
}

.typeset-ref-source a {
    color: #555;
    text-decoration: none;
}

/* ── Reference group ────────────────────────────────────────────────── */

.typeset-reference-group {
    margin-bottom: 5pt;
    page-break-inside: auto;
}

.typeset-ref-group-label {
    font-weight: 700;
    color: #1a1a1a;
    font-size: 9pt;
    margin-bottom: 2pt;
    margin-top: 8pt;
}

/* ── Print / PDF ────────────────────────────────────────────────────── */

@page {
    size: A4;
    margin: 20mm 20mm 18mm 20mm;
    @bottom-left {
        content: "{footer_label}";
        font-family: 'Inter', sans-serif;
        font-size: 7pt;
        color: #999;
    }
    @bottom-right {
        content: "Page " counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 7pt;
        color: #999;
    }
}

@page :first {
    @bottom-left { content: none; }
    @bottom-right { content: none; }
}

h1, h2, h3 { page-break-after: avoid; }
.typeset-reference,
.typeset-card,
.typeset-item { page-break-inside: avoid; }
p { orphans: 3; widows: 3; }
"""


# ---------------------------------------------------------------------------
# REPORT — blue technical, Inter + Space Grotesk
# ---------------------------------------------------------------------------

REPORT: str = """\
@import url('https://fonts.googleapis.com/css2?\
family=Inter:wght@300;400;500;600;700\
&family=Space+Grotesk:wght@400;500;600;700\
&family=JetBrains+Mono:wght@400;500\
&display=swap');

* {
    box-sizing: border-box;
}

/* ── Document wrapper ───────────────────────────────────────────────── */

.typeset-document {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 9.5pt;
    line-height: 1.55;
    color: #212529;
    max-width: 100%;
    padding: 0;
}

/* ── Heading block ──────────────────────────────────────────────────── */

.typeset-heading {
    margin-bottom: 1.5em;
    padding-bottom: 8px;
    border-bottom: 3px solid #0d6efd;
}

.typeset-heading h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20pt;
    font-weight: 700;
    color: #0a58ca;
    margin: 0 0 6px;
    line-height: 1.25;
}

.typeset-subtitle {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12pt;
    color: #495057;
    margin: 0 0 4px;
    font-weight: 400;
}

.typeset-meta {
    font-size: 9pt;
    color: #6c757d;
    margin: 4px 0 0;
}

/* ── Prose (markdown body) ──────────────────────────────────────────── */

.typeset-prose {
    margin: 0 0 1.5em;
}

.typeset-prose h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14pt;
    font-weight: 600;
    color: #0d6efd;
    border-bottom: 1.5px solid #dee2e6;
    padding-bottom: 4px;
    margin-top: 1.8em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
}

.typeset-prose h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11.5pt;
    font-weight: 600;
    color: #495057;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}

.typeset-prose p {
    margin: 0.5em 0;
    orphans: 3;
    widows: 3;
}

.typeset-prose ul,
.typeset-prose ol {
    margin: 0.4em 0;
    padding-left: 1.6em;
}

.typeset-prose li {
    margin-bottom: 0.15em;
}

.typeset-prose table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.8em 0;
    font-size: 8.5pt;
    page-break-inside: auto;
}

.typeset-prose thead {
    display: table-header-group;
}

.typeset-prose th {
    background-color: #0d6efd;
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 8.5pt;
    text-align: left;
    padding: 6px 8px;
    border: 1px solid #0b5ed7;
}

.typeset-prose td {
    padding: 5px 8px;
    border: 1px solid #dee2e6;
    vertical-align: top;
}

.typeset-prose tr {
    page-break-inside: avoid;
}

.typeset-prose tbody tr:nth-child(even) {
    background-color: #f8f9fa;
}

.typeset-prose code {
    font-family: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace;
    font-size: 8pt;
    background-color: #f1f3f5;
    color: #d63384;
    padding: 1px 4px;
    border-radius: 3px;
    border: 0.5px solid #dee2e6;
}

.typeset-prose pre {
    background-color: #282c34;
    color: #abb2bf;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 8pt;
    line-height: 1.5;
    overflow-x: auto;
    margin: 0.8em 0;
    page-break-inside: avoid;
}

.typeset-prose pre code {
    background: none;
    color: inherit;
    padding: 0;
    border: none;
    border-radius: 0;
    font-size: inherit;
}

.typeset-prose blockquote {
    border-left: 4px solid #0d6efd;
    margin: 0.8em 0;
    padding: 0.4em 1em;
    background-color: #f8f9fa;
    color: #495057;
}

.typeset-prose a {
    color: #0d6efd;
    text-decoration: none;
}

.typeset-prose a:hover {
    text-decoration: underline;
}

.typeset-prose strong {
    font-weight: 600;
}

.typeset-prose em {
    font-style: italic;
}

.typeset-prose hr {
    border: none;
    border-top: 1.5px solid #dee2e6;
    margin: 1.5em 0;
}

.typeset-prose img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

/* ── Callout ────────────────────────────────────────────────────────── */

.typeset-callout {
    margin: 1em 0;
    padding: 12px 16px;
    background: #e7f1ff;
    border-left: 4px solid #0d6efd;
    border-radius: 0 6px 6px 0;
    font-size: 9pt;
    line-height: 1.55;
    color: #212529;
}

.typeset-callout.tone-info {
    background: #e7f1ff;
    border-left-color: #0d6efd;
}

.typeset-callout.tone-warning {
    background: #fff8e1;
    border-left-color: #ffc107;
}

.typeset-callout.tone-success {
    background: #e8f5e9;
    border-left-color: #28a745;
}

.typeset-callout.tone-note {
    background: #f8f9fa;
    border-left-color: #6c757d;
}

.typeset-callout.tone-quote {
    background: #f8f9fa;
    border-left-color: #adb5bd;
    font-style: italic;
    color: #495057;
}

/* ── Items ──────────────────────────────────────────────────────────── */

.typeset-items {
    margin: 0.8em 0;
    padding: 12px 16px;
    background: #f8f9fa;
    border-radius: 6px;
    font-size: 9pt;
    line-height: 1.55;
}

.typeset-item {
    margin-bottom: 8px;
}

.typeset-item:last-child {
    margin-bottom: 0;
}

/* variant-pairs: label above, value below */
.typeset-items.variant-pairs .typeset-item-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 8.5pt;
    color: #0d6efd;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 2px;
}

.typeset-items.variant-pairs .typeset-item-body {
    color: #212529;
}

/* variant-right: body left, label right */
.item-right {
    display: flex;
    align-items: baseline;
}

.item-right .typeset-item-body {
    flex: 1;
    color: #212529;
}

.item-right .typeset-item-label {
    width: 100px;
    text-align: right;
    flex-shrink: 0;
    color: #6c757d;
    font-size: 8.5pt;
}

/* variant-left: label left, body right */
.item-left {
    display: flex;
    align-items: baseline;
}

.item-left .typeset-item-label {
    width: 80px;
    flex-shrink: 0;
    font-size: 8.5pt;
    color: #6c757d;
}

.item-left .typeset-item-body {
    flex: 1;
    color: #212529;
}

/* ── Aside (content mode) ───────────────────────────────────────────── */

.typeset-aside {
    margin: 0.8em 0;
    padding: 10px 14px;
    background: #f8f9fa;
    border-left: 3px solid #0d6efd;
    border-radius: 0 4px 4px 0;
    font-size: 9pt;
    line-height: 1.5;
    color: #495057;
}

/* ── Sidebar (groups / grid mode) ───────────────────────────────────── */

.typeset-sidebar {
    margin-top: 2em;
    padding-top: 12px;
    border-top: 1.5px solid #dee2e6;
    font-size: 8.5pt;
    line-height: 1.45;
    color: #6c757d;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
}

.typeset-sidebar-group {
    margin-bottom: 0;
}

.typeset-sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 7.5pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #0d6efd;
    margin: 0 0 4px;
}

.typeset-sidebar-row {
    display: flex;
    justify-content: space-between;
    padding: 1px 0;
    font-size: 8.5pt;
}

.typeset-sidebar-label {
    color: #6c757d;
}

.typeset-sidebar-value {
    font-weight: 500;
    color: #212529;
}

/* ── Card ───────────────────────────────────────────────────────────── */

.typeset-card {
    margin: 10px 0;
    padding: 12px 16px;
    border: 1px solid #cfe2ff;
    border-left: 4px solid #0d6efd;
    border-radius: 0 6px 6px 0;
    background: #fff;
}

.typeset-card:last-child {
    margin-bottom: 0;
}

.typeset-card-body {
    line-height: 1.55;
    color: #212529;
}

.typeset-badge {
    display: inline-block;
    font-size: 7.5pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding: 2px 8px;
    border-radius: 10px;
    background: #e7f1ff;
    color: #0d6efd;
}

.typeset-refs {
    font-size: 8pt;
    color: #6c757d;
}

.typeset-refs a {
    color: #0d6efd;
    text-decoration: none;
}

.typeset-refs a:hover {
    text-decoration: underline;
}

.typeset-card-source {
    font-size: 8pt;
    color: #6c757d;
    word-break: break-all;
    margin-top: 4px;
}

.typeset-card-source a {
    color: #0d6efd;
    text-decoration: none;
}

.typeset-card-details {
    margin-top: 6px;
}

.typeset-card-details summary {
    font-size: 8.5pt;
    color: #6c757d;
    cursor: pointer;
}

.typeset-card-details summary:hover {
    color: #0d6efd;
}

.typeset-card-details[open] summary {
    margin-bottom: 6px;
}

/* ── Reference ──────────────────────────────────────────────────────── */

.typeset-reference {
    display: flex;
    margin: 8px 0;
    padding: 8px 12px;
    background: #f8f9fa;
    border-radius: 4px;
    font-size: 9pt;
    line-height: 1.5;
}

.typeset-reference:last-child {
    margin-bottom: 0;
}

.typeset-ref-number {
    width: 28px;
    flex-shrink: 0;
    font-size: 8.5pt;
    font-weight: 600;
    color: #0d6efd;
    text-align: right;
    padding-right: 8px;
}

.typeset-ref-content {
    flex: 1;
}

.typeset-ref-body {
    color: #212529;
    font-size: 9pt;
    line-height: 1.5;
}

.typeset-ref-source {
    font-size: 8pt;
    color: #6c757d;
    margin-top: 2px;
    word-break: break-all;
}

.typeset-ref-source a {
    color: #0d6efd;
    text-decoration: none;
}

.typeset-ref-source a:hover {
    text-decoration: underline;
}

/* ── Reference group ────────────────────────────────────────────────── */

.typeset-reference-group {
    margin: 12px 0;
}

.typeset-ref-group-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 9pt;
    font-weight: 600;
    color: #0d6efd;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

/* ── Print ──────────────────────────────────────────────────────────── */

@page {
    size: A4;
    margin: 22mm 20mm;
}

@media print {
    .typeset-document { font-size: 9.5pt; }
    h1, h2, h3 { page-break-after: avoid; }
    .typeset-reference,
    .typeset-card,
    .typeset-item { page-break-inside: avoid; }
    p { orphans: 3; widows: 3; }
}
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STYLES: dict[str, str] = {
    "article": ARTICLE,
    "cv": CV,
    "report": REPORT,
}


def get_style(name: str) -> str:
    """Return the CSS string for *name*.

    Parameters
    ----------
    name:
        Style name (one of ``"article"``, ``"cv"``, ``"report"``).

    Returns
    -------
    str
        CSS string.

    Raises
    ------
    KeyError
        If *name* is not a known style.  The error message includes the list
        of available names.
    """
    if name not in STYLES:
        available = ", ".join(sorted(STYLES))
        raise KeyError(f"Unknown style {name!r}. Available styles: {available}.")
    return STYLES[name]
