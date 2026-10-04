# andamentum.figures — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.figures` — publication-quality scientific figure generation: 10 chart types (including `stacked`), 7 journal palettes and the `andamentum` report palette, a `report` preset that matches the design system (transparent background, design-token colours, Inter, text kept as text in SVG, `inline_svg` for dropping into a page), journal-matched sizing, auto chart-kind selection, and auto bar orientation (long-categorical-label bar charts auto-flip to horizontal — overridable with `horizontal=True/False`). `bundle_glue.insert_figure` renders + inserts into a bundle section in one call. Absorbed from the standalone `mosaic-figures` package.

## Layering

- `figures` depends only on matplotlib + numpy + pydantic. The optional `figures.bundle_glue` submodule is the ONLY place where `compose.bundle` is imported; the rest of `figures` MUST NOT touch it. `figures` MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `typeset`, or `core`.

## Design

**Figures module** (`andamentum.figures`) — publication-quality scientific figure generation. High-level entry point: `figure(data, *, kind="auto", style="npg", journal="default", horizontal=None, output="figure.pdf")`. Lower-level building blocks: `setup_style`, `get_palette`, `savefig`, `panel_label`, `shared_legend`, `despine` (in `style.py`); plot primitives in `plots.py`; auto-detection (chart kind, log scale, column roles, bar orientation, x-tick rotation) in `auto.py` and `advisor.py`; bootstrap stats in `stats.py`. Bar charts with many or long categorical labels auto-flip to horizontal (`recommend_horizontal_bars` in `advisor.py`); vertical bars auto-rotate x-tick labels when crowded (`recommend_label_rotation`). Both behaviours emit `advisor_notes` so the caller knows what was changed. The package was absorbed from `mosaic-figures` (uninstall the standalone tool with `uv tool uninstall mosaic-figures` once this branch lands). Bundle integration lives in `figures/bundle_glue.py`: `insert_figure(doc, section, *, output_dir, caption, label, **chart_kwargs)` renders a PNG and inserts a `Figure` block via `Document.insert_into_section`.
