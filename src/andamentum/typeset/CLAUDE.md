# CLAUDE.md — andamentum.typeset

When making changes to CSS, HTML output, atom validation, or any visual surface in this module, **read `docs/design/CLAUDE.md` and `docs/design/DESIGN.md` first**. The design system in `docs/design/` is the canonical specification for what the typeset module's HTML output should look like.

## The spec/implementation gap

- `docs/design/components.css` — the canonical stylesheet (full design system: tokens, light + dark, document + app chrome).
- `src/andamentum/typeset/styles.py` — the live CSS strings (`ARTICLE`, `CV`, `REPORT`) the renderer compiles into output today.

These are **not yet in sync**. `components.css` is the destination; `styles.py` is the current state. When asked for visual changes, edit the spec first and call out the gap explicitly. Don't silently re-invent a design that already exists in `docs/design/`.

## Public API is preserved

The `.typeset-*` class family is the renderer's stable public contract. The design system extends it with `.am-*` for non-document surfaces but does NOT rename or remove existing `.typeset-*` classes. The atom dict shape (`kind`, required fields, `CALLOUT_TONES`, `ITEMS_VARIANTS`) is also stable — see `atoms.py`.

Any new atom or field needs both an `atoms.py` validation update AND a corresponding entry in `docs/design/` (component CSS, a snippet, and an addition to `showcase.html`).

## When in doubt

Follow the precedence in `docs/design/CLAUDE.md`:

1. Look at `docs/design/showcase.html` to see whether the component already exists.
2. Look at the relevant `docs/design/snippets/*.html` for canonical markup.
3. Check `docs/design/DESIGN.md` for the philosophy before inventing.
4. If still uncertain, ask before inventing.

## From the root CLAUDE.md

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

### Overview

- `andamentum.typeset` — standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules for rendering

### Design

**Typeset module** (`andamentum.typeset`) — a standalone typesetting system with 7 visual atoms (`heading`, `prose`, `callout`, `items`, `aside`, `card`, `reference`), 3 named styles (`article`, `cv`, `report`), and HTML + PDF output. Used by the epistemic audit report (`epistemic/audit_report.py`, the only HTML renderer in the epistemic pipeline — the prose-led `typeset_report.py` was retired when the audit layout reached production), by `whetstone` for its HTML review output, and by `compose.bundle` for its HTML/PDF render path. The epistemic data schema (the dataclasses consumed by the renderer) lives in `epistemic/report_data.py`. See `src/andamentum/typeset/USAGE.md` for the full API reference.

### Design language (andamentum visual system)

The repo's visual design system lives in `docs/design/`. It covers both document rendering (the `.typeset-*` family used by `andamentum.typeset`) and broader app/UI surfaces (the `.am-*` family for chrome — buttons, sidebars, chat, tool logs, settings, modals).

| File | What it is |
|---|---|
| `docs/design/components.css` | The complete stylesheet — all tokens + all components, light + dark via `data-theme`. Heavily section-commented. |
| `docs/design/DESIGN.md` | The *why*: vibe, palette logic, density rules, tone of voice, do's and don'ts. |
| `docs/design/CLAUDE.md` | Operating instructions for agents — surface→class cheat sheet, rules for new components, explicit list of things to refuse (gradients, drop shadows on cards, emoji icons, saturated semantic colours, pill buttons, etc.). |
| `docs/design/showcase.html` | Live visual reference for every component. |
| `docs/design/snippets/` | One copy-paste HTML file per component. |


**Spec vs. live implementation.** `docs/design/components.css` is the canonical specification of the visual system. `src/andamentum/typeset/styles.py` is the live CSS that the renderer compiles into output today. They are not yet in sync — the design directory is the source of truth that `styles.py` should converge toward. When making visual changes, edit the spec first; the migration from `styles.py` to `components.css` is a separate, larger piece of work.
