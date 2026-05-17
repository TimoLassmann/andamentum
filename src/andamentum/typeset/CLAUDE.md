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
