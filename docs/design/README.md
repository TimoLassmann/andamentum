# Andamentum

> A scholarly, letterpress-honest design system.

Andamentum styles documents — and the apps that produce them. Cream paper, serif headings, hairline rules, muted earth tones, sans-serif metadata. Dark mode is vellum-by-lamplight, not negative space.

This package is everything an engineer (human or agent) needs to apply the system to a new project or restyle an existing one.

## What's in here

| Path | What it is |
|---|---|
| **`components.css`** | The whole system in one stylesheet. Design tokens for light + dark, plus every component. Drop this on a page and you're styled. The file is heavily commented; read its header for the section index. |
| **`showcase.html`** | Live, browseable demo of every component. Toggle the theme at the top right. The canonical visual reference — when in doubt, look here. |
| **`alignment-audit.html`** | Every use of every tone, side-by-side in a single grid. Useful for verifying nothing in the system has drifted out of the palette family. |
| **`brain-mockup.html`** | A full worked example: the system applied to a knowledge-management / agent app — proof that andamentum extends past the document into chrome (sidebar, header, chat, task list, tool log). |
| **`DESIGN.md`** | The *why*. Philosophy, do's and don'ts, tone of voice, density rules. Read this before designing anything new. |
| **`CLAUDE.md`** | Operating instructions for coding agents (Claude Code and friends). Drop a copy at the root of any project that adopts andamentum so the rules persist between sessions. |
| **`README.md`** | This file. Quick start + map. |
| **`snippets/`** | One small standalone HTML file per component. Each snippet has the canonical markup and inline comments. Copy-paste ready. See [`snippets/README.md`](./snippets/README.md). |
| **`index.html`** | Front door — links to all of the above for human browsing. |

## Quick start

```html
<link rel="stylesheet" href="andamentum/components.css">
```

That's it. Use `.typeset-*` classes for document content (the original API is preserved exactly), and `.am-*` classes for app chrome.

Force light or dark with `<html data-theme="light">` / `data-theme="dark"`. Omit the attribute to honour the OS preference.

## The two namespaces

- **`.typeset-*`** — document content. Reading view, assistant prose, exports. Anything a person reads to learn.
- **`.am-*`** — andamentum's full app vocabulary. Buttons, inputs, lists, modals, chat, tool log, task rows, settings, everything else. Anything the system says about itself.

They share tokens. They look like they belong together. They never compete for the same selector.

## Tokens

Every paint, every font size, every gap reads from a CSS custom property. The full list is in `components.css` § 2; the headlines:

```css
/* Surface */
--am-paper        /* primary surface — warm cream / dark vellum */
--am-paper-2..4   /* recessed surfaces (asides, code, hairline strong) */

/* Text */
--am-ink          /* darkest text — titles */
--am-ink-2..7     /* six more steps, body → muted */

/* Rules & links */
--am-rule         /* primary hairline */
--am-link         /* the brown of an underlined word in an old book */

/* Semantic accents — earth-toned on purpose */
--am-info-accent      /* slate  #6e7a8c */
--am-warn-accent      /* ochre  #9a8559 */
--am-success-accent   /* olive  #5e7a6a */
--am-danger-accent    /* rust   #9c4a36 */
--am-note-accent      /* neutral cream-grey */
/* …each has matching --am-{tone}-bg and --am-{tone}-ink */

/* Scale */
--am-sp-1..10     /* spacing — 4 → 64 */
--am-rad-1..4     /* radii — andamentum prefers small */
--am-fs-*         /* type scale */

/* Typefaces */
--am-font-serif   /* Source Serif 4 — body */
--am-font-sans    /* Inter — UI / metadata */
--am-font-mono    /* SF Mono — code */
```

**Edit a token to reskin. Never edit a component rule to change palette.**

## Dark mode

Dark is a first-class citizen, not a negative photo. The page is dark warm — never `#000`. Inks become warm bone-white. The link becomes warm tan, not blue. Semantic accents shift to lighter equivalents (slate → light slate, rust → tan-rust) but stay in the same earth family. The aesthetic survives the switch.

Toggle with the `data-theme` attribute. The showcase, alignment audit, worked-example mockups, and front-door index all have a toggle.

## Applying it elsewhere

See **`CLAUDE.md`**. Short version:

1. Copy the `andamentum/` folder into your project (or symlink it, or git-submodule it from a shared repo).
2. Add `<link rel="stylesheet" href="andamentum/components.css">` to your pages.
3. Drop a `CLAUDE.md` at your project root that points coding agents at the system.
4. Swap your selectors over to the `.am-*` / `.typeset-*` families.
5. For anything new, lean on tokens; trust hairlines over shadows; choose quiet over loud.

## A note on the name

*Andamento* is an art-historical term for the rhythm and flow of an inlaid mosaic — the line the tesserae follow across a field. The system aims at that quality: each row, card, paragraph laid down with deliberate flow, never one piece louder than the line it sits in.
