# `snippets/`

One small standalone HTML file per component. Each snippet:

- Is a complete HTML document — open it in a browser to see the component live.
- Forces light mode (`<html data-theme="light">`) so the canonical reading view is what you see, regardless of OS preference.
- Has a short header banner naming the component, then a clearly-marked **snippet markup** section containing the actual HTML to copy.
- Includes inline comments explaining variants, state classes, and rules of use.

## What snippets are for

Snippets are the **copy-paste API of andamentum**. They are not the source of truth — `components.css` and `showcase.html` are. Snippets exist so a human or coding agent who wants to drop a single component into a new page can grab the markup quickly without scanning the entire showcase.

When you need to build a new screen:

1. Skim `showcase.html` to see what components exist and how they're meant to look.
2. Open the relevant snippet file(s) here for clean copy-paste markup.
3. Paste into your own HTML, linking `components.css`.

When you build a new component:

- Add the component to `components.css` and `showcase.html` **first** — those are canonical.
- Then drop a snippet here as the copy-paste version, following the same wrapping pattern as the other files in this folder.

## How a snippet file is structured

```
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>…</head>
<body>

<header class="snip-header">Andamentum · snippet → name</header>

<!-- ─── snippet markup ─────────────────────────────────────────── -->
<div class="snip-stack">
    … copy everything inside this block …
</div>
<!-- ─── end snippet ────────────────────────────────────────────── -->

</body>
</html>
```

The `<header class="snip-header">` and `.snip-stack` wrapper are preview chrome, not part of the system. **Don't copy them.** The thing to copy is what sits between the `snippet markup` and `end snippet` comment markers.

## Inventory

### Document (typeset)
- **`typeset-document.html`** — full reading-view skeleton (heading, callout, items, prose, finding cards, references, sidebar)
- **`callout.html`** — default verdict block + tone variants (info / warning / success / note / quote / danger)
- **`card.html`** — app-mode `.am-card` and document-mode `.typeset-card`
- **`footnote.html`** — superscript footnote + bottom-of-doc list, plus Tufte-style margin note

### App chrome
- **`tabs.html`** — top-level tab strip
- **`sub-tabs.html`** — in-page (within-section) sub-tabs, underline-only
- **`sidebar.html`** — left rail with stats / filter / projects / areas
- **`settings.html`** — settings section with rows
- **`modal.html`** — modal / dialog with backdrop
- **`empty-state.html`** — quiet empty-state block
- **`menu.html`** — dropdown / context menu
- **`tooltip.html`** — small floating label

### Forms & controls
- **`button.html`** — primary, default, ghost, danger, sizes, icon-only
- **`input.html`** — text, textarea, select, search, labels, help text
- **`checkbox.html`** — styled checkbox + radio
- **`toggle.html`** — toggle switch for binary settings
- **`validation.html`** — error / warning / success states on form fields
- **`kbd.html`** — keyboard shortcut chips
- **`chip.html`** — filter chips
- **`tag.html`** — removable content tags + add-tag affordance
- **`badge.html`** — UI badges, document badges, status dots
- **`avatar.html`** — identity circles with initials

### Lists & rows
- **`task-row.html`** — six task statuses (inbox / to-do / in progress / waiting / done / cancelled)
- **`action-row.html`** — scheduled action rows
- **`source-row.html`** — watched-feed rows
- **`progress.html`** — ink-grey default + danger variant
- **`toast.html`** — title-tinted notifications
- **`pagination.html`** — page numbers + prev/next
- **`timeline.html`** — vertical activity feed
- **`stepper.html`** — numbered multi-step progress
