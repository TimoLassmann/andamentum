# CLAUDE.md — Andamentum

> Operating instructions for coding agents (Claude Code and friends) working in a project that uses the andamentum design system.

This file should live at the **root of any project that adopts andamentum**, alongside (or symlinked next to) the `andamentum/` folder. If you're an agent reading this, the rules below apply to every change you make in this project.

---

## TL;DR

1. **Read `andamentum/DESIGN.md` before designing anything new.** It's the source of truth for *what andamentum is and why*.
2. **Look at `andamentum/showcase.html`** to see what components exist and how they look.
3. **Link `andamentum/components.css`** on any HTML/JSX surface you touch.
4. **Don't invent CSS for things the system already has.** Look for an existing `.am-*` or `.typeset-*` class first.
5. **Use tokens (CSS custom properties) for any new colour, size, or spacing.** Never write a raw hex code or px value if there's a `--am-*` token for it.
6. **Two registers**: `.typeset-*` for document content (anything a person reads to learn), `.am-*` for app chrome (anything UI).
7. **Quiet by default.** When choosing between louder and quieter, choose quieter. Earth-tone accents, hairline rules, no shadows, no gradients.

---

## The shape of the package

Inside `andamentum/`:

| File | What it does | When you read it |
|---|---|---|
| `components.css` | All tokens + all component CSS. Heavily commented. | When you want to know what a class does or what's tokenisable. |
| `DESIGN.md` | The philosophy: vibe, palette logic, density, tone of voice, do's and don'ts. | First, when starting any UI work. |
| `showcase.html` | Live demo of every component, light + dark toggle. | When you need to *see* a component before building with it. |
| `alignment-audit.html` | Every use of every tone in one grid. | When verifying that nothing has drifted out of the palette family. |
| `brain-mockup.html` | A full worked example: the system applied to a knowledge-management / agent app (header, sidebar, task list, chat, tool log). | When you need a worked example of app chrome. |
| `snippets/` | Standalone copy-paste markup per component. Each is a viewable HTML page. | When you want to grab a component's HTML quickly. |
| `index.html` | Front-door page for human browsing. | When orienting a new contributor. |

---

## How to apply andamentum to a page

### Document-style page (article, report, reading view, generated document)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="andamentum/components.css">
</head>
<body>
    <article class="typeset-document">
        <header class="typeset-heading">
            <h1>Title</h1>
            <p class="typeset-subtitle">Optional one-line subtitle</p>
            <p class="typeset-meta">2026-05-15 · model · run_id</p>
        </header>

        <aside class="typeset-callout">
            <p>One-sentence verdict the reader sees first.</p>
        </aside>

        <section class="typeset-prose">
            <h2>A section</h2>
            <p>Body prose…</p>
        </section>
    </article>
</body>
</html>
```

The full taxonomy of `.typeset-*` classes is in `showcase.html` under § 03 — Document, and the canonical markup is in `snippets/typeset-document.html`.

### App-style page (lists, forms, tools, dashboards)

```html
<body class="am-app">
    <header class="am-app-header">
        <div class="am-app-title">…</div>
        <nav class="am-tabs">…</nav>
    </header>

    <div style="display: grid; grid-template-columns: 240px 1fr;">
        <aside class="am-sidebar">…</aside>
        <main>…</main>
    </div>
</body>
```

App-style classes are in showcase.html under §§ 04–12. `brain-mockup.html` and `browse-mockup.html` are the canonical worked examples — the system applied to a real productivity / agent app.

### Mixed (chat with the assistant replying in prose)

User messages are bubbles (`.am-chat__msg--user`). Assistant messages are full-width typeset prose (`.am-chat__msg--assistant`) and inherit `.typeset-prose` styling. Tool calls collapse behind `.am-toollog`. Documents the agent produced show as `.am-artefacts`. See `snippets/chat.html` and `snippets/toollog.html`.

---

## Surface → class cheat sheet

If you're restyling an existing app, use this mapping. Andamentum has been built specifically to cover every surface listed below.

### Document content

| Surface | Andamentum class |
|---|---|
| Article / report / reading view | `.typeset-document` + `.typeset-*` family |
| Title block | `.typeset-heading` + `<h1>`, `.typeset-subtitle`, `.typeset-meta` |
| Opening verdict (no tinted box) | `.typeset-callout` |
| Tinted callouts | `.typeset-callout.tone-info / -warning / -success / -note / -danger / -quote` |
| Body prose | `.typeset-prose` (wraps headings, paragraphs, tables, lists, code, blockquotes, links) |
| Label/value items | `.typeset-items.variant-pairs / .item-right / .item-left` |
| Finding / claim card | `.typeset-card` + `.typeset-badge[data-value="supports|contradicts|…"]` |
| Numbered reference | `.typeset-reference` + `.typeset-ref-number` / `.typeset-ref-content` |
| Footnote-grid sidebar | `.typeset-sidebar` (use as `<aside class="typeset-aside typeset-sidebar">`) |

### Chat & instrumentation

| Surface | Andamentum class |
|---|---|
| User chat bubble | `.am-chat__msg--user` |
| Assistant chat reply (typeset prose) | `.am-chat__msg--assistant` |
| Composer (chat input row) | `.am-composer`, `.am-composer__row`, `.am-composer__input` |
| Tool-log archive | `<details class="am-toollog">` + `.am-toollog__summary`, `.am-toollog__body`, `.am-toollog__entry` |
| Router decision line | `.am-router`, `.am-router__label`, `.am-router__reason` |
| Artefacts container ("N documents created") | `.am-artefacts`, `.am-artefacts__header`, `.am-artefacts__list`, `.am-artefacts__row` |
| Single inline artefact link | `.am-artefact` |
| Thinking indicator | `.am-thinking` + `.am-thinking__dots` |

### App chrome

| Surface | Andamentum class |
|---|---|
| Top app header | `.am-app-header`, `.am-app-title`, `.am-app-header__right` |
| Top-level tabs | `.am-tabs`, `.am-tab` (+ `.is-active`) |
| Left sidebar / rail | `.am-sidebar`, `.am-sidebar__section`, `.am-sidebar__title`, `.am-sidebar__item` (+ `.is-active`) |
| Stats tiles | `.am-stats`, `.am-stat`, `.am-stat__value` (+ `.is-danger / .is-success`), `.am-stat__label` |
| Project / area nav rows | `.am-nav-row`, `.am-nav-row__title`, `.am-nav-row__meta`, `.am-nav-row__progress` |

### Controls

| Surface | Andamentum class |
|---|---|
| Buttons | `.am-btn`, `.am-btn--primary / --ghost / --danger`, `.am-btn--sm / --lg / --icon` |
| Inputs | `.am-input`, `.am-textarea`, `.am-select` |
| Search input (with leading icon) | `.am-search` |
| Checkbox | `.am-checkbox` (wrap with `.am-checkbox-row` label) |
| Radio | `.am-radio` (same wrapping pattern as checkbox) |
| Toggle switch | `.am-toggle` (same wrapping pattern; binary settings only) |
| Form validation | `.am-input.is-error / .is-warn`, `.am-help--error / --warn / --success` |
| Labels / help text | `.am-label`, `.am-help` |
| Filter chips | `.am-chip` (+ `.is-active`) |

### Classification

| Surface | Andamentum class |
|---|---|
| UI badge | `.am-badge`, `.am-badge--info / --warn / --success / --danger / --note / --outline` |
| Status dot | `.am-dot`, `.am-dot--info / --warn / --success / --danger` |
| Tag (removable, on content) | `.am-tag` (+ `.am-tag__close`), `.am-tag--info / --warn / --success / --danger` |
| Tag-add (`+` button) | `.am-tag-add` |
| Tag row wrapper | `.am-tags` |

### Lists & cards

| Surface | Andamentum class |
|---|---|
| App-mode card | `.am-card`, `.am-card--quiet` |
| Document-mode card (inside prose) | `.typeset-card` |
| Generic list | `.am-list` (+ `.am-list--bordered`), `.am-row` |
| Task row | `.am-task` + `.am-task__check` |
| Action row (scheduled action) | `.am-action-row` |
| Source row (feed / page) | `.am-source-row` |

### Task status icons (`.am-task__check.is-*`)

Six states. Each is a 18px circle in andamentum's palette. Apply the matching `.is-*` to **both** the check element AND the parent `.am-task` when you want title styling (strike-through for done/cancelled).

| State | Check class | Visual | Use for |
|---|---|---|---|
| Inbox | `.is-inbox` | Dashed ink-grey outline | Newly captured, not yet committed |
| To Do | `.is-todo` *(or no modifier)* | Solid ink-grey outline | Open, ready to be picked up |
| In Progress | `.is-progress` | Slate half-fill | Being worked on right now |
| Waiting | `.is-waiting` | Ochre outline | Awaiting input or response |
| Done | `.is-done` | Solid olive fill | Completed |
| Cancelled | `.is-cancelled` | Rust outline + tiny × | Abandoned or no longer relevant |

### Date pills (inside `.am-task__sub`)

| Class | Use |
|---|---|
| `.am-task__due` | Default — within normal range |
| `.am-task__due.is-soon` | Within ~3 days (warn / ochre) |
| `.am-task__due.is-overdue` | Past due (danger / rust) |

### Overlays & feedback

| Surface | Andamentum class |
|---|---|
| Modal / dialog | `.am-modal-backdrop`, `.am-modal`, `.am-modal__header / __title / __close / __body / __footer` |
| Toast | `.am-toast`, `.am-toast--info / --success / --warn / --danger` (title-tinted, no bar) |
| Tooltip | `.am-tooltip` (default dark) / `.am-tooltip--paper` |
| Dropdown / context menu | `.am-menu`, `.am-menu__group`, `.am-menu__item` (+ `.is-danger`), `.am-menu__item__icon`, `.am-menu__item__shortcut`, `.am-menu__label` |
| Empty state | `.am-empty`, `.am-empty__mark / __title / __hint` |
| Progress bar | `.am-progress`, `.am-progress--thin`, `.am-progress--danger` |

> **Note**: progress only has `--thin` and `--danger`. There is no `--info / --warn / --success` variant — by design. Andamentum's default progress is a quiet ink-grey; the only progress that takes colour is one that represents something genuinely failing.

### Settings & forms

| Surface | Andamentum class |
|---|---|
| Settings section | `.am-settings-section`, `.am-settings-section__title / __intro` |
| Settings row (label + control) | `.am-settings-row`, `.am-settings-row__label / __help` |

---

## Rules for new components

You'll occasionally need a component andamentum doesn't have. Follow these rules:

1. **Don't invent a fourth font.** Source Serif 4, Inter, SF Mono. That's it.
2. **Don't invent new colour values.** If you need a colour, it's already in `--am-*`. If you think it isn't, you're probably reaching for a colour that doesn't belong in andamentum.
3. **Use the spacing scale.** `var(--am-sp-N)`, not raw pixels.
4. **Use the radius scale.** `var(--am-rad-N)`, not 5px or 7px.
5. **Hairlines, not shadows.** Default to `border: 1px solid var(--am-rule);` for separation. Box-shadows are forbidden outside the three places they already appear (modal, toast, ground-line on cards).
6. **No gradients.** Ever.
7. **No emoji in UI copy or as icons.** Use stroke SVGs at 1.5 stroke-width.
8. **Naming**: `.am-{noun}` block, `.am-{noun}__{part}` element, `.am-{noun}--{variant}` modifier, `.is-{state}` for state classes.
9. **Add the new component to `showcase.html`** under the relevant section, with a label and short description.
10. **Drop a copy-pasteable snippet in `snippets/`**, wrapped like the others (light-mode HTML doc with the begin/end snippet markers).

---

## Things to refuse

Even if asked, **do not** (without pushing back first):

- Add a gradient to anything in andamentum.
- Increase a semantic colour's saturation to "make it pop".
- Add a drop-shadow to a card or row.
- Add emoji to interface copy.
- Add a pill-shaped button.
- Add a fourth font.
- Add an animation longer than 200ms.
- Add a "hero" decoration to a document page.
- Use exclamation marks in system-authored copy.
- Bring back the `--info / --warn / --success` progress variants. They were removed deliberately.
- Add a coloured left bar to toasts. They were removed deliberately — tone lives in the title only.

If the user asks for one of these, push back once with reasoning rooted in `DESIGN.md`. If they confirm, satisfy the request — but isolate it as an explicit override rather than naturalising it into the system.

---

## Tone of voice (when writing UI copy)

- Short, factual, dry.
- No greetings ("Welcome back!").
- No celebrations ("Nice work!", "🎉").
- No exclamation marks.
- Name failures with the same calm as successes.
- Prefer the noun over the imperative ("Source unreachable" over "Couldn't reach the source").

See `DESIGN.md` § 7 for examples.

---

## Dark mode

Andamentum supports dark mode as a first-class citizen via the `data-theme` attribute on `<html>`:

```html
<html data-theme="light">   <!-- forces light -->
<html data-theme="dark">    <!-- forces dark -->
<html>                       <!-- follows OS preference -->
```

Every component is dark-aware because every component reads from tokens. You should never need to write `@media (prefers-color-scheme: dark)` inside a component rule. If you do, you're styling something with raw hex codes that should be using a token.

**Light is the canonical mode.** All snippets and the index/showcase default to light unless toggled. When sharing screenshots or building demos, default to light.

---

## When you genuinely don't know

If a design decision isn't covered by `DESIGN.md` or this file:

1. Look at how `showcase.html` solves the closest analogous problem.
2. Look at how `brain-mockup.html` or `browse-mockup.html` solves it for a real screen.
3. Look at the matching snippet in `snippets/`.
4. If still uncertain, ask the user before inventing.

Inventing is the worst option.
