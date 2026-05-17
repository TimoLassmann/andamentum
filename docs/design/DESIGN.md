# The Andamentum Way

> The book is the unit. The app is the margin.

This document is the *why*. `components.css` is the *what*. Read this when you need to make a judgement call the stylesheet can't answer for you — should this element exist, should it have a border, should it have a shadow, should the user see a colour here.

---

## 1 — Vibe in three words

**Scholarly. Honest. Typographic.**

- **Scholarly** — the system looks like a well-set monograph. Long measures, generous leading, marginal notes, footnotes, references. It rewards reading.
- **Honest** — no decoration that doesn't serve. No gradients pretending to be light. No shadows pretending the page is floating. No skeuomorphism. No glitter. A hairline says "this is a boundary." That's all the line needs to do.
- **Typographic** — the type does the heavy lifting. Hierarchy comes from size and weight and family, not from colour or chrome. Sans-serif is reserved for metadata, UI, and the small mechanical labels of the system; serif carries every word a human is meant to read slowly.

If you're unsure whether a treatment belongs in andamentum, ask: *would a careful book designer in 1962 have done this?*

---

## 2 — The two registers

Andamentum thinks in two registers, and the type families enforce the distinction.

**Body register (Source Serif 4).** Anything a person reads to learn. Headings. Paragraphs. References. Findings. The user's own document. The assistant's prose response.

**Mechanical register (Inter).** Anything the system says about itself. Metadata. Labels. Buttons. Sidebar items. Badges. Timestamps. Tool logs. The composer.

When in doubt: *is this text the content, or about the content?* Content gets serif. About-the-content gets sans.

---

## 3 — Colour philosophy

### Paper, ink, rules.

Three things touch every component:

- **Paper** — the warm cream background (`--am-paper`). Step deeper for asides, code blocks, recessed inputs. Never more than four steps; andamentum is a flat system that uses tone, not depth, to separate.
- **Ink** — seven steps of warm near-black. Pick the *lightest* step that still holds contrast against its background. Title-weight headings use `--am-ink`; body uses `--am-ink-2`; secondary uses `--am-ink-3`; meta tops out around `--am-ink-5`.
- **Rule** — `--am-rule` is the hairline that separates rows, sections, panels. A 1px line, never thicker except for table headers (which get 2px). If you reach for `box-shadow` to separate two things, stop. Reach for a hairline first.

### The semantic palette.

Five tones — **info, warn, success, danger, note** — each named, each tokenised. But andamentum is deliberately stingy about *when* tone shows up, and deliberately quiet about *how loud* it gets when it does.

**The accents:**

- **info** — quiet slate (`#6e7a8c`)
- **warn** — ochre-grey (`#9a8559`)
- **success** — olive-stone (`#5e7a6a`)
- **danger** — warm rust (`#9c4a36`)
- **note** — paper-2 (neutral)

These are earth-toned on purpose — they share a family with the cream paper and the brown link, so they read as one ink set rather than as five separate status colours. Andamentum's success-green is not Slack-green. Andamentum's warn-yellow is not iOS-yellow. Saturation reads as alarmism in a scholarly system, and we never alarm.

**Where tone shows up:**

| Component | How it carries tone |
|---|---|
| Callout (info, warn, success, note) | Same paper-2 background as every other callout. Only the 3px left bar takes the accent. No tinted text. |
| Callout (danger) | The one exception — slight warm-tint background **and** danger-ink text. Earns the extra signal because it must draw the eye. |
| Badge | Tinted background (in the paper-2 family) + accent text. Quiet at small sizes. |
| Document badge (`supports`, `challenged`) | Same as badge — quiet success/danger bgs with accent text. These are doing real epistemic work, so they keep their colour distinction; they're just no longer shouting. |
| Toast | **No left bar, no tinted background.** The title alone takes the accent. The whole pile sits quietly until you read it. |
| Progress | Default is ink-grey. Only `--danger` takes a coloured fill, used sparingly. There is no `--info / --warn / --success` progress variant — they didn't earn their loudness. |
| Task due pill | `is-soon` uses warn-bg; `is-overdue` uses danger-bg. These are functional — calendar pressure that must be glanceable. |

**The rule of thumb:** when a component can carry tone in only the *type* (title colour, eyebrow label, italic word) rather than in a *shape* (left bar, full-bg fill, border tint), prefer the typographic option. Andamentum is a typesetting system; let the type carry the load.

### The link colour.

`--am-link` is the brown of an underlined word in an old book. It's andamentum's *only* warm accent that isn't earth-neutral or semantic. Reserve it for interactive text and for accenting things you genuinely want the reader to walk toward — the artefact card's left border, focus rings, the source-citation underline.

### Dark mode.

Dark is not "flip everything." Backgrounds carry the warmth of the paper family — `#18160f`, not `#0a0a0a`. Ink becomes warm bone-white. The link becomes warm tan, not blue. Semantic accents shift to lighter equivalents (slate → light slate, rust → tan-rust) but stay in the same earth family. The aesthetic survives the switch.

---

## 4 — Spacing & rhythm

Andamentum uses a 4-step scale: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64.

**Most things sit on 16, 24, or 32.** Cards pad 20–24, sections separate by 48–64, list rows pad 12–16 vertically. Reach for the *named* tokens (`--am-sp-4`, `--am-sp-6`, etc.) rather than dropping raw pixels — it's the easiest way to keep rhythm.

Vertical rhythm matters more than horizontal alignment. Prose runs at 1.85 line height; UI runs at 1.5; tight headings run at 1.3. Don't fight these defaults.

---

## 5 — Things to do

- **Hairlines, not shadows.** Box-shadow appears in exactly three places: the modal, the toast, and a single 1px ground-line on cards (`--am-shadow-1`). Everywhere else, use `border: 1px solid var(--am-rule)`.
- **Small radii.** `--am-rad-1` (3px) for badges, `--am-rad-2` (4px) for callouts, `--am-rad-3` (6px) for cards / modals / inputs. Pills (`border-radius: 999px`) are forbidden except inside the chip component, which is the system's one rounded-edge moment.
- **Type changes register more than colour does.** When you need to separate "this is data" from "this is reading," swap serif → sans, not black → grey.
- **Let the type carry the tone where it can.** Toasts colour the title only. Eyebrows tint to match their section. The system prefers a tinted word to a tinted box wherever the typography can hold the signal.
- **Numbers tabular.** Any time you display a number that aligns with another number — dates, counts, percentages — set `font-variant-numeric: tabular-nums`.
- **Eyebrows for section labels.** 9–11px Inter, 600 weight, uppercase, 0.5–0.6px tracking, `--am-ink-5`. This is andamentum's signature smallcaps moment.
- **Quiet by default.** When deciding between "make this louder" and "make this quieter," the andamentum answer is almost always *quieter*.

---

## 6 — Things to never do

- **No gradients.** Not as backgrounds, not behind buttons, not as accents.
- **No pure black, no pure white.** `#000` and `#ffffff` do not appear anywhere in the palette. Even print uses the dark inks.
- **No drop shadows on cards or rows.** The single allowed elevation is `--am-shadow-1` — a hairline-thin ground line. Modals and toasts get one further step.
- **No coloured left bars on toasts.** Toasts carry tone in the title text only — the body, border, and background stay neutral. This was a deliberate design call; do not re-introduce the left bar.
- **No saturated progress fills.** Default progress is ink-grey. Only `.am-progress--danger` takes colour. There are no `--info / --warn / --success` progress variants — they were removed deliberately.
- **No emoji as UI iconography.** Andamentum uses simple stroke SVGs. If you need a quick mark, set it in the type system (a `+`, an `×`, a `›`).
- **No pill buttons.** Buttons use `--am-rad-2`.
- **No saturated semantic colours.** If your green looks like a status light, it's too saturated.
- **No "vibrant" or "bold" accents.** The brown link is the loudest andamentum gets.
- **No skeuomorphic icons (folders with shadow, paper with curl).** Iconography is flat, single-stroke, calligraphic.
- **No animations longer than 200ms.** Transitions exist only to mute state changes, never to perform.
- **No left-accent-bar with rounded corners.** A semantic callout has a 3px solid left bar against a paper-2 background. It is *not* "a rounded card with a coloured border" — that's a different design language entirely.

---

## 7 — Tone of voice

Andamentum's UI copy reads like the marginalia in a scholarly edition: short, factual, occasionally dry.

| Tone | Don't | Do |
|---|---|---|
| Empty state | "🎉 You're all caught up!" | "No tasks here yet." |
| Error | "Oops! Something went wrong." | "bbc.com/news returned 503. Will retry in 10m." |
| Toast | "Task created successfully ✨" | "&ldquo;Read Pierce 2nd article&rdquo; added to Research." |
| Section intro | "Welcome to your inbox!" | "Items routed here for triage. Filed into a project to start work." |
| Confirm dialog | "Are you sure you want to delete this?" | "Delete this task? Sub-tasks and notes are removed too." |

No exclamation marks. No emoji in interface copy. No "we" or "you're" if you can avoid it. Andamentum does not greet the user; it informs them.

When something has gone wrong, andamentum names the failure with the same calm it uses for a successful result. The system's emotional register is *constant*.

---

## 8 — Iconography

Stroke-based SVGs, 1.5 px weight, 14–18px square, current-colour. The system is text-led; the label is usually enough on its own, so **when in doubt, omit the icon**.

Filled icon shapes are reserved for a small whitelist:

- **Status dots** (`.am-dot--*`) — tiny semantic indicators
- **Inline badge dots** (`.am-badge__dot`)
- **Task status circles** (`.am-task__check`) — see § 9
- The small **×** inside `.am-task__check.is-cancelled`

If you find yourself drawing a filled icon that isn't on this whitelist, you're probably reaching for a UI metaphor that doesn't fit andamentum.

---

## 9 — Task status family

Six statuses, each a 18-pixel circle. The set spans the palette deliberately — neutral for not-yet-started states, semantic for committed work — so a row of mixed tasks reads as a quiet horizontal rhythm rather than a row of flags.

| Status | Visual | Palette |
|---|---|---|
| `.is-inbox` | Dashed ink-grey outline | neutral |
| `.is-todo` *(or no modifier)* | Solid ink-grey outline | neutral |
| `.is-progress` | Slate half-fill | info |
| `.is-waiting` | Ochre outline | warn |
| `.is-done` | Solid olive disc | success |
| `.is-cancelled` | Rust outline + tiny × | danger |

Done and Cancelled both carry strike-through on the title (via `.am-task.is-done` / `.am-task.is-cancelled` on the parent row) — the typography does the second half of the work the icon starts.

---

## 10 — Density

Andamentum is a medium-density system. Less dense than a spreadsheet, denser than a marketing page. List rows are 44–52px tall. Cards pad 20–24px. Section heads breathe 48–64px above their content.

If you find yourself adding "let's just give this some room to breathe" margin everywhere, you're probably right — but reach for the next spacing step up, not an arbitrary 32px.

---

## 11 — Applying the system to new components

When you need a component that doesn't exist yet:

1. **Does the type system already say it?** A new "info banner" is probably just `<aside class="typeset-callout tone-info">`. A new "label/value table" is probably `.typeset-items`. A new "settings row" is `.am-settings-row`. Look first.
2. **If not, mock it with tokens, not values.** `padding: 20px; background: #f4f1ec; border-radius: 6px;` → `padding: var(--am-sp-5); background: var(--am-paper-2); border-radius: var(--am-rad-3);`.
3. **Name it `.am-{noun}`.** Block-element-modifier: `.am-toast`, `.am-toast__title`, `.am-toast--success`. Verbose state classes use `.is-*` (`.is-active`, `.is-done`).
4. **Add it to `showcase.html`** under the right section, with a short label and a tiny `<small>` description. The showcase is the source of truth.
5. **Drop a snippet in `snippets/`** with the markup, following the wrapper pattern of the other snippet files (light-mode HTML doc, begin/end snippet markers).

---

## 12 — A note on the name

*Andamento* is an art-historical term for the rhythm and flow of an inlaid mosaic — the line the tesserae follow across a field. The system aims at that quality: each row, each card, each row of metadata laid down with deliberate flow, never one piece louder than the line it sits in.
