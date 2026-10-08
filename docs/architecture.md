# Architecture summary

A one-page orientation to the `andamentum` package. For deeper detail
see the per-sub-module documentation and the project's `CLAUDE.md`.

## What it is

`andamentum` is a single Python distribution of tightly-scoped
sub-modules covering the document-handling pipeline a researcher uses
end-to-end: extract sources, chunk them, index them, search them,
draft new documents, review the result, render figures, typeset for
publication. The modules are independently useful and compose
cleanly — there is no "main entry point" and no hidden orchestrator.

## The two pillars

### 1 — Responsible by construction

The first design commitment is that responsible-use affordances are
implemented as in-code mechanisms, not markdown disclaimers. A reader
can audit *what the package refuses to do* by reading the code, not by
trusting the README.

- **Fetch gate** (`core.fetch_gate`) — every external HTTP(S) fetch
  in `harvest` and `deep_research` consults `/robots.txt` for the
  target host and refuses paywalled academic publishers (Elsevier,
  Springer Nature, Wiley, IEEE, ACM, NEJM, JAMA, Cell Press, Nature,
  Science) unless the caller passes an explicit `tdm_allowed_hosts`
  set attesting to a TDM licence.
- **Confidentiality tripwire** (`whetstone._confidentiality`) —
  scans the document before any whetstone mode (`review`, `panel`,
  `checklist`) calls a model, and refuses to run when the text
  contains markers suggestive of a peer-review submission
  (Manuscript ID:, MS#,
  Submission ID:, Confidential — do not distribute, Reviewer
  Instructions, Editorial Office, Decision Letter, …). Override is
  the explicit `confirm_own_draft=True` flag.
- **AI-provenance watermarking** (`whetstone._watermark`,
  `compose.bundle.parser`, `compose.bundle.render_docx`) — invisible metadata
  stamping (docx core properties, HTML `<meta>` tags, markdown
  HTML-comment headers) always on; visible banner on standalone
  review reports default ON, default OFF on `apply-patches`
  modified-manuscript output. The user can mark prose spans
  with the conventional `[ai-drafted]` / `[ai-edited]` inline
  markers that the bundle surfaces and stamps as docx keywords.
- **Audit log** (`core.audit_log`) — an opt-in cloud-call paper-trail
  *facility* (timestamp · component · model · sha256(prompt) · size,
  one line per call; explicit log path only — no XDG default, no hidden
  home directory). **Not yet wired into the pipelines**: the sub-module
  LLM call sites don't invoke it automatically yet, so it is a building
  block callers can use rather than an automatic guarantee. Full
  coverage is planned — see `RESPONSIBLE_USE.md`.
- **Cloud-vs-local awareness** (`core.cloud_gate`) — classifies a
  pydantic-ai model string as cloud / local / unknown. Unknown
  providers conservatively treated as cloud — over-warning is the
  safe failure mode.
- **Explicit-model-argument pattern.** Every public function that
  calls an LLM takes `model=` as a keyword-only argument. There
  is no shared config module, no env-var-driven behaviour
  selection, no silent fallback. A grep for `model=` traces every
  LLM call site in the codebase.

### 2 — Composable document pipeline

The eleven publicly-documented modules compose into the pipeline
below. Arrows are one-way data flows. There is no orchestrator
that runs the whole thing — callers wire the pieces they need.

```
external source
  (URL / file)
       │
       ▼                                          ┌──────────┐
   ┌─────────┐                                    │ figures  │ ── deterministic
   │ harvest │ ── fetch + format detection +     │          │    plotting only
   │         │    backend dispatch (trafilatura  └────┬─────┘
   └────┬────┘    / docling / passthrough)            │
        │                                             ▼
        ▼                                       ┌─────────────┐
   ┌─────────┐         ┌─────────────────┐      │   bundle    │ ── folder-bundle
   │ chunker │ ────▶   │ document_store  │ ◀────│             │    .docx authoring
   └────┬────┘   2k-10k│ FTS5 + sqlite-  │      └──────┬──────┘
        │       chunks │ vec + RRF       │             │
        │              └─────────────────┘             ▼
        ▼                       ▲                ┌──────────┐
   ┌──────────┐                 │                │ typeset  │
   │   text   │                 │                │ HTML/PDF │
   │  for     │           ┌─────┴─────────┐      └────┬─────┘
   │  review  │           │ deep_research │           │
   └────┬─────┘           │ search+verify │           ▼
        │                 └───────────────┘    rendered output
        ▼
   ┌─────────────┐        ┌──────────┐
   │ whetstone   │        │proofread │
   │ criterion-  │        │  style + │
   │ cascade     │        │readability│
   │ review      │        └──────────┘
   └─────────────┘
        │
        ▼
   ┌─────────────────┐
   │ vision_critique │ ── bounded JSON
   │  of rendered    │    layout audit
   │  figures        │
   └─────────────────┘
```

## Module dependency layering

Sub-modules respect a strict dependency layering so that consumers can
adopt one piece without pulling the others. The constraints are
machine-checkable via grep and are enforced by code review.

- **`core`** is the shared base. No sub-module imports another via
  core; core depends on neither.
- **`harvest`** is a leaf service — depends only on `httpx`,
  `trafilatura`, `docling`, and stdlib. Other modules may depend on
  harvest to convert URLs/files to markdown.
- **`chunker`** depends on nothing but the standard library. Other modules
  may depend on chunker.
- **`document_store`** is foundational. Depends on `core`, `chunker`,
  `references` (enrichers) and `harvest` (ingest only). Other modules may
  depend on it.
- **`deep_research`** depends on `core`, `references`, `harvest`. MUST NOT
  depend on `epistemic`, `document_store`, `compose` or `whetstone`.
- **`typeset`** is a leaf. Used by `whetstone`, `compose.bundle`,
  `deep_research` for HTML/PDF output.
- **`whetstone`** depends on `core`, `typeset`, `chunker`, `harvest`,
  `references` (its only citation reader — the review's citation
  summary, the default reference check and `revise --until-clean citations`), and (opt-in only) `deep_research`. MUST NOT depend on
  `document_store`.
- **`references`** depends on `pydantic`, `httpx`, `rapidfuzz` and —
  for its two model calls only — `core`. It imports no other
  sub-module.
- **`compose.bundle`** (a subpackage of `compose`) depends only on
  `typeset`, `python-docx` and `pyyaml` (a bundle is a plain folder;
  there is no database). It imports no model library, no `core`, no
  `whetstone`, and no compose code outside `bundle/`; a test enforces it.
- **`compose`** depends on `core`, `compose.bundle`, `whetstone` (public API
  only), `proofread`, `llm_judge`, `harvest` and `figures`;
  `deep_research` only behind `--research`. Nothing depends on
  `compose` except `figures.bundle_glue` (the model-free `compose.bundle` only). MUST NOT depend on `epistemic` or `document_store`.
- **`figures`** depends on `matplotlib`, `numpy`, `pydantic`. The
  optional `figures.bundle_glue` is the ONLY place in `figures` that
  imports `compose.bundle`; the rest of `figures` MUST NOT touch it.
- **`vision_critique`** depends on `core` and `pydantic-ai`. Used
  standalone, by `figures` for refinement, or by `whetstone` for
  figure review.
- **`proofread`** depends only on `pydantic` and `textstat`. The
  optional `proofread.cli` is the ONLY place `harvest` is imported.

## What's where

```
src/andamentum/
├── core/             — shared model resolution, fetch gate, audit log
├── harvest/          — universal source → markdown extraction
├── chunker/          — deterministic markdown chunking
├── document_store/   — one SQLite file of documents; hybrid search, search agent
├── deep_research/    — the research loop (plan → queries → search → choose → read → reflect → write) over web and scholarly sources
├── typeset/          — typesetting system (HTML + PDF output)
├── compose/          — a task in, a finished document out (brief → evidence → draft → checks → revise); `compose/bundle/` is the model-free folder-bundle layer → .docx / .md / .html / .txt
├── figures/          — publication-quality figure rendering
├── whetstone/        — document review over user drafts (criterion cascade, panel, checklist, revise)
├── references/       — reference checking in any citation style, with a Crossref lookup
├── proofread/        — deterministic readability + style checking
└── vision_critique/  — bounded LLM critique of rendered figures
```

The package also includes one experimental sub-module that ships
installed but is not described in this overview — see the
"Pre-release / experimental" section of `README.md`.

## Where to read next

- `CLAUDE.md` — full project conventions, sub-module dependencies,
  command reference, and the canonical green-state baseline.
- `RESPONSIBLE_USE.md` — intended use, out-of-scope uses, AI-disclosure
  guidance, source-access expectations, and the in-code protection
  surface.
- `docs/deep_research/overview.md` — deep_research walkthrough.
- `docs/document_store/overview.md` — document_store search and ingestion.
- `docs/design/` — the andamentum visual design system (cream paper,
  serif body, hairline rules) shared by `typeset`, `whetstone`'s HTML
  output, and `compose.bundle`'s render path.

## Pre-release checkpoint

Behaviour is locked for the `0.3.0` release. Test baseline: pyright 30
errors (pre-existing test-only typing noise), ruff clean, pytest
2234 passing.
