# Changelog

## Unreleased

### Added

  - **`andamentum-serve`**, a local HTTP server (127.0.0.1 only) over the
    document library (search and read), whetstone review of a markdown
    note, `accept_edits` and research. The model is named in each request;
    calls on a local Ollama model run one at a time; every procedure call is
    kept under `-o DIR`. Needs the new `serve` extra (FastAPI, uvicorn).
  - **whetstone:** every `Edit` has an `id`;
    `ReviewedDocument.from_markdown(text)` reviews text without reading it as
    a path or a URL; `accept_edits(markdown, edits)` applies the edits a
    reader accepted to the markdown and lists each one it could not apply.
  - **core:** `build_pydantic_ai_agent(..., prompted=True)` builds the
    PromptedOutput agent (schema in the prompt, answer as text).

### Fixed

  - **andamentum-word:** `check`, `respond`, `refresh`, `bind-numbers` and
    `insert-citation` no longer write over an existing copy or report in
    `-o DIR` (a copy edited in Word was lost on a rerun); they stop with
    exit 2 before writing anything, as `draft` already did.
  - **whetstone editor:** an edit without a title is kept and labelled from
    its rationale, instead of costing the section all its edits; when the
    tool-call run fails (gemma4 on Ollama sometimes returns an empty reply),
    the editor runs again with the schema in the prompt.

### Removed

  - **`andamentum-whetstone apply-patches`** is removed, with
    `finalize_reviewed_document` and `PatchDocxEditor`, which only it used.

### Interface (every tool)

Breaking: command lines and some Python names changed so that every tool has
the same shape (`docs/interface.md`).

  - **`andamentum list`** (`--json`) lists every tool from `andamentum.TOOLS`;
    `andamentum <tool> ...` runs one. Each tool keeps its own
    `andamentum-<tool>` command.
  - **The same flags everywhere:** `-o/--out` (a folder when a command writes
    several files; nothing is written anywhere else: no files in the current
    folder or the home folder), `--json`, `-` for stdin, `--version`, and one
    set of exit codes: 0 success, 1 negative result (problems found, goal not
    met), 2 usage, 3 input unreadable, 4 run failed. Errors are one
    `error:` line.
  - **Models:** every command that calls a model needs `--model`. The
    environment variables `ANDAMENTUM_MAIN_LLM_MODEL`,
    `ANDAMENTUM_EMBEDDING_MODEL`, `ANDAMENTUM_LLM_CONCURRENCY`,
    `ANDAMENTUM_AUDIT_LOG`, `ANDAMENTUM_PANEL_OWN_AUTHOR`,
    `DOCUMENT_STORE_DIR` and `ANDAMENTUM_DATABASES_DIR` are gone. Second
    models have no default (`harvest --figure-model`).
  - **compose is two commands:** `andamentum-compose` writes a new document
    from a task into `-o DIR` (`write`, `continue`, `check`, `render`, ...;
    `run` is `continue`, `check-bundle` is `check`, `render --raw` is
    `render --plain`); `andamentum-word` works on a Word file you already
    have (`check`, `respond`, `refresh`, `bind-numbers`, `draft`,
    `find-citation`, `insert-citation`, `paragraphs`, `mcp`), always into
    `-o DIR`.
  - **New commands:** `andamentum-references` (the reference check, was
    `andamentum-whetstone references`); `andamentum-whetstone proofread` is
    gone (use `andamentum-proofread`).
  - **whetstone:** real subcommands, each with its own flags; `review` stays
    the default; `--list-reviewers` is `reviewers`; the `--apply-patches`
    flag is the `apply-patches` command; `--goal-figure-model` is
    `--figure-model`; `_v3` names and the `v3/` folder are plain
    (`review_document`, `pipeline/`).
  - **research:** `ask` (the default), `mcp`, `cache info|prune`; the report
    goes to stdout or into `-o DIR` with its log; nothing in the current
    folder.
  - **epistemic:** `ask`/`verify` keep the database and report only in
    `-o DIR` (no home-folder databases); `inspect DIR`, `confidence DIR`;
    Python `andamentum.epistemic.ask()` / `verify()`.
  - **Python:** one main function per tool (`andamentum.TOOLS[...].main`),
    one base error per module, smaller `__all__` lists; renamed:
    `vision_critique` `extra_context` → `context`, `llm_judge` `output` →
    `text`, `figures.figure(output=)` and `forge.run_forge(dest=)` → `out=`,
    `compose_bundle` → `continue_document`.

### Documentation for users

- New user guide, `docs/guide.md`: a table from the job you want done to the command, then each of the sixteen tools with commands to copy.
- Every command and subcommand ends its `--help` with one to three examples. `tests/test_interface.py` checks that each command has one and that every example in `--help` and in the guide is accepted by the command.
- The README has a command-line quickstart and a tool table with one example per tool.
- Installation is from GitHub (`uv tool install "andamentum @ git+https://github.com/timolassmann/andamentum.git"`). The name `andamentum` on PyPI belongs to an unrelated project, so the README, the compose and word guides and the PDF and HTML extra hints no longer say `pip install andamentum`. `docs/index.md` no longer says there are no extras.
- `andamentum-agentic-dialect skeleton --help` says to save the skeleton as `graph.py`, the only file that may import the graph engine.

### Fewer flags, presets instead (round 2)

Breaking: command lines lost flags that only tuned numbers or repeated
another flag. The rule is now in `docs/interface.md` ("Presets, not knobs"):
the command line has presets, the Python functions keep every setting.
Removed flags exit 2; there are no aliases.

- **research:** `andamentum-research ask` replaces twelve flags
  (`--max-rounds`, `--queries-per-round`, `--reads-per-round`,
  `--window-chars`, `--expand-per-round`, `--results-per-query`,
  `--max-llm-calls`, `--max-wall-s`, `--max-fetch-bytes`, `--concurrency`,
  `--openalex-search`, `--sources`) with `--depth {quick,standard,deep}`.
  `--local-corpus` with `--mode web` is a usage error (the corpus is searched
  with the scholarly sources).
  `standard`, the default, is the old default budget exactly; `quick` and
  `deep` are starting points, not measured. Python: `Budget.for_depth(depth)`
  and the `Depth` type. The scholarly sources are always OpenAlex, Europe PMC
  and arXiv; `--local-corpus PATH` adds your own documents. Semantic Scholar
  is reachable only from Python (`SourceConfig`).
- **whetstone:** the goal loop is the command `revise`:
  `andamentum-whetstone revise DRAFT (--until "TEXT" | --until-clean
  {structure,citations,figures}) --model M -o OUT` (at most 4 rounds,
  `RevisionSpec(max_rounds=)` in Python; `--figure LABEL=FILE` and `--figure-model` with `--until-clean figures`;
  exit 1 when the condition was not met, output still written). `review`
  loses `--goal`, `--goal-check`, `--goal-figure`, `--figure-model`,
  `--max-rounds`, `--editor-criteria`, `--rounds`, `--novelty-search-depth`
  (fixed at 2) and `--novelty-store`. `--criteria` and `--guidelines` are one
  `--brief TEXT|@FILE`. `--lookup/--no-lookup` is `--offline` (refused with
  `--check-novelty`). Python: `run_goal_loop` / `GoalSpec` / `GoalRunResult`
  → `revise_document(doc, *, spec=RevisionSpec(...), model=)` /
  `RevisionSpec` / `RevisionResult` (`condition_met`); `whetstone/goal/` →
  `whetstone/revise/`. The watermark and author flags stay.
  `revise --until-clean citations` refuses `--no-references` (exit 2). The
  suggested next-step `revise` command repeats `--decision-model`,
  `--confirm-own-draft`, `--offline` and `--no-references`.
- **references:** `--no-lookup` → `--offline`, the same word as whetstone and
  word.
- **`--offline` means "make no network call besides the model"** in every
  tool, with one help sentence. whetstone `review` / `revise` refuse a web
  address as the input with it, and `andamentum-word draft` refuses a web
  address as `--source` (exit 2: save the page as a file first).
- **compose and word:** `andamentum-compose write`, `continue`, `check`,
  `eval` and `andamentum-word draft` take one required `--model`; the role
  flags `--drafter`, `--critic`, `--judge`, `--reader` are gone (Python keeps
  `Roles`). `--max-rounds` is gone (presets: 4 rounds for compose, 2 for word
  draft). Research is one flag with a required value,
  `--research {auto,web,scholarly,both}` (`auto`: the genre chooses);
  `--research-mode` is gone. `compose continue FOLDER --model M` repeats the
  run with the settings `write` recorded in the folder (research mode,
  formats, provenance, `--confirm-own-sources`), so it no longer takes
  `--format`, `--no-provenance` or `--research-mode`; a folder without the
  record exits 3. `continue` and `check` no longer fall back to the models
  recorded in the folder.
- **figures:** `plot` loses `--mode` (`--journal showcase` draws the same
  figure) and `--dpi` (300).
- **epistemic:** `ask` and `verify` lose `--quick` and `--trace` (the full
  run and the timeline trace, the old defaults).

### document_store

  - **One store, rebuilt.** The v1 store (`DocumentStore`, `search`,
    `find_by_metadata`, the deferred-ingestion queue, DHP clustering) and the
    v2 section store are replaced by `Store`: fixed source tables that keep
    every revision in full and the original bytes, disposable indexes rebuilt
    from a change log, and a fixed async API (`put`, `get`, `resolve`,
    `update`, `delete`, `restore`, `purge`, `list`, `count`, `search`, `read`,
    `sync`). `list` has no default limit. Old database files are not read.
  - **Search.** `HybridRerankSearcher` (default): BM25 + vectors over chunks,
    RRF top 50, cross-encoder rerank. `AgentSearcher` (opt-in): a model-driven
    tool loop on top of it.
  - **Enrichers** (`document_store.enrich`) write identifiers, bibliographic
    records, basic facts, LLM-extracted fields or decision-model labels into
    a document's metadata.
  - CLI: `andamentum-docstore ingest | sync | search | get | list | delete | enrich`.

### chunker

  - **One deterministic chunker.** `chunk(text, *, max_chars)` splits at
    headings, then paragraph breaks, then sentence ends, never inside fenced
    code or tables. The embedding split, the LLM judge and `extract_units`
    are removed.

### references

  - `extract_identifiers` (now also PMCID, ISBN, URLs), Crossref `fetch_work`
    / `search_crossref` / `to_csl` and PubMed `parse_pubmed_xml` /
    `fetch_pubmed` move here from epistemic and compose.

### proofread

  - **Slide mode.** `andamentum-proofread deck.pptx --slides` runs per-slide
    checks instead of the prose checks: headline length, a headline that
    opens on "This" or "These", banned words, acronyms not expanded on the
    slide (`--allow MEK,RAS` for ones the room knows), one entity under two
    names, and a body that restates its headline. Library:
    `analyze_slides(list[Slide])`, pure. `python-pptx` is now declared; it
    was already installed through docling.

### whetstone

  - **One run does most things; flags turn parts off.** The default
    `andamentum-whetstone draft.docx --model M --out OUT` now reviews,
    writes rewrites as tracked changes (the editor is on by default), checks
    the references with a Crossref lookup, and lists what the draft claims
    as new. `--no-editor`, `--no-references` and `--no-lookup` turn parts
    off. **Behaviour change:** `--editor` is now the default, and the
    Crossref lookup sends the draft's reference entries (never its text)
    to Crossref unless `--no-lookup`. The library `review_document` keeps
    `editor=False`.
  - The reference check runs before the review, so the editor answers its
    problems beside the review's own; each of its comments is signed by the
    reference checker. The confidentiality tripwire runs first, over the
    headers and notes as well as the body.
  - Two new report sections in all three renderers: **Reference check**
    (how the draft cites, the counts, the Crossref result) and **Next
    step** — when problems remain, the exact goal-loop command, writing to
    a new `.goal` file. The goal loop stays opt-in: only the user can state
    the goal, it rewrites the draft several times, and each round costs a
    full review.
  - The reference check alone is the separate command `andamentum-references`
    (see Interface below); no model needed (`--model` settles only what the
    text leaves open).
  - `--goal-check citations` now uses the reference checker (every citation
    style), and its problems reach the round's editor, so the loop can fix a
    wrong citation, not only report it. The simpler marker scan it used
    (`goal/citations.py`) was removed.
  - The review's `CITATIONS` prompt line now comes from the reference
    checker (style, count, list size). The old marker regexes read
    "(none)" for any superscript or footnote document. The unused
    `Claim.has_citation` was removed.
  - A point several reviewers raise is one comment, naming the others;
    each finding has a headline, an issue and one fix; the editor's answer
    sits in the finding's own comment ("Suggested rewrite" / "Left to
    you"). Fixed: a Word file that registered its core properties under the
    variant relationship type came out with two `docProps/core.xml` parts
    and would not open.

  - **Checklist mode** — a third review mode beside `review` and `panel`,
    for the question "run the standard check" rather than "help me improve
    this draft". Instead of the criterion cascade it runs a fixed bank of
    `Probe`s (one defect class each, phrased as extraction under a quota:
    "find the 3 claims the evidence supports least well"), and instead of
    LLM-written prose it summarises with a computed coverage table — one
    row per probe, in bank order, where a crashed probe reads `not run`
    and never `0`. Findings are anchored the same way review-mode findings
    are; `absence` probes, which have no quote to anchor, are dropped when
    any of the search terms they ship turns up in the source. Two probes
    landing on the same anchor are cross-linked (`corroborated_by`) and
    both kept, never merged. Deliberately no gap loop, no consolidation
    call and no synthesis call: the cost is one agent call per probe, and
    predictability is the product.
  - Fourteen default probes ship, routed by the same six document types as
    the criterion sets (`probe_bank_for`; unknown type falls back to the
    general bank with a warning). A caller with a house checklist passes
    `probes=[Probe(...)]` — the machinery is fixed, the probe list is data.
  - New API: `run_checklist(markdown, *, model)` and
    `checklist_document(source, *, model)`, both returning the same
    `ReviewResult` the existing renderers already consume. New CLI
    subcommand: `andamentum-whetstone checklist INPUT --model M --out OUT`.
  - The confidentiality-marker tripwire runs before any model call in
    checklist mode exactly as it does in review mode; `--confirm-own-draft`
    bypasses it. Unlike `panel`, there is no mandatory authorship
    affirmation — checklist output is a defect table, not a document shaped
    like a journal peer-review report.

### references (new)

  - `andamentum.references.check_references(text, *, sections, hints,
    model, lookup)`: finds the reference list and every candidate
    citation, decides how the document cites from evidence (numbers in
    brackets or parentheses, superscript, author-year, footnotes and
    endnotes, Pandoc keys), checks the citations against the list, and
    looks each entry up in Crossref. A model is asked only when the
    evidence is mixed, with short excerpts and flat answers sized for
    local models; without one, open cases are reported as undecided,
    never guessed. Measured on a test set of 13 real CSL styles plus
    formatting-stripped and notes variants, from Word and from PDF.

### harvest

  - `preserve_superscripts=True` (PDF only, off by default): raised text
    becomes `<sup>`, and page footnotes become Markdown footnotes, read from
    the PDF's own characters with pypdfium2.

### document_store

  - **Set-membership filters in `find_by_metadata`.** A filter value may now be
    a list/tuple/set, matched as `field IN (...)` — e.g. `{"status": ["todo",
    "in_progress", "blocked"]}` in a single query. A scalar still matches by
    equality and `None` by SQL NULL; a string stays a scalar (not iterated). An
    empty collection matches nothing (fail-closed) rather than every document.
  - **`include_content=False` on `find_by_metadata`** skips the per-document
    content read, leaving `snippet` empty — a cheap path for overviews/counts
    over large result sets.
  - **New `describe_metadata(database, *, filters=None, max_values=25)`** —
    discovers the schema-less metadata vocabulary: returns each top-level field
    as a `FieldProfile` (`present_in`, `distinct`, and a bounded value→count
    breakdown). Optional `filters` scopes the profile to a subset for drill-in.
    Lets a caller (human or agent) fill `find_by_metadata` filters without prior
    knowledge of which fields/values exist. Also exposed as
    `DocumentStore.describe_metadata`.

### Packaging

  - **`requires-python` raised to `>=3.12`** (was `>=3.11`). Python 3.11
    is dropped: the modern pydantic / pydantic-ai stack assumes the 3.12
    `typing.TypedDict` semantics (3.11 needs `typing_extensions.TypedDict`),
    and the maintenance cost of tracking that drift outweighs 3.11 reach.
    The CI matrix now tests 3.12 + 3.13 on Linux and macOS.

### CI

  - Resolve `torch`/`torchvision` from the PyTorch CPU index, dropping the
    multi-GB NVIDIA CUDA runtime from Linux installs.
  - Pin CI `uv` to the `0.11.x` series so `uv sync --locked` matches the
    lockfile format; force the test matrix to use its declared Python
    interpreter (a committed `.python-version` was pinning every leg to 3.12).
  - Add `scripts/ci-local.sh` to reproduce the workflow locally via `act`.

## 0.3.0 — 2026-05-31

First stable release. Headline change is the whetstone v3 consolidation:
v2's lens-based section-by-section pipeline is gone; v3's criterion-
cascade is the only review path. The rest of this entry covers
follow-up polish and release-prep work since `0.3.0rc3`.

### Whetstone v3 consolidation

  - Six pluggable criterion sets routed by document type (academic /
    external_communication / essay / tutorial / creative / general),
    auto-detected by a one-shot classifier or set explicitly via
    `--document-type`. Essays, tutorials, and creative writing get
    criteria that fit the form instead of being forced through SPECS.
  - `--criteria` / `--guidelines` collapsed into one unified
    criterion-input surface. Pass either pre-decomposed `criteria=`
    or free-text `guidelines_text=`; an extractor decomposes the
    latter into a criterion list in one LLM call.
  - Confidentiality tripwire (`_confidentiality.py`) ported to v3 and
    runs as the very first step of `run_review_v3`, before any LLM
    call including the auto-classifier.
  - Editor phase ported as an optional node between Synthesise and
    Finalize; concrete `Edit` objects feed the docx track-changes
    renderer when `--editor` is passed.
  - Novelty check restructured from v2's single-node-with-tool shape
    into a deterministic 3-node pipeline (FlagNoveltyTargets →
    RunNoveltySearches → JudgeNovelty). Reproducible flag-set,
    bounded search budget, on-disk novelty cache dropped per the
    no-hidden-home-dirs rule.
  - Panel mode now lives in its own graph (`whetstone.v3.panel`).
    Invoked via the new `andamentum-whetstone panel` subcommand;
    keeps the existing N-expert biosketches + per-expert reviews +
    panel synthesis shape, with the in-code authorship gate
    (`--i-am-the-author`) preserved.

### CLI: five subcommands

  `andamentum-whetstone` is now organised around verbs:
    review (default — criterion-cascade), panel, proofread,
    apply-patches, verify-provenance.
  Bare positional invocations still route to `review` so existing
  scripts don't break. 7 dead v2 flags pruned (`--v3`,
  `--embedding-model`, `--no-challenge`, `--perspectives`,
  `--no-llm`, `--no-proofread`, `--persist-novelty-cache`).
  Remaining flags organised into argparse argument groups so the
  `--help` output reads cleanly.

### Output layout

  - HTML + markdown renderers rewritten to an editorial-annotation
    layout: per-finding section header → quoted passage in
    `tone-quote` callout → comment in `tone-warning` / `tone-note`
    callout with severity / confidence chips. Composes existing
    typeset atoms — no CSS edits, no new visual primitives.
  - Document map moves from the bottom of the report to the top so
    section ids in finding headers are interpretable on first read.
  - The two stacked top banners (note + AI-watermark) merge into
    one combined warning.
  - No more collapsed `<details>` cards; comment body is visible
    at first read.

### Metrics + reliability

  - Real LLM-call + gap-round counters wired throughout v3 (was
    always `0` after the consolidation regression). Uses a
    `ContextVar`-held counter so `asyncio.gather`'d tasks aggregate
    into one total. Reads pydantic-ai's `result.usage().requests`
    to capture tool-call expansion.
  - Loud-fail on missing input path: a path-shaped string that
    doesn't exist now raises `FileNotFoundError` with a clear
    message; previously the path string was silently treated as raw
    markdown content (the LLM "reviewed" the literal string
    `/tmp/draft.md`).

### Release-prep

  - **Supply-chain hardening.** Added a `[tool.uv]` block with a
    rolling 28-day cooldown — at resolve time `uv` ignores any
    package version uploaded in the last 28 days, blocking
    freshly-published malicious releases from being pulled before
    the community has time to flag them. The lockfile remains the
    actual pin; CI / containers should run `uv sync --locked` (or
    `UV_LOCKED=1`) to refuse silent drift.
  - **Dependency upper bounds.** Every direct dependency now has an
    explicit upper bound (e.g. `pydantic>=2,<3`, `docling<3`,
    `matplotlib<4`, `sqlite-vec<0.2`) so a future major bump can't
    silently break the install.
  - **Dependency surface.** `trafilatura` (Apache-2.0,
    MIT-compatible) is an optional extra
    (`pip install andamentum[html-articles]`) to keep the default
    install lean — the harvest backend falls back to `docling`
    automatically when it isn't present.
  - **PyPI metadata.** `[project.urls]` block added (Homepage,
    Repository, Documentation, Changelog, Issues) so PyPI's package
    page links back to the source.
  - **Internal-only material kept out of the published package.**
    Internal design docs / PRDs / plans are gitignored and not
    tracked; smoke-run review outputs are not shipped. The
    published tree carries only user-facing docs and examples.
  - `andamentum-epistemic` prints an explicit `⚠ EXPERIMENTAL`
    banner to stderr on every invocation. The module ships
    installed and is callable, but output shape / agent names /
    flags may still change without notice.
  - **Dual-use hardening for whetstone.**
    - Confidentiality tripwire extended with three phrases that
      describe the *act of reviewing someone else's grant*
      (`Assessor report/comments`, `Funding-panel review`,
      `Peer review of grant/application/proposal`). Scheme
      prefixes (NHMRC APP, ARC DP, NIH RFA-) and role labels
      (`Lead CI`, `Chief Investigator A`) deliberately do NOT
      fire — they appear in the user's own draft as well, and
      gating on them would produce a false-positive rate high
      enough to train users into reflex `--confirm-own-draft`
      bypass. The category-level prohibition lives in
      `RESPONSIBLE_USE.md`.
    - `.docx` provenance now has a second invisible layer: a
      `customXml/andamentum-provenance.xml` part inside the docx
      zip (namespace `urn:andamentum:provenance:v1`), anchored
      via a package-root relationship so it survives the next
      python-docx round-trip. The layer-1 core-properties fields
      (author, keywords, comments) are user-editable from Word's
      File → Info pane; the customXml part requires manual zip
      surgery to remove, so the provenance survives the most
      common "scrub before sharing" actions.
    - New `andamentum-whetstone verify-provenance INPUT.docx`
      subcommand — read-only inspection that reports which AI-
      provenance markers a `.docx` carries (core-properties +
      customXml). `--format json` for machine-readable output.
      Exit 0 if any marker present, 2 if readable but clean, 1
      on read error.
  - **CI workflow.** `.github/workflows/test.yml` runs pytest,
    ruff check, ruff format check, pyright (non-blocking on the
    pre-existing test-only typing errors), and an examples smoke
    job across Python 3.11 / 3.12 / 3.13 on Linux and macOS.
    `uv sync --locked` refuses silent lock drift. A weekly
    cron run picks up supply-chain rot in dependencies.

### Other

  - Documentation refreshed for the v3-only surface (README,
    docs/index.md, docs/architecture.md, both RESPONSIBLE_USE.md
    files). `docs/whetstone/lenses/strunk.md` removed — the Strunk
    lens was a v2-only feature deleted in the consolidation.
  - `<your-github-handle>` placeholder in `CITATION.cff` and the
    `harvest` / `deep_research` User-Agent string replaced with
    `TimoLassmann`.

### Release hardening (pre-release audit)

A comprehensive pre-release audit drove a final hardening pass:

  - **SSRF: redirect re-validation + size cap.** `is_safe_url`
    validated only the initial URL while httpx followed 3xx hops
    unchecked — a public URL that redirected to a private /
    cloud-metadata address defeated the whole defence. The new
    `core.url_safety.fetch_with_safe_redirects` re-validates every
    hop and streams the body under a size cap (`ResponseTooLarge`).
    Wired into `harvest`, `deep_research`, and `vision_critique`
    (which previously did no SSRF check at all). `url_safety` moved
    `harvest` → `core` so all three can share it without a layering
    breach.
  - **HTML/attribute escaping in `typeset`.** Plain-text and
    attribute fields (subtitle, labels, badges, sidebar values,
    `href`s) are now HTML-escaped and link schemes validated —
    previously an injection vector for LLM- or web-derived content
    rendered into a report.
  - **Honest failure modes.** whetstone surfaces `failed_criteria`
    instead of silently dropping a crashed criterion; the novelty
    checker returns `is_novel=None` (undetermined) on search failure
    rather than a misleading "novel"; document-store migrations log
    rather than swallow; the AI-provenance watermark logs on failure.
  - **`find_by_metadata`** validates caller field names before
    interpolating them into a SQLite JSON path.
  - **Packaging.** `requires-python` raised to `>=3.11` (the floor
    actually imported); the wheel excludes tests; the sdist ships
    `docs/` + `RESPONSIBLE_USE.md` + `CHANGELOG.md` and excludes
    internal material; Trove classifiers + keywords added; `py.typed`
    completed across all sub-packages; an optional `pdf` extra
    (WeasyPrint) for typeset PDF output; `--version` on every CLI.
  - **Bedrock model-id map** reduced to an empty stub — the exact
    friendly→ID mappings are account/region-specific local config,
    not shipped surface.
  - Removed internal scratch files.
  - Repo-wide `ruff format`; README reframed to a neutral toolkit
    positioning.

## 0.3.0rc3 — 2026-05-13

Adds a **second report layout** alongside the classic one — Cochrane-
style audit report. The classic layout (``typeset_report.py``) is
unchanged; the new audit layout lives in ``audit_report.py``. Choose
between them via ``--report-style {classic,audit,both}``. ``both``
writes two files so they can be opened side-by-side for comparison.

### Why a parallel layout

The classic report is prose-led and reads cleanly when the answer is
self-evident. The audit report is structured for cases where the
*reasoning trail* is the value — clinical decision support,
regulatory submissions, anywhere "show your work" matters. The two
layouts are not redundant; they target different reader needs.

### What the audit layout shows

1. **Headline panel** — claim, verdict badge (Supported / Refuted /
   Inconclusive / Insufficient evidence), posterior pill.
2. **Summary of findings** — small Cochrane-style table immediately
   under the headline: directional split (supports / contradicts /
   no bearing) with counts and percentages.
3. **Plain-language summary** — the existing answer prose, separated
   from the evidence breakdown.
4. **Key evidence per claim** — claim card (with collapsible details
   carrying the audit trail) plus the top 3-5 supporting items and
   the strongest counter-evidence rendered inline as a markdown list
   with **clickable source links** (DOI / PMID / NCT auto-link). The
   classic layout's inline reference-number list (which on a 98-item
   claim rendered as ``1, 2, 3, …, 98``) is gone.
5. **Audit trail per claim** — investigation rounds as a real list
   (each round its own bullet, properly rendered), IBE chain
   candidates as a markdown table with loveliness / likeliness
   scores, and the adversarial probe explicitly framed. All folded
   into the claim card's ``<details>`` so the reader opts in.
6. **Caveats & Limitations** — same data as classic, visually
   separated.
7. **Appendix** — a single collapsible card containing the full
   evidence trail (every retrieved item with its one-sentence
   judgement and clickable source), grouped by direction. The full
   list is available for verification but not imposed on the scanner.

### Source-ref clickability

DOI, PMID, and NCT identifiers are now auto-converted to clickable
URLs:

- ``doi:10.1234/abc`` → ``https://doi.org/10.1234/abc``
- ``PMID:12345678`` → ``https://pubmed.ncbi.nlm.nih.gov/12345678/``
- ``NCT04501978`` → ``https://clinicaltrials.gov/study/NCT04501978``
- Bare ``https://`` URLs are unchanged.

Used by the audit layout for both inline highlights and the
appendix's full list. The classic layout is unchanged.

### Run modes

The audit layout handles both run modes from the same code path:

- **Verify mode**: one claim seeded from ``claim_to_verify`` — single
  ``Key evidence`` section with the claim card and audit trail.
- **Research mode**: question decomposed into sub-investigations —
  ``Sub-investigations`` section with each sub-claim numbered
  (#1, #2, …), its own card, key-evidence breakdown, and audit
  trail. The combined verdict shows in the headline.

### Files

- ``src/andamentum/epistemic/audit_report.py`` (new) — the Cochrane-
  style renderer. ~500 LOC. Uses only the 7 built-in typeset atoms
  (heading, prose, callout, items, aside, card, reference) — no
  custom atoms added.
- ``src/andamentum/epistemic/report_generator.py`` — ``generate_html``
  and ``save_html`` now take a ``style`` parameter
  (``"classic"`` (default) | ``"audit"``).
- ``src/andamentum/epistemic/cli.py`` — ``--report-style`` flag added
  to ``ask`` and ``verify`` subcommands. ``both`` writes two files
  with ``-classic.html`` and ``-audit.html`` suffixes.
- ``src/andamentum/epistemic/cli_handlers.py`` — ``handle_ask``
  threads ``report_style`` through to ``save_html``.

### Tests

- 22 new tests in ``test_audit_report.py``. Covers source-URL
  conversion, verdict callouts (Supported / Refuted / Inconclusive /
  Insufficient), Summary-of-Findings table, single-claim rendering
  (no inline number list, evidence counts in details, clickable
  source links, audit trail in card details), research-mode rendering
  (sub-investigations section, numbered sub-claims), and appendix
  presence/absence.

### Verification

- pyright: 23 errors (unchanged baseline).
- ruff: clean.
- pytest: 2105 passing (+22 audit-report tests), 2 skipped,
  25 deselected.

The classic report is bit-for-bit unchanged; existing databases
re-render identically under ``--report-style classic``. The audit
layout consumes the same ``ReportData`` so existing dev30 v9
databases render audit-style with zero LLM tokens.

## 0.3.0rc2 — 2026-05-13

Report-rendering update. Surfaces the audit trail of investigative
work that was already stored in entity state but previously invisible
to readers of the rendered report. **No behaviour change** — the
pipeline, operations, agents, and scoring are unchanged. Pure
rendering and report-data extraction. Existing databases re-render
into the new shape; no re-run required.

### New report sections

- **How this claim was investigated** (per claim) — every follow-up
  intent the gap-analysis agent proposed across investigation rounds,
  with the routing-yield count per intent. Empty when the claim
  reached a verdict on initial gather alone. Reads from
  `Claim.investigation_intents`.
- **Inference to the best explanation** (per claim) — every IBE
  candidate the integration step enumerated, with its loveliness /
  likeliness scores, the chosen candidate marked, the integrated
  assessment shown. Reads from `Claim.integration_candidates`.
- **Adversarial probe** (per claim) — counterarguments are now
  prefaced with an explicit "the system searched for evidence that
  would contradict this claim" intro, so the reader sees the probe,
  not just the result. Same underlying data; reframed presentation.
- **Evidence judgement breakdown** (top-of-Sources) — total support /
  contradict / no_bearing counts with percentages. The reader can see
  at a glance the audit-trail view of how each retrieved item was
  categorised.

### Schema additions to `report_data.py`

- `InvestigationRound` dataclass — text + evidence_count per round.
- `IBECandidate` dataclass — candidate description, loveliness,
  likeliness, chosen / runner_up flags, gap scores.
- `ClaimSummary` extended with `investigation_rounds`,
  `ibe_candidates`, `integrated_assessment`, `integrated_confidence`.
- `InvestigationStats` extended with `evidence_supports`,
  `evidence_contradicts`, `evidence_no_bearing`, `evidence_invalidated`,
  `investigation_rounds_total`.

### Tests

- 8 new tests in `test_report_audit_trail.py` covering: investigation
  rounds rendering (present, empty, singular/plural yield), IBE
  candidates rendering (chosen/runner-up/rejected), adversarial probe
  intro, evidence judgement breakdown.

### Verification

- pyright: 23 errors (unchanged baseline).
- ruff: clean.
- pytest: 2083 passing (+8 from new audit-trail tests), 2 skipped,
  25 deselected.

Re-renders cleanly against existing databases. Smoke-tested against
`test2_hcq` (100 evidence items, 9 investigation intents across 3
rounds, verdict: fail — correctly identified) and `test3_statins`
(33 items, 5 IBE candidates, chosen=B "supports_refined", verdict:
pass).

## 0.3.0rc1 — 2026-05-13

Pre-release tag. This is the first version of `andamentum` with a
unified routing layer across initial gather and follow-up
investigation rounds, plus an explicit stage invariant that prevents
the class of silent-calibration-regression bug that the architectural
work surfaced and fixed.

### Highlights

- **Description-driven dispatch is the only evidence-gathering path.**
  Provider knowledge lives on each provider class as four self-
  description attributes (`description`, `query_guidance`,
  `query_examples`, `output_kind`). A single generic dispatch agent
  reads those attributes at runtime to commit native-syntax queries
  or abstain. Adding a new provider is a class-attribute + HTTP-
  wrapper task with no agent design.
- **Investigation rounds use the same dispatch path as initial
  gather.** `epistemic_investigate_claim` is now a pure gap-analysis
  agent: it generates 0–3 methodological *intents* (named angles
  along method / population / temporal frame / control / level-of-
  analysis dimensions) and the dispatch agent handles routing. The
  intent layer carries per-round yield-annotated memory and may
  return zero intents to rationally suspend judgment when the search
  space is exhausted.
- **Source-agnostic judging contract, enforced as a stage
  invariant.** `ExtractNewEvidence` judges by predicate (any
  claim-linked, content-bearing Evidence with no `support_judgment`
  is judged) regardless of which path created it. The
  `scrutiny_and_investigation` stage exit invariant
  (`_all_active_claim_evidence_judged`) refuses to proceed if any
  such Evidence remains unjudged — a future creation path that
  bypasses judging fails loudly at the stage boundary rather than
  silently degrading calibration.
- **Filtered resolved uncertainties from the investigation agent's
  input.** Resolved uncertainties no longer leak into
  `scrutiny_issues`, so the agent doesn't re-target gaps that have
  already been closed.

### Benchmarks (dev30, n=20 calibratable)

- **Epistemic (gpt-5.4-nano)**: AUC 0.89, Brier 0.166, ECE 0.167.
- **vs. baseline_frontier (gpt-5.4 alone)**: tied within CIs.
- **vs. rag_replay on the same evidence pool**: Brier reduced from
  0.294 to 0.166 (40% reduction). This is the cleanest
  architecture-attributable claim from the run.

See `docs/results/dev30_v9.md` for the full table, methodology, and
honest discussion of what the run does and does not support.

### Removed

- The legacy three-agent evidence-gathering chain
  (`epistemic_select_provider`, `epistemic_rank_providers` for
  initial gather, `epistemic_formulate_query`).
- `PlanTaskOperation` and the `_run_provider_tournament` helper.
- The `dispatch_mode` toggle on `run_epistemic_graph` (single
  routing path now; no switch needed).
- The `PROVIDER_QUERY_GUIDANCE` and `PROVIDER_DESCRIPTIONS`
  module-level registry dicts. Provider data lives on the provider
  class.
- `get_source_catalogue` helper (orphaned).
- The orphaned `should_flag_for_review` helper in
  `adversarial_balance.py`.
- Three obsolete test files (`test_provider_tournament.py`,
  `test_phase2_lazy_planning.py`, `routing_benchmark_queries.py`).

### Added

- `andamentum.epistemic.dispatch` module — description-driven
  dispatch implementation (`DispatchResult`,
  `formulate_provider_query`, `gather_evidence_new`).
- `andamentum.epistemic.operations.dispatch_gather` —
  `DispatchGatherOperation` (initial gather) and the shared
  `dispatch_and_persist_for_text` helper used by both initial gather
  and investigation rounds.
- `andamentum.epistemic.entities.intent_record` — `IntentRecord`
  Pydantic model carrying intent text + per-intent yield count.
- `Claim.investigation_intents: list[IntentRecord]` — per-claim
  memory of follow-up search angles tried in prior rounds.
- `epistemic_dispatch_provider` agent (one per-provider call;
  receives `claim` + optional `angle`).
- `epistemic_investigate_claim` agent — rewritten to output intents
  with the dimension-shift discipline, replacing the prior
  query-generation output.
- Stage invariant `_all_active_claim_evidence_judged` enforced at
  the `scrutiny_and_investigation` boundary.
- New dispatch-quality benchmark harness at
  `benchmarks/epistemic/dispatch_quality/` (Tier 1 triage accuracy
  per provider with mocked-LLM unit tests).
- Smoke-test artefacts at `docs/results/dev30_v9.md`.
- Architecture summary at `docs/architecture.md`.

### Compatibility

This release is a behaviour change relative to `0.2.0`. Callers of
`run_epistemic_graph(...)` should:

- **Remove** any `dispatch_mode=` kwarg (the parameter no longer
  exists; the behaviour the old `"new"` value selected is now the
  default and only path).
- **Be aware** that `Claim.investigation_intents` is now
  `list[IntentRecord]` rather than `list[str]`; old database
  representations load via `IntentRecord.from_dict` with
  `evidence_count` defaulting to 0 for legacy rows.

No other public API changes.

### Canonical green state

- pyright: 23 errors (pre-existing test-only typing noise —
  pydantic-graph generic variance in `test_topology.py`,
  `Decomposition` dict-form fixtures in several test files).
- ruff: clean.
- pytest: 2075 passing, 2 skipped, 25 deselected.
