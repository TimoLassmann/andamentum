# CLAUDE.md

Follow the rules in @CONSTITUTION.md. Project-specific rules below override or extend them where explicitly stated.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Andamentum is a single Python package (`src/andamentum/`) of tightly-scoped sub-modules for building agentic reasoning pipelines. The core three were consolidated from separate packages; the commit history around that migration is relevant context.

- `andamentum.epistemic` — formal epistemology: evidence/claims/uncertainty entities, deterministic stage gates, pattern-driven work scheduling, multi-agent verification
- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`
- `andamentum.document_store` — SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal Reciprocal Rank Fusion search and LLM metadata extraction
- `andamentum.whetstone` — structured multi-lens document review over your own drafts. `await review_document(source, *, model)` is the single entry point: pydantic-graph driven, deterministic structural substrate plus single-job LLM agents. Returns a `ReviewResult` with confidence-tagged `Finding`s, concrete `Edit`s (when `editor=True`), `AuthorQuestion`s, and a synthesised `summary`. Three renderers (`render_markdown`, `render_html`, `render_docx`) consume the same result; the docx renderer feeds an in-tree track-changes machinery (`whetstone.docx`) via a thin Edit/Finding→DocumentPatch adapter. Beyond the basic critical-review pipeline: panel mode (multi-expert review), guidelines mode (`--guidelines @file`), custom criteria mode (`--criteria`), statistical self-consistency check (statcheck-equivalent), claim → evidence anchoring lens, novelty / prior-work check via deep_research, and overclaim ("reviewer 2 bait") detection.
- `andamentum.scribe` — structured document drafting: block-based authoring (paragraph, heading, figure, table), section abstraction, built-in `article`/`grant` scaffolds, SQLite-backed source of truth, one-way render to `.docx`. Replaces the standalone `document-tools:doc-draft` plugin.
- `andamentum.figures` — publication-quality scientific figure generation: 9 chart types, 7 journal palettes, journal-matched sizing, auto chart-kind selection. `scribe_glue.insert_figure` renders + inserts into a scribe section in one call. Absorbed from the standalone `mosaic-figures` package.
- `andamentum.chunker` — structural-first semantic chunking of long markdown into 2k–10k char self-contained units. Stage 1: split at markdown headings. Stage 2: embedding-based split for over-budget sections. Stage 3 (optional): small-LLM judge for grey-zone boundaries. The LLM is never the primary segmenter.
- `andamentum.harvest` — universal source → markdown extraction. Single async API: `extract(source: str | Path) -> str`. Detects format (PDF / HTML / DOCX / PPTX / Markdown / plain) and dispatches to the best backend; for HTML, sniffs `og:type` / JSON-LD `@type` and routes article-like pages to trafilatura, index/listing pages to Docling, and races both extractors when the page metadata is ambiguous (picks the higher-scoring output by structural quality).
- `andamentum.typeset` — standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules for rendering
- `andamentum.core` — shared model-resolution, `AgentRunner`, and (future) embedding infrastructure used by all sub-modules

Everything ships in one distribution. There are no optional extras — dependencies are the flat union of what the sub-modules need.

## Commands

Use `uv run` for everything Python-related (not plain `python`).

```bash
# Install dev deps
uv sync --extra dev

# Run the full test suite (814 tests default, asyncio_mode=auto)
# The `ollama` and `benchmark` markers are deselected by default.
uv run pytest

# Run the 200-query semantic routing benchmark (requires live Ollama)
uv run pytest -m benchmark -s

# Run tests for a single sub-module
uv run pytest src/andamentum/epistemic/tests
uv run pytest src/andamentum/deep_research/tests
uv run pytest src/andamentum/document_store/tests

# Run a single test file / test
uv run pytest src/andamentum/epistemic/tests/test_gates.py
uv run pytest src/andamentum/epistemic/tests/test_gates.py::test_name -x

# Type check (pyright, targets Python 3.13 per pyproject.toml)
uv run pyright

# Lint
uv run ruff check
uv run ruff format

# Build distribution
uv build
```

The canonical green state: **pyright 0 errors, ruff clean, pytest 1560 passing (1 skipped, 6 deselected)**. Test count grew through 2026-04 with the whetstone v2 feature-parity work (deterministic checklist, panel mode, guidelines+custom modes, consistency lens, statcheck, claim→evidence anchoring, novelty check, overclaim lens) and dropped by 124 when v1 was decommissioned. Don't claim completion until you've run these three and seen that state.

## CLIs

Six scripts installed by the package:

```bash
andamentum-epistemic --help
andamentum-research --help
andamentum-whetstone --help
andamentum-scribe --help
andamentum-figures --help
andamentum-chunker --help
```

`andamentum-epistemic`, `andamentum-research`, and `andamentum-chunker` require a model via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`, routed through `core.models.resolve_model_from_args`, which `sys.exit(1)`s if neither is set — no hidden defaults. `andamentum-whetstone` takes `--model` directly (any pydantic-ai model id) and supports a `--no-llm` flag for the deterministic-only path. `andamentum-scribe` and `andamentum-figures` have no LLM dependency.

## Architectural conventions

**Explicit model argument, no hidden defaults.** Every public function that calls an LLM takes `model=` as a keyword-only argument. There is no shared config module, no silent fallback, no ambient default. When adding new LLM-calling code, match this pattern — don't reach for a global config.

**Core module** (`andamentum.core`) — shared infrastructure for model resolution, agent execution, and (future) embeddings. All sub-modules import from core instead of maintaining independent implementations. When adding LLM-calling code, use `core.agents.AgentRunner` or `core.agents.run_agent_with_fallback()` — they provide model resolution (ollama, bedrock, passthrough) and PromptedOutput fallback for free.

**Layering:**
- `core` is the shared base — no sub-module imports from another sub-module via core's surface (core itself depends on neither).
- `document_store` is foundational; `epistemic` may depend on it directly (e.g., `EpistemicRepository` wraps a `DocumentStore`).
- `deep_research` is foundational for evidence gathering; `epistemic.evidence_gathering` may depend on it.
- `document_store` and `deep_research` MUST NOT depend on `epistemic` or on each other.
- `whetstone` depends on `core` (for `AgentRunner`/`AgentDefinition`), on `typeset` for HTML rendering, on `chunker` for section splitting, on `harvest` for source ingestion, and on `deep_research` (only when the opt-in novelty check is enabled — `--check-novelty` / `check_novelty=True`; otherwise the import is deferred at runtime). It MUST NOT depend on `epistemic` or `document_store`.
- `scribe` depends only on `typeset` (for HTML/PDF rendering) and stdlib `sqlite3`. MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `figures`, or `core`.
- `figures` depends only on matplotlib + numpy + pydantic. The optional `figures.scribe_glue` submodule is the ONLY place where `scribe` is imported; the rest of `figures` MUST NOT touch `scribe`. `figures` MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `typeset`, or `core`.
- `chunker` depends only on `core` (for `AgentRunner`, model resolution) and `rapidfuzz` (for tiered anchor matching). MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `scribe`, `figures`, or `typeset`. Other modules MAY depend on `chunker` (e.g. whetstone for section-by-section review on huge documents, document_store for embedding-quality chunks).
- `harvest` depends only on `httpx`, `trafilatura`, `docling`, and stdlib. MUST NOT depend on any other andamentum sub-module — it's a leaf service. Other modules MAY depend on `harvest` to convert URLs/files to markdown before further processing (e.g. chunker, whetstone, future deep_research consolidation).

**Public API lives in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal. `document_store` additionally re-exports from `public.py` — that module is the authoritative public surface for document_store (10 functions: `ingest`, `search`, `find_by_metadata`, `update_metadata`, `delete`, `restore`, `purge`, `list_deleted`, `repair`, `find_duplicates`).

**Evidence providers** follow a strict specification documented in `src/andamentum/epistemic/providers/CONTRIBUTING.md`. Read it before adding or modifying any provider. Key rules: providers retrieve and structure evidence, never assess quality (`quality_score=None` always), never truncate content, and return `list[GatheredEvidence]` (empty list on error, never raise).

**Epistemic architecture principles** (enforced across the module):
- **P1: Operations are pure transforms.** An operation reads entities, does work (LLM calls, computations), and writes the result back. It NEVER manipulates fields on other entities to signal what should happen next.
- **P2: The graph is the sole flow controller.** Only graph nodes (in `graph/nodes.py`) decide what runs next, based on operation results, entity state, and graph state.
- **P3: Entity fields are data, not signals.** Every field on Claim, Evidence, Objective represents something real — a verdict, a score, a stage. No field exists solely to tell the scheduler what to do.
- **P4: Graph state tracks pipeline progress.** `EpistemicGraphState` (in `graph/state.py`) tracks what work has been done and what needs doing — not entity fields.
- **P5: Operations don't reach across entity boundaries.** An operation on an Uncertainty does not modify Claims. Cross-entity effects are the graph's job.

**Epistemic core abstractions** (understand these before touching `epistemic/`):
- `entities/` — `Objective`, `Evidence`, `Claim`, `Uncertainty`, `Decision`, `Snapshot`, `Artefact` (all `EpistemicEntity` subclasses). `Objective.claim_to_verify` enables seed-claim verification mode.
- `gates.py` — `STAGE_GATES` + `validate_promotion`: deterministic, routing-aware checks that must pass before a `Claim` advances stages. Gates query the question type's routing profile and only require tracks that are PRIMARY or SECONDARY — not SKIP.
- `graph/` — pydantic-graph DAG scheduler. 15 nodes with typed return edges. Operations execute in explicit dependency order. Entry point: `run_epistemic_graph()` in `graph/__init__.py`.
- `operations/` — pure `BaseOperation` subclasses. Each takes an `OperationInput`, does work, returns `OperationResult`. They do NOT control flow — the graph does.
- `repository.py` — `EpistemicRepository` wraps a `StorageBackend` (in-memory backend ships in `storage.py`)
- `runner.py` — `DefaultAgentRunner` wraps `core.AgentRunner` with epistemic agent registry lookup.

**Deep research pipeline** is a `pydantic-graph` state machine. `state.py` holds `ResearchState`; `graph.py` / `nodes.py` define the nodes; `orchestrator.py` / `runner.py` drive execution. `searxng.py` manages the local SearxNG instance and `circuit_breaker.py` wraps it. Content extraction (`content_extractor.py`) uses `trafilatura` for HTML and `docling` for PDF. Search-query production runs through a per-slot generate→verify loop (`PrepareSearchCycle` → `GenerateOne` ⇄ `Verify` → `ParallelSearch`): the `query_generator` agent emits one query at a time, the `topic_verifier` agent judges it against the goal, rejected queries get up to `MAX_SLOT_RETRIES = 3` retries with verifier feedback before skip-and-tighten kicks in. The previous regex+stopword "topic guard" in `text_utils.py` has been retired.

**Document store** is SQLite-first. Databases live in `~/.local/share/document-store/{name}.db` (override with `DOCUMENT_STORE_DIR`; legacy `ANDAMENTUM_DATABASES_DIR` is also honoured). Ingestion is two-phase: document registered immediately (FTS5-searchable), chunks + embeddings written in a background pass that `repair()` can resume after a crash. Search fuses four signals via RRF: FTS5 keyword, chunk embeddings, doc embeddings, and DHP (temporal clustering, see `dhp.py`). Requires Ollama running locally with `embeddinggemma:latest` for embeddings.

**Typeset module** (`andamentum.typeset`) — a standalone typesetting system with 7 visual atoms (`heading`, `prose`, `callout`, `items`, `aside`, `card`, `reference`), 3 named styles (`article`, `cv`, `report`), and HTML + PDF output. Used by the epistemic report adapter (`typeset_report.py`, the only HTML path now that the legacy `html_report.py` has been retired), by `whetstone` for its HTML review output, and by `scribe` for its HTML/PDF render path. The epistemic data schema (the dataclasses consumed by the adapter) lives in `epistemic/report_data.py`. See `src/andamentum/typeset/USAGE.md` for the full API reference.

**Whetstone module** (`andamentum.whetstone`) — structured multi-lens feedback over drafts the user wrote themselves. Single entry point: `await review_document(source, *, model)` returns a `ReviewResult` with confidence-tagged `Finding`s, concrete `Edit`s (when `editor=True`), `AuthorQuestion`s, and a synthesised `summary`. Implementation is a pydantic-graph DAG (`graph.py`, `nodes/`, `state.py`, `deps.py`) over a deterministic structural substrate (`structural/`) plus single-job pydantic-ai agents (`agents/`). Renderers (`render_markdown`, `render_html`, `render_docx`) in `renderers/` consume the same `ReviewResult`; the DOCX renderer adapts each `Edit`/`Finding` into a `DocumentPatch` (defined in `models.py`) and feeds the in-tree track-changes machinery in `whetstone.docx`. Beyond the basic critical-review pipeline: panel mode (multi-expert review), guidelines mode (`--guidelines @file`), custom criteria mode (`--criteria`), statistical self-consistency (statcheck-equivalent, in `structural/stat_consistency.py`), claim → evidence anchoring lens, novelty / prior-work check via `deep_research`, and overclaim ("reviewer 2 bait") detection. The legacy v1 surface (`sharpen_document`, `consistency_scanners`, `checklist_scanners`, `dynamic_models`, `orchestrator.py`, the v1 `agents/` and `renderers/`) was removed once feature parity was reached.

**Figures module** (`andamentum.figures`) — publication-quality scientific figure generation. High-level entry point: `figure(data, *, kind="auto", style="npg", journal="default", output="figure.pdf")`. Lower-level building blocks: `setup_style`, `get_palette`, `savefig`, `panel_label`, `shared_legend`, `despine` (in `style.py`); plot primitives in `plots.py`; auto-detection (chart kind, log scale, column roles) in `auto.py` and `advisor.py`; bootstrap stats in `stats.py`. The package was absorbed from `mosaic-figures` (uninstall the standalone tool with `uv tool uninstall mosaic-figures` once this branch lands). Scribe integration lives in `figures/scribe_glue.py`: `insert_figure(doc, section, *, output_dir, caption, label, **chart_kwargs)` renders a PNG and inserts a `Figure` block via `Document.insert_into_section`.

**Chunker module** (`andamentum.chunker`) — structural-first semantic chunker. Single entry point: `extract_units(source, *, target_min_chars=2000, target_max_chars=10000, embedding_fn=None, judge_executor=None, domain="general") -> ChunkingResult`. Pipeline (literature consensus, 2026): **Stage 1** — split at markdown headings (`structural.py`); **Stage 2** — for sections >`target_max_chars`, semantic-split at paragraph boundaries chosen by largest cosine drops between adjacent paragraph embeddings (`semantic_split.py`, defaults to local Ollama `embeddinggemma:latest` via `embeddings.py`); **Stage 3** (optional) — small LLM judge (`judge.py`) for cuts whose drop-percentile is in the grey zone (60–90th by default), answers `keep | merge`. The LLM is never the primary segmenter. Each unit's `text` is byte-identical to a source span. The legacy agentic `windowing.py`/`refinement.py`/`NextUnitResult` machinery has been removed; legacy kwargs (`primary_executor`, `window_size`, `lookahead`, `extension_chars`, `max_iterations`) are accepted on `extract_units` for backward compat but ignored.

**Harvest module** (`andamentum.harvest`) — universal source → markdown extraction. Single entry point: `await extract(source: str | Path) -> str`. Resolves URLs (httpx fetch with SSRF protection) and file paths to bytes + format (PDF / HTML / DOCX / PPTX / Markdown / plain via three-tiered detection: extension → MIME → magic-byte sniff). For PDF / DOCX / PPTX uses Docling. For HTML, sniffs `og:type` / JSON-LD `@type` / `<article>` tag (`metadata.py`) and routes article-tagged pages to trafilatura, listing/index pages (`@type=WebPage|CollectionPage|ItemList|...`) to Docling. When metadata is ambiguous, races both extractors concurrently and picks the higher-scoring output (`scoring.py` weights heading count × 10 + paragraph breaks × 1 + char count × 0.001, penalises >5% link density, disqualifies output with zero headings AND zero `\n\n`). Loud failure model — `HarvestError` / `FetchError` / `UnsupportedFormatError` / `ExtractionError` are typed; the function never silently returns empty. Extraction backends in `backends/` are thin async adapters with the same `extract(data, source_url) -> str` shape so the orchestrator can race them generically.

**Scribe module** (`andamentum.scribe`) — block-based document authoring. Documents live in SQLite at `~/.local/share/scribe/<name>.db` (override with `SCRIBE_DIR`). Public entry point: `Document.create(title=..., database=..., scaffold="article" | "grant" | None)`; mutate with `append`/`replace`/`replace_section`; render with `render(path, format="docx")`. Section operations (`list_sections`, `section`, `replace_section`) are derived from heading blocks — there is no separate sections table. Each block has an integer revision counter; `replace()` enforces optimistic locking via `BEGIN IMMEDIATE` and writes an audit row to `scribe_revisions`. Citations are Pandoc-flavoured `[@key]` spans extracted by regex; references live in their own table; `[verify]` and `[citation needed]` markers are recognised and reported by `validate()`. Inline markdown (bold/italic/code) renders as styled runs in `.docx`. HTML/PDF rendering goes through `typeset` (block→atom mapping in `render_typeset.py`). Scribe replaces the standalone `document-tools:doc-draft` plugin for Word file authoring; `.pptx` stays out of scope.

## Working in git worktrees

When you need a worktree (e.g. for executing a multi-step plan in isolation), create it **inside this repo at `.worktrees/<feature-name>/`** — never as a sibling directory like `../andamentum-<feature>/`. Sibling worktrees fall outside Claude Code's permission scope and force the user to manually approve every single tool call.

`.worktrees/` is gitignored, so the nested worktrees don't pollute git status. Standard commands:

```bash
git worktree add .worktrees/<feature-name> -b <branch-name>
# ... do the work ...
git worktree remove .worktrees/<feature-name>
git branch -d <branch-name>  # after merging back
```

## Known quirks

- `pytest.ini_options.testpaths = ["src/andamentum"]` — tests live next to the code they test, not in a top-level `tests/` directory.
- `asyncio_mode = "auto"` — async tests don't need `@pytest.mark.asyncio`.
- Tests marked `@pytest.mark.ollama` or `@pytest.mark.benchmark` are deselected by default (`addopts = "-m 'not ollama and not benchmark'"`). Run explicitly with `uv run pytest -m benchmark`.
- WeasyPrint's flex layout breaks when `<p>` tags are inside flex children. The typeset renderer strips `<p>` wrapping from item and reference bodies as a workaround.
