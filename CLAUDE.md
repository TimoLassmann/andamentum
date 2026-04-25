# CLAUDE.md

Follow the rules in @CONSTITUTION.md. Project-specific rules below override or extend them where explicitly stated.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Andamentum is a single Python package (`src/andamentum/`) of tightly-scoped sub-modules for building agentic reasoning pipelines. The core three were consolidated from separate packages; the commit history around that migration is relevant context.

- `andamentum.epistemic` — formal epistemology: evidence/claims/uncertainty entities, deterministic stage gates, pattern-driven work scheduling, multi-agent verification
- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`
- `andamentum.document_store` — SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal Reciprocal Rank Fusion search and LLM metadata extraction
- `andamentum.whetstone` — structured multi-lens document review over your own drafts (grammar/style edits, specialist critique, multi-expert panel); output as Word track-changes, HTML, or markdown diff
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

The canonical green state: **pyright 0 errors, ruff clean, pytest 814 passing (1 benchmark deselected)**. Test count reflects removal of 75 pattern-scheduler tests after graph migration. Don't claim completion until you've run these three and seen that state.

## CLIs

Three scripts installed by the package:

```bash
andamentum-epistemic --help
andamentum-research --help
andamentum-whetstone --help
```

All three CLIs require a model via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`, routed through `core.models.resolve_model_from_args`, which `sys.exit(1)`s if neither is set — no hidden defaults.

## Architectural conventions

**Explicit model argument, no hidden defaults.** Every public function that calls an LLM takes `model=` as a keyword-only argument. There is no shared config module, no silent fallback, no ambient default. When adding new LLM-calling code, match this pattern — don't reach for a global config.

**Core module** (`andamentum.core`) — shared infrastructure for model resolution, agent execution, and (future) embeddings. All sub-modules import from core instead of maintaining independent implementations. When adding LLM-calling code, use `core.agents.AgentRunner` or `core.agents.run_agent_with_fallback()` — they provide model resolution (ollama, bedrock, passthrough) and PromptedOutput fallback for free.

**Layering:**
- `core` is the shared base — no sub-module imports from another sub-module via core's surface (core itself depends on neither).
- `document_store` is foundational; `epistemic` may depend on it directly (e.g., `EpistemicRepository` wraps a `DocumentStore`).
- `deep_research` is foundational for evidence gathering; `epistemic.evidence_gathering` may depend on it.
- `document_store` and `deep_research` MUST NOT depend on `epistemic` or on each other.
- `whetstone` depends only on `core` (for `AgentRunner`/`AgentDefinition`) and optionally on `typeset` for HTML rendering. It MUST NOT depend on `epistemic`, `deep_research`, or `document_store`.

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

**Deep research pipeline** is a `pydantic-graph` state machine. `state.py` holds `ResearchState`; `graph.py` / `nodes.py` define the nodes; `orchestrator.py` / `runner.py` drive execution. `searxng.py` manages the local SearxNG instance and `circuit_breaker.py` wraps it. Content extraction (`content_extractor.py`) uses `trafilatura` for HTML and `docling` for PDF.

**Document store** is SQLite-first. Databases live in `~/.local/share/document-store/{name}.db` (override with `DOCUMENT_STORE_DIR`; legacy `ANDAMENTUM_DATABASES_DIR` is also honoured). Ingestion is two-phase: document registered immediately (FTS5-searchable), chunks + embeddings written in a background pass that `repair()` can resume after a crash. Search fuses four signals via RRF: FTS5 keyword, chunk embeddings, doc embeddings, and DHP (temporal clustering, see `dhp.py`). Requires Ollama running locally with `embeddinggemma:latest` for embeddings.

**Typeset module** (`andamentum.typeset`) — a standalone typesetting system with 7 visual atoms (`heading`, `prose`, `callout`, `items`, `aside`, `card`, `reference`), 3 named styles (`article`, `cv`, `report`), and HTML + PDF output. Used by the epistemic report adapter (`typeset_report.py`) for side-by-side comparison with the legacy `html_report.py` renderer, and by `whetstone` for its HTML review output. See `src/andamentum/typeset/USAGE.md` for the full API reference.

**Whetstone module** (`andamentum.whetstone`) — structured multi-lens feedback over drafts the user wrote themselves. Entry point: `sharpen_document(text, task=...)` returns a `ReviewResult` holding `DocumentPatch` edits and `DocumentIssue` findings. Scanners live in `consistency_scanners.py` and `checklist_scanners.py`; the agent registry is in `agents/`; renderers (`render_docx`, `render_html`, `render_diff`, `apply_patches`) in `renderers/`. The DOCX renderer emits track-changes Word output and prepends a checklist section when checklist tasks are present.

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
