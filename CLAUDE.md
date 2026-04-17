# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Andamentum is a single Python package (`src/andamentum/`) containing three tightly-scoped sub-modules for building agentic reasoning pipelines. It was consolidated from three separate packages; the commit history around that migration is relevant context.

- `andamentum.epistemic` — formal epistemology: evidence/claims/uncertainty entities, deterministic stage gates, pattern-driven work scheduling, multi-agent verification
- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`
- `andamentum.document_store` — SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal Reciprocal Rank Fusion search and LLM metadata extraction

Everything ships in one distribution. There are no optional extras — dependencies are the flat union of what the three sub-modules need.

## Commands

Use `uv run` for everything Python-related (not plain `python`).

```bash
# Install dev deps
uv sync --extra dev

# Run the full test suite (764 tests default, asyncio_mode=auto)
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

The canonical green state: **pyright 0 errors, ruff clean, pytest 764 passing (1 benchmark deselected)**. Don't claim completion until you've run these three and seen that state.

## CLIs

Two scripts installed by the package:

```bash
andamentum-epistemic --help
andamentum-research --help
```

Both require a model, either via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`. There are no hidden defaults — if a model isn't provided, the CLI exits with an error.

## Architectural conventions

**Explicit model argument, no hidden defaults.** Every public function that calls an LLM takes `model=` as a keyword-only argument. There is no shared config module, no silent fallback, no ambient default. When adding new LLM-calling code, match this pattern — don't reach for a global config.

**Sub-modules are independent within one package.** `epistemic`, `deep_research`, and `document_store` do not import from each other. Treat them as three libraries that happen to share a distribution. When something would need to cross the boundary, stop and ask.

**Public API lives in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal. `document_store` additionally re-exports from `public.py` — that module is the authoritative public surface for document_store (10 functions: `ingest`, `search`, `find_by_metadata`, `update_metadata`, `delete`, `restore`, `purge`, `list_deleted`, `repair`, `find_duplicates`).

**Epistemic core abstractions** (understand these before touching `epistemic/`):
- `entities/` — `Objective`, `Evidence`, `Claim`, `Uncertainty`, `Decision`, `Snapshot`, `Artefact` (all `EpistemicEntity` subclasses). `Objective.claim_to_verify` enables seed-claim verification mode.
- `gates.py` — `STAGE_GATES` + `validate_promotion`: deterministic, routing-aware checks that must pass before a `Claim` advances stages. Gates query the question type's routing profile and only require tracks that are PRIMARY or SECONDARY — not SKIP.
- `patterns.py` — `PatternScheduler` + `WORK_PATTERNS`: pattern-driven work scheduling with per-operation budgets. Phase 4 has two mutually exclusive claim creation modes: `seed_claim` (verification mode when `claim_to_verify` is set) and `propose_claims` (research mode when it's None).
- `provider_routing.py` — semantic provider selection via embedding cosine similarity. Replaces the old keyword-based `DOMAIN_PROVIDER_MAP`. Benchmarked at 97.5% top-3 recall across 200 queries.
- `operations/` — `BaseOperation` subclasses that agents dispatch to; registered via `OPERATION_CLASSES` / `create_operations`. Includes `SeedClaimOperation` for verification mode.
- `repository.py` — `EpistemicRepository` wraps a `StorageBackend` (in-memory backend ships in `storage.py`)
- `runner.py` — `DefaultAgentRunner` is lazy-imported to keep `pydantic-ai` off the critical import path. Handles `ollama:` prefix by constructing `OllamaProvider` with default `OLLAMA_BASE_URL`.

**Deep research pipeline** is a `pydantic-graph` state machine. `state.py` holds `ResearchState`; `graph.py` / `nodes.py` define the nodes; `orchestrator.py` / `runner.py` drive execution. `searxng.py` manages the local SearxNG instance and `circuit_breaker.py` wraps it. Content extraction (`content_extractor.py`) uses `trafilatura` for HTML and `docling` for PDF.

**Document store** is SQLite-first. Databases live in `~/.local/share/document-store/{name}.db` (override with `DOCUMENT_STORE_DIR`; legacy `ANDAMENTUM_DATABASES_DIR` is also honoured). Ingestion is two-phase: document registered immediately (FTS5-searchable), chunks + embeddings written in a background pass that `repair()` can resume after a crash. Search fuses four signals via RRF: FTS5 keyword, chunk embeddings, doc embeddings, and DHP (temporal clustering, see `dhp.py`). Requires Ollama running locally with `embeddinggemma:latest` for embeddings.

**Typeset module** (`andamentum.typeset`) — a standalone typesetting system with 7 visual atoms (`heading`, `prose`, `callout`, `items`, `aside`, `card`, `reference`), 3 named styles (`article`, `cv`, `report`), and HTML + PDF output. Used by the epistemic report adapter (`typeset_report.py`) for side-by-side comparison with the legacy `html_report.py` renderer. See `src/andamentum/typeset/USAGE.md` for the full API reference.

## Known quirks

- `pytest.ini_options.testpaths = ["src/andamentum"]` — tests live next to the code they test, not in a top-level `tests/` directory.
- `asyncio_mode = "auto"` — async tests don't need `@pytest.mark.asyncio`.
- Tests marked `@pytest.mark.ollama` or `@pytest.mark.benchmark` are deselected by default (`addopts = "-m 'not ollama and not benchmark'"`). Run explicitly with `uv run pytest -m benchmark`.
- WeasyPrint's flex layout breaks when `<p>` tags are inside flex children. The typeset renderer strips `<p>` wrapping from item and reference bodies as a workaround.
