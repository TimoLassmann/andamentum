# CLAUDE.md

Follow the rules in @CONSTITUTION.md. Project-specific rules below override or extend them where explicitly stated.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Andamentum is a single Python package (`src/andamentum/`) of tightly-scoped sub-modules for building agentic reasoning pipelines. The core three were consolidated from separate packages; the commit history around that migration is relevant context.

- `andamentum.epistemic` — formal epistemology: evidence/claims/uncertainty entities, deterministic stage gates, pattern-driven work scheduling, multi-agent verification
- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`
- `andamentum.document_store` — SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal Reciprocal Rank Fusion search and LLM metadata extraction
- `andamentum.whetstone` — structured criterion-cascade review over your own drafts. `await review_document(source, *, model)` is the single entry point: pydantic-graph driven, deterministic substrate (sectionize + claims digest) feeding a one-call-per-criterion review with a gap-loop re-grounding pass, optional editor, and synthesis + self-critique. Returns a `ReviewResult` with confidence-tagged `Finding`s, concrete `Edit`s (when `editor=True`), `AuthorQuestion`s, and a synthesised `summary`. Three renderers (`render_markdown`, `render_html`, `render_docx`) consume the same result; the docx renderer feeds an in-tree track-changes machinery (`whetstone.docx`) via a thin Edit/Finding→DocumentPatch adapter. Document type routes to one of six pluggable criterion sets (academic / external_communication / essay / tutorial / creative / general) auto-detected via a one-shot classifier or set explicitly. Optional novelty / prior-work check via deep_research (3-node pipeline: flag → search → judge). Panel mode (multi-expert simulated peer review) lives in its own graph (`whetstone.v3.panel`) and is invoked via the `andamentum-whetstone panel` subcommand. Style / readability checks live in the separate `andamentum.proofread` module — the two are deliberately disjoint pipelines.
- `andamentum.scribe` — structured document drafting: block-based authoring (paragraph, heading, figure, table), section abstraction, built-in `article` scaffold (no grant scaffold — funder AI-disclosure rules are tightening and we don't want to make AI-assisted grant drafting more frictionless), SQLite-backed source of truth, one-way render to `.docx`. Replaces the standalone `document-tools:doc-draft` plugin.
- `andamentum.figures` — publication-quality scientific figure generation: 9 chart types, 7 journal palettes, journal-matched sizing, auto chart-kind selection, and auto bar orientation (long-categorical-label bar charts auto-flip to horizontal — overridable with `horizontal=True/False`). `scribe_glue.insert_figure` renders + inserts into a scribe section in one call. Absorbed from the standalone `mosaic-figures` package.
- `andamentum.chunker` — structural-first semantic chunking of long markdown into 2k–10k char self-contained units. Stage 1: split at markdown headings. Stage 2: embedding-based split for over-budget sections. Stage 3 (optional): small-LLM judge for grey-zone boundaries. The LLM is never the primary segmenter.
- `andamentum.harvest` — universal source → markdown extraction. Single async API: `extract(source: str | Path) -> str`. Detects format (PDF / HTML / DOCX / PPTX / Markdown / plain) and dispatches to the best backend; for HTML, sniffs `og:type` / JSON-LD `@type` and routes article-like pages to trafilatura, index/listing pages to Docling, and races both extractors when the page metadata is ambiguous (picks the higher-scoring output by structural quality).
- `andamentum.vision_critique` — bounded vision critique of rendered figures. Single async API: `await critique_figure(image, *, model) -> FigureCritique`. Image is bytes / Path / http(s) URL; model is any pydantic-ai multimodal id (Ollama, Anthropic, OpenAI). Schema is a tight pydantic model with close-set enums (`label_overlap`, `labels_legible`, `legend_blocks_data`, `aspect_ratio_issue`, `suggested_fixes` from a fixed `Literal` set, `confidence`, `one_line_summary`) so small local vision models can fill it reliably. Validated default: `ollama:gemma4:e4b-it-q4_K_M` (passes the calibration fixture; `gemma4:e2b` confidently fails it). Used standalone, by `figures` for figure refinement, or by other modules that want to lint rendered output.
- `andamentum.proofread` — deterministic, dependency-light proofreading. Single sync API: `analyze(text: str) -> ProofreadResult`. Combines five readability scores (SMOG, Flesch–Kincaid grade, Flesch reading ease, Gunning Fog, Coleman–Liau, ARI) with classic style checks (weasel words à la Matt Might, passive voice via be-verb + past-participle heuristic, duplicate words, weak sentence openers, -ly adverb density). No LLM, no I/O, pure function. Designed to complement `whetstone`'s LLM-driven review, not replace it. Findings carry char offsets and `sentence_index` so callers can map back into the source. The CLI (`andamentum-proofread`) accepts URLs and document files via `harvest.extract` — the only point where the module touches another sub-module.
- `andamentum.typeset` — standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules for rendering
- `andamentum.agentic_dialect` — the house style for building agentic graph systems (pydantic-graph + pydantic-ai), as data + portable conformance checks. The prose canon is `src/andamentum/agentic_dialect/DIALECT.md` (3 surfaces / 2 roles / 1 agent / 8 laws); the structured laws, role briefs (`for_role`), checklist, and skeleton are the enforceable kernel, drift-tested against the doc. `check_code(path)` runs the portable static gates (no engine import in workers, no `dict[str, Any]`, no nondeterminism/client/literal-bound in node bodies). A leaf module: depends only on `pydantic` + stdlib, imports no other sub-module, relative imports only — kept independently extractable.
- `andamentum.core` — shared model-resolution, `AgentRunner`, and embedding infrastructure (`core.embeddings`) used by all sub-modules

Everything ships in one distribution. The default dependencies are the flat union of what the sub-modules need. Two optional extras add heavier, self-contained capabilities with graceful fallback when absent: `html-articles` (trafilatura, Apache-2.0 — falls back to docling) and `pdf` (WeasyPrint, BSD-3 — `typeset` raises a clear ImportError without it).

## Commands

Use `uv run` for everything Python-related (not plain `python`).

```bash
# Install dev deps
uv sync --extra dev

# Run the full test suite (asyncio_mode=auto; see the canonical green state below)
# The `ollama`, `benchmark`, `cloud`, and `live_provider` markers are deselected by default.
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

The canonical green state: **pyright 30 errors (pre-existing test-only typing noise — pydantic-graph generic variance in `test_topology.py`, `Decomposition` dict-form fixtures in several test files; zero errors in shipped `src/` code), `ruff check` clean (note: `ruff format` still wants to reformat ~110 pre-existing files — a known cleanup, run `uv run ruff format` once and commit), pytest 2234 passing (2 skipped, 25 deselected)**. Don't claim completion until you've run these three and seen that state. (History: the count moved 2075 → 2152 across the whetstone-v3 consolidation — v2 deletion removed ~380 v2-only tests, v3 added ~460 — then release-prep hardening added the SSRF-redirect / typeset-escaping / size-cap / partial-coverage / panel-confidentiality regression tests to reach 2234.)

## CLIs

Ten scripts installed by the package. Run `--help` on any binary for the exhaustive flag reference; the table below is the at-a-glance map of what to reach for.

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-epistemic` | Formal-epistemology pipeline. Two modes: `ask "<question>"` (research mode — system attempts decomposition, falls back to open research if the question doesn't decompose) or `verify "<claim>"` (single-claim verification, SciFact-style — seeds exactly one Claim from the provided text, skips decomposition) | required |
| `andamentum-research` | Web-research pipeline (search → fetch → extract → verify → synthesise) over a local SearxNG | required |
| `andamentum-whetstone` | Criterion-cascade review of your own draft → markdown / HTML / .docx with track changes. Four subcommands: bare/`review` (default — criterion-cascade review), `panel` (multi-expert simulated peer review), `proofread` (shells out to `andamentum-proofread`), `apply-patches` (apply a pre-built JSON list of `DocumentPatch` objects to a .docx — no LLM) | required for `review` + `panel`; `proofread` + `apply-patches` need no model |
| `andamentum-scribe` | Block-based document authoring backed by SQLite; renders to .docx | none |
| `andamentum-figures` | Publication-quality scientific figures (9 chart types, 7 journal palettes, journal-matched sizing) | none |
| `andamentum-chunker` | Verifiable semantic chunking of long markdown into 2k–10k char self-contained units | required |
| `andamentum-harvest` | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain, auto-detected; for ambiguous HTML, races trafilatura and Docling and picks the better-structured output) | none |
| `andamentum-vision-critique` | Vision-critique of a rendered figure → bounded JSON (label overlap, legibility, legend placement, aspect-ratio, suggested fixes from a fixed set, confidence). Validated local default: `ollama:gemma4:e4b-it-q4_K_M` | required (multimodal) |
| `andamentum-proofread` | Deterministic readability + style check (SMOG, Flesch-Kincaid, weasel words, passive voice, duplicate words, weak openers, adverb density). Accepts URLs, PDF, DOCX, HTML, PPTX, Markdown, plain text (via `harvest.extract`) — or `-` for stdin. `--format human` (default) or `--format json` | none |
| `andamentum-agentic-dialect` | The agentic-systems house style: `laws` / `law <id>` (the 8 laws), `roles` / `role <role>` (the prompt-slice for a job), `checklist`, `skeleton`, `check <path>` (portable conformance gates, nonzero exit on fail) | none |

`andamentum-epistemic`, `andamentum-research`, and `andamentum-chunker` resolve their LLM via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`, routed through `core.models.resolve_model_from_args`, which `sys.exit(1)`s if neither is set — no hidden defaults. `andamentum-whetstone` and `andamentum-vision-critique` take `--model` directly (any pydantic-ai model id; vision-critique requires a multimodal one). `andamentum-scribe`, `andamentum-figures`, `andamentum-harvest`, and `andamentum-proofread` have no LLM dependency.

### Patch-only mode for `andamentum-whetstone`

The patch-only path is the analogue of a standalone "apply review patches to a Word file" script: feed it a .docx and a JSON array of `DocumentPatch` objects (the type defined in `whetstone/models.py`) and it produces a tracked-changes .docx. The renderer reuses the same `whetstone.docx.finalization.finalize_reviewed_document` machinery the full review pipeline uses, so output is byte-identical to running a review that produced those same patches.

```bash
andamentum-whetstone draft.docx \
    --apply-patches patches.json \
    --out draft.reviewed.docx \
    --patch-author "Claude" \
    --patch-report review-summary.md   # optional markdown to prepend
```

`patches.json` is a JSON array; each element is a `DocumentPatch` with `patch_type` ∈ {`text_edit`, `comment`, `document_analysis`}. `text_edit` requires `text_pattern` + `new_text` + `explanation`; `comment` requires `text_pattern` + `comment_text` + `explanation`. This is the foundation for AI-driven draft → review → revise loops where Claude (or another agent) drafts patches separately and then applies them in one tracked-changes pass.

## Architectural conventions

**Explicit model argument, no hidden defaults.** Every public function that calls an LLM takes `model=` as a keyword-only argument. There is no shared config module, no silent fallback, no ambient default. When adding new LLM-calling code, match this pattern — don't reach for a global config.

**Core module** (`andamentum.core`) — shared infrastructure for model resolution, agent execution, embeddings (`core.embeddings`), and SSRF-safe URL fetching (`core.url_safety`). All sub-modules import from core instead of maintaining independent implementations. When adding LLM-calling code, use `core.agents.AgentRunner` or `core.agents.run_agent_with_fallback()` — they provide model resolution (ollama, bedrock, passthrough) and PromptedOutput fallback for free.

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
- `vision_critique` depends on `core` (for `resolve_model`) and `pydantic` / `pydantic-ai` / `httpx`. MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `scribe`, `figures`, `typeset`, `chunker`, or `harvest`. Other modules MAY depend on `vision_critique` (e.g. `figures` for refinement loops, `whetstone` for figure review during manuscript checking).
- `proofread` library depends only on `pydantic` and `textstat`. The optional `proofread.cli` submodule is the ONLY place where `harvest` is imported — so the CLI can accept PDF / HTML / DOCX / PPTX / Markdown sources — and the rest of `proofread` MUST NOT touch `harvest` or any other andamentum sub-module. Other modules MAY depend on `proofread` (e.g. `whetstone` to attach deterministic readability scores to a review, `scribe` for pre-render style flags).
- `agentic_dialect` depends only on `pydantic` + stdlib. MUST NOT import any other andamentum sub-module, and uses only relative imports internally, so it stays independently extractable (a later spin-out to a standalone package is a `git filter-repo`). Other modules MAY depend on `agentic_dialect`. It is governance/knowledge, not a runtime pipeline — nothing in the data pipelines imports it.

**Public API lives in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal. `document_store` additionally re-exports from `public.py` — that module is the authoritative public surface for document_store (11 functions: `ingest`, `search`, `find_by_metadata`, `describe_metadata`, `update_metadata`, `delete`, `restore`, `purge`, `list_deleted`, `repair`, `find_duplicates`). The read side is three uniform functions usable by humans and agents alike: `search` (unstructured content recall), `find_by_metadata` (structured query — scalar value = equality, list value = set-membership `IN`, plus `include_content=False` for cheap metadata-only overviews), and `describe_metadata` (schema discovery over the schema-less metadata — returns per-field `FieldProfile` with value distributions, bounded by `max_values`). The store stays domain-agnostic: consumers define their own metadata vocabulary and build their own tools on these primitives rather than the store growing a function per consumer.

**Evidence providers** follow a strict specification documented in `src/andamentum/epistemic/providers/CONTRIBUTING.md`. Read it before adding or modifying any provider. Key rules: providers retrieve and structure evidence, never assess quality (`quality_score=None` always), never truncate content, and return `list[GatheredEvidence]` (empty list on error, never raise).

**Epistemic architecture principles** (enforced across the module):
- **P1: Operations are pure transforms.** An operation reads entities, does work (LLM calls, computations), and writes the result back. It NEVER manipulates fields on other entities to signal what should happen next.
- **P2: The graph is the sole flow controller.** Only graph nodes (in `graph/nodes.py`) decide what runs next, based on operation results, entity state, and graph state.
- **P3: Entity fields are data, not signals.** Every field on Claim, Evidence, Objective represents something real — a verdict, a score, a stage. No field exists solely to tell the scheduler what to do.
- **P4: Graph state tracks pipeline progress.** `EpistemicGraphState` (in `graph/state.py`) tracks what work has been done and what needs doing — not entity fields.
- **P5: Operations don't reach across entity boundaries.** An operation on an Uncertainty does not modify Claims. Cross-entity effects are the graph's job.
- **P6: Every node has an explicit contract; routing decisions are data, not implicit imperative flow.** Each `Node` subclass (in `graph/base.py`) declares `reads` (state fields read), `writes` (state fields mutated), `operations` (operation classes dispatched via `_run_op`), and `post_invariants` (predicates that must hold after the node runs). Successors live in the `run()` return type annotation — pyright + pydantic-graph enforce them, and `graph/topology.py:topology()` exposes them as a Python value you can inspect, diff, and assert reachability over (see `tests/test_topology.py`). The recurring "silent dead zone" routing-bug class (where a node returns to a successor that doesn't continue work the claim still needs) becomes structurally detectable rather than benchmark-detectable. Cross-cutting data carried on entities (e.g. `Objective.decomposition` → `entities/decomposition.py:Decomposition`) uses typed Pydantic models, not `dict[str, Any]`, so consumers access fields by attribute and divergent-lookup bugs become impossible.
- **P7: Lazy escalation — each layer asks for what it's missing; breadth comes from demand, not eagerness.** Planning, investigation, scrutiny, and synthesis don't pre-commit to a wide search. Each layer that detects insufficient evidence emits a `Demand` (`andamentum.epistemic.demand.Demand` — a 3-field flat `needs_more` / `justification` / `target_hint` Pydantic model designed for small Ollama models), and the graph routes that demand to the layer that can satisfy it minimally. Evidence gathering (both initial and follow-up rounds) flows through description-driven dispatch (`epistemic_dispatch_provider`): each provider class self-describes via `description`, `query_guidance`, `query_examples`, `output_kind`, and the dispatch agent is invoked once per provider in parallel — it either commits one or two native-syntax queries or abstains. Cross-domain convergence still requires ≥2 independent sources (Reichenbach common-cause), but the system reaches that threshold through per-provider commit/abstain rather than a separate tournament. Investigation rounds add a methodological *angle* on top of the claim: `epistemic_investigate_claim` generates intents (e.g., "adversarial replication"), and the dispatch agent receives the original claim as the subject plus the intent as the angle modifier — queries stay grounded in the claim's lexicon while exploring the intent's dimension. At synthesis, `CheckSynthesisDemand` runs cheap deterministic gates (open-research, no-combined-verdict, stranded-claims, decisive posterior ≥`POSTERIOR_DECISIVE_THRESHOLD` / ≤`1−POSTERIOR_DECISIVE_THRESHOLD`) before the LLM judgment, and on `needs_more=True` loops back to `Scrutinize` for the eligible claims. The load-bearing termination guarantee is the per-sub-claim cap (`PEIRCE_CYCLE_CAP`), not a global give-up budget. Add new "do more work" decisions to this scheme — emit a `Demand` and let the graph route it — rather than reaching for an eager broad-search shortcut.

**Epistemic core abstractions** (understand these before touching `epistemic/`):
- `entities/` — `Objective`, `Evidence`, `Claim`, `Uncertainty`, `Decision`, `Snapshot`, `Artefact` (all `EpistemicEntity` subclasses), plus `decomposition.py` (typed `Decomposition` / `SubInvestigation` / `CombinedVerdictData` for `Objective.decomposition`). `Objective.claim_to_verify` enables seed-claim verification mode (a Pydantic `model_validator` refuses Objectives that set both `claim_to_verify` and `decomposition` — they're mutually exclusive seed modes; `CreateClaims` would silently pick single-seed and discard the decomposition).
- **Snapshot/Artefact indirection.** The final report is reachable from `Objective` via two routes — direct (`Objective.artefact_id → Artefact.content`, the 1-hop signal stamped by `SynthesizeReportOperation`/`SynthesizeInsufficientReportOperation` on completion) and via the immutable freeze (`Objective.snapshot_id → Snapshot.artefact_id → Artefact.content`, the auditable history). Both point at the same `Artefact`. `Objective` does NOT have a `report` attribute; readers looking for "where's the output" should use `Objective.artefact_id` for the load-bearing presence check (e.g. the synthesis stage exit invariant in `graph/stages.py:_check_synthesis`) and traverse the snapshot path when they need the immutable freeze that produced it. Artefacts have `artefact_type` ∈ {`summary`, `insufficient`, ...} — the typed signal that distinguishes a directional verdict from a fallibilism terminal (`SynthesizeInsufficient`); consumers check this field rather than parsing the verdict prose.
- `gates.py` — `STAGE_GATES` + `validate_promotion`: deterministic, routing-aware checks that must pass before a `Claim` advances stages. Gates query the question type's routing profile and only require tracks that are PRIMARY or SECONDARY — not SKIP.
- `graph/` — pydantic-graph DAG scheduler. 21 nodes with typed return edges and explicit contract metadata (P6). Entry point: `run_epistemic_graph()` in `graph/__init__.py`. Key files: `base.py` (the `Node` class with `reads` / `writes` / `operations` / `post_invariants` ClassVars), `topology.py` (`topology()` reflection helper exposing the graph as a Python dict), `invariants.py` (reusable post-condition predicates like `no_stranded_claims`), `nodes.py` (the 21 concrete node classes).
- `operations/` — pure `BaseOperation` subclasses. Each takes an `OperationInput`, does work, returns `OperationResult`. They do NOT control flow — the graph does.
- `repository.py` — `EpistemicRepository` wraps a `StorageBackend` (in-memory backend ships in `storage.py`)
- `runner.py` — `DefaultAgentRunner` wraps `core.AgentRunner` with epistemic agent registry lookup.

**Deep research pipeline** is a `pydantic-graph` state machine. `state.py` holds `ResearchState`; `graph.py` / `nodes.py` define the nodes; `orchestrator.py` / `runner.py` drive execution. `searxng.py` manages the local SearxNG instance and `circuit_breaker.py` wraps it. Content extraction (`content_extractor.py`) uses `trafilatura` for HTML and `docling` for PDF. Search-query production runs through a per-slot generate→verify loop (`PrepareSearchCycle` → `GenerateOne` ⇄ `Verify` → `ParallelSearch`): the `query_generator` agent emits one query at a time, the `topic_verifier` agent judges it against the goal, rejected queries get up to `MAX_SLOT_RETRIES = 3` retries with verifier feedback before skip-and-tighten kicks in. The previous regex+stopword "topic guard" in `text_utils.py` has been retired.

**Document store** is SQLite-first. Databases live in `~/.local/share/document-store/{name}.db` (override with `DOCUMENT_STORE_DIR`; legacy `ANDAMENTUM_DATABASES_DIR` is also honoured). Ingestion is two-phase: document registered immediately (FTS5-searchable), chunks + embeddings written in a background pass that `repair()` can resume after a crash. Chunking goes through `andamentum.chunker.extract_units` with `target_max_chars=4000` (a paragraph-of-paragraphs sizing tuned for the chunk-level metadata extractor); the `Unit → Chunk` adapter (`chunker_adapter.py`) recovers full heading-breadcrumb `section_path` values via the chunker's structural API. Search fuses four signals via RRF: FTS5 keyword, chunk embeddings, doc embeddings, and DHP (temporal clustering, see `dhp.py`). Requires Ollama running locally with `embeddinggemma:latest` for embeddings.

**Typeset module** (`andamentum.typeset`) — a standalone typesetting system with 7 visual atoms (`heading`, `prose`, `callout`, `items`, `aside`, `card`, `reference`), 3 named styles (`article`, `cv`, `report`), and HTML + PDF output. Used by the epistemic audit report (`epistemic/audit_report.py`, the only HTML renderer in the epistemic pipeline — the prose-led `typeset_report.py` was retired when the audit layout reached production), by `whetstone` for its HTML review output, and by `scribe` for its HTML/PDF render path. The epistemic data schema (the dataclasses consumed by the renderer) lives in `epistemic/report_data.py`. See `src/andamentum/typeset/USAGE.md` for the full API reference.

**Whetstone module** (`andamentum.whetstone`) — structured criterion-cascade review over drafts the user wrote themselves. Public entry: `await review_document(source, *, model)`. Implementation is a pydantic-graph DAG inside `whetstone/v3/` — sectionize → extract claims → build document model → run-criteria cascade → verify findings → gap-loop re-grounding → consolidate → 3-node optional novelty check → gate → synthesise → critique-and-revise → optional editor → finalize. Renderers (`render_markdown`, `render_html`, `render_docx`) consume the resulting `ReviewResult` unchanged; the DOCX renderer adapts each `Edit`/`Finding` into a `DocumentPatch` (defined in `models.py`) and feeds the in-tree track-changes machinery in `whetstone.docx`. Document type routes to one of six pluggable criterion sets via the classifier in `whetstone/_document_type.py`; an unknown type falls back to `GENERAL`. Caller can override the active criterion set with `criteria=[Criterion(...)]` or pass free-text reviewer guidelines via `guidelines_text="..."` (one LLM call decomposes them into a criterion list). Confidentiality-marker tripwire (`whetstone/_confidentiality.py`) refuses to run on peer-review material; `--confirm-own-draft` bypasses. Panel mode (multi-expert simulated peer review) is its own graph in `whetstone/v3/panel/` — different shape (N expert biosketches × N independent reviews + 1 synthesis) — invoked via `run_panel` / the `andamentum-whetstone panel` subcommand. Style / readability checks (SMOG, weasel words, passive voice, etc.) are NOT bundled into the review pipeline — use the separate `andamentum.proofread` module or the `andamentum-whetstone proofread <source>` convenience subcommand. The legacy v1 + v2 surfaces (lens-based section-by-section pipeline, `--mode guidelines/custom/panel` flags, `--perspectives`, `--no-challenge`, `--embedding-model`, `--no-proofread`, `--persist-novelty-cache`) were removed during the v3 consolidation; the active code lives under `whetstone/v3/`.

**Figures module** (`andamentum.figures`) — publication-quality scientific figure generation. High-level entry point: `figure(data, *, kind="auto", style="npg", journal="default", horizontal=None, output="figure.pdf")`. Lower-level building blocks: `setup_style`, `get_palette`, `savefig`, `panel_label`, `shared_legend`, `despine` (in `style.py`); plot primitives in `plots.py`; auto-detection (chart kind, log scale, column roles, bar orientation, x-tick rotation) in `auto.py` and `advisor.py`; bootstrap stats in `stats.py`. Bar charts with many or long categorical labels auto-flip to horizontal (`recommend_horizontal_bars` in `advisor.py`); vertical bars auto-rotate x-tick labels when crowded (`recommend_label_rotation`). Both behaviours emit `advisor_notes` so the caller knows what was changed. The package was absorbed from `mosaic-figures` (uninstall the standalone tool with `uv tool uninstall mosaic-figures` once this branch lands). Scribe integration lives in `figures/scribe_glue.py`: `insert_figure(doc, section, *, output_dir, caption, label, **chart_kwargs)` renders a PNG and inserts a `Figure` block via `Document.insert_into_section`.

**Chunker module** (`andamentum.chunker`) — structural-first semantic chunker. Single entry point: `extract_units(source, *, target_min_chars=2000, target_max_chars=10000, embedding_fn=None, judge_executor=None, domain="general") -> ChunkingResult`. Pipeline (literature consensus, 2026): **Stage 1** — split at markdown headings (`structural.py`); **Stage 2** — for sections >`target_max_chars`, semantic-split at paragraph boundaries chosen by largest cosine drops between adjacent paragraph embeddings (`semantic_split.py`, defaults to local Ollama `embeddinggemma:latest` via `embeddings.py`); **Stage 3** (optional) — small LLM judge (`judge.py`) for cuts whose drop-percentile is in the grey zone (60–90th by default), answers `keep | merge`. The LLM is never the primary segmenter. Each unit's `text` is byte-identical to a source span. The legacy agentic `windowing.py`/`refinement.py`/`NextUnitResult` machinery has been removed; legacy kwargs (`primary_executor`, `window_size`, `lookahead`, `extension_chars`, `max_iterations`) are accepted on `extract_units` for backward compat but ignored.

**Harvest module** (`andamentum.harvest`) — universal source → markdown extraction. Single entry point: `await extract(source: str | Path) -> str`. Resolves URLs (httpx fetch with SSRF protection) and file paths to bytes + format (PDF / HTML / DOCX / PPTX / Markdown / plain via three-tiered detection: extension → MIME → magic-byte sniff). For PDF / DOCX / PPTX uses Docling. For HTML, sniffs `og:type` / JSON-LD `@type` / `<article>` tag (`metadata.py`) and routes article-tagged pages to trafilatura, listing/index pages (`@type=WebPage|CollectionPage|ItemList|...`) to Docling. When metadata is ambiguous, races both extractors concurrently and picks the higher-scoring output (`scoring.py` weights heading count × 10 + paragraph breaks × 1 + char count × 0.001, penalises >5% link density, disqualifies output with zero headings AND zero `\n\n`). Loud failure model — `HarvestError` / `FetchError` / `UnsupportedFormatError` / `ExtractionError` are typed; the function never silently returns empty. Extraction backends in `backends/` are thin async adapters with the same `extract(data, source_url) -> str` shape so the orchestrator can race them generically.

**Scribe module** (`andamentum.scribe`) — block-based document authoring. Documents live in SQLite at `~/.local/share/scribe/<name>.db` (override with `SCRIBE_DIR`). Public entry point: `Document.create(title=..., database=..., scaffold="article" | None)`; mutate with `append`/`replace`/`replace_section`; render with `render(path, format="docx")`. Section operations (`list_sections`, `section`, `replace_section`) are derived from heading blocks — there is no separate sections table. Each block has an integer revision counter; `replace()` enforces optimistic locking via `BEGIN IMMEDIATE` and writes an audit row to `scribe_revisions`. Citations are Pandoc-flavoured `[@key]` spans extracted by regex; references live in their own table; `[verify]` and `[citation needed]` markers are recognised and reported by `validate()`. Inline markdown (bold/italic/code) renders as styled runs in `.docx`. `.docx` is the only format wired into `render()`; an internal block→atom mapping toward typeset HTML/PDF lives in `render_typeset.py` but is not yet exposed through the public API. Scribe replaces the standalone `document-tools:doc-draft` plugin for Word file authoring; `.pptx` stays out of scope.

## Design language (andamentum visual system)

The repo's visual design system lives in `docs/design/`. It covers both document rendering (the `.typeset-*` family used by `andamentum.typeset`) and broader app/UI surfaces (the `.am-*` family for chrome — buttons, sidebars, chat, tool logs, settings, modals).

| File | What it is |
|---|---|
| `docs/design/components.css` | The complete stylesheet — all tokens + all components, light + dark via `data-theme`. Heavily section-commented. |
| `docs/design/DESIGN.md` | The *why*: vibe, palette logic, density rules, tone of voice, do's and don'ts. |
| `docs/design/CLAUDE.md` | Operating instructions for agents — surface→class cheat sheet, rules for new components, explicit list of things to refuse (gradients, drop shadows on cards, emoji icons, saturated semantic colours, pill buttons, etc.). |
| `docs/design/showcase.html` | Live visual reference for every component. |
| `docs/design/snippets/` | One copy-paste HTML file per component. |

When working on HTML output, CSS, or any new visual surface, **read `docs/design/CLAUDE.md` and `docs/design/DESIGN.md` first.** Match the system rather than inventing new selectors, colours, or spacing. The system uses CSS custom properties (`--am-*`) for every colour, size, and gap — never write raw hex codes or pixel values when a token exists.

**Spec vs. live implementation.** `docs/design/components.css` is the canonical specification of the visual system. `src/andamentum/typeset/styles.py` is the live CSS that the renderer compiles into output today. They are not yet in sync — the design directory is the source of truth that `styles.py` should converge toward. When making visual changes, edit the spec first; the migration from `styles.py` to `components.css` is a separate, larger piece of work.

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
