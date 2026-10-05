# CLAUDE.md

Follow the rules in @CONSTITUTION.md. Project-specific rules below override or extend them where explicitly stated.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Public repository: what may be committed

This repository is **public** on GitHub. Everything committed — files, test fixtures, examples in prompts and docstrings, code comments, commit messages — is published, and a force push does not unpublish it (old commits stay reachable by SHA). Treat every commit as a publication.

- **Test data is synthetic, always.** Fixtures, examples, prompt few-shots and docstring samples are invented for the purpose or taken from public sources (arXiv, public datasets). Never derive them from a real document, meeting, email, draft, review run or document-store content — not anonymised, not a short fragment in a comment. Invented names are obviously invented (Apollo, SaltLog, Jane Doe), never real project names, colleagues, students or collaborators. When a comment cites real-world behaviour ("measured on a real manuscript"), give counts, never quoted text.
- **Kept local, never committed** (all in `.gitignore`): working notes (`WORK.md`, `PARKED.md`, `REVIEW_LOG.md`), plans and specs (`docs/plans/`, `docs/superpowers/`, `docs/.internal/`, `docs/epistemic/*_PLAN.md` and the ledger, audit and spec files beside them), unpublished results (`docs/results/`, `paper/`, experiment `FINDINGS.md` / `*_PLAN.md`), `private/`, and `.docx` / `.pdf` / smoke output at the repo root. Never `git add -f` any of these. Public code does not link to them either: a link to an ignored file is broken and advertises that the file exists.
- **No machine-specific data.** No absolute home paths (`/Users/<name>/`), hostnames or usernames in code, docs or recorded results; record paths repo-relative.
- **Commit messages are neutral.** Describe what changed mechanically. Never write "remove private / sensitive / confidential material" — on a public repo that is a beacon.
- **Pushing is the user's action, never Claude's.**

## Project

Andamentum is a single Python package (`src/andamentum/`) of tightly-scoped sub-modules for building agentic reasoning pipelines. The core three were consolidated from separate packages; the commit history around that migration is relevant context.

- `andamentum.epistemic` — formal epistemology: evidence/claims/uncertainty entities, deterministic stage gates, pattern-driven work scheduling, multi-agent verification
- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`
- `andamentum.document_store` — SQLite + FTS5 + sqlite-vec knowledge base: 4-signal RRF search ranks documents; `get_chunks` / `get_chunk` read the passages, and an opt-in rerank (`rerank` extra) scores them
- `andamentum.whetstone` — criterion-cascade review of your own drafts → markdown / HTML / tracked-changes `.docx`; reviewer profiles, checklist and panel modes, the goal loop
- `andamentum.figures` — publication-quality scientific figures: 10 chart types, journal palettes and the andamentum report look
- `andamentum.chunker` — structural-first semantic chunking of long markdown into 2k–10k char self-contained units. Stage 1: split at markdown headings. Stage 2: embedding-based split for over-budget sections. Stage 3 (optional): small-LLM judge for grey-zone boundaries. The LLM is never the primary segmenter.
- `andamentum.harvest` — any source (PDF / HTML / DOCX / PPTX / Markdown / plain) → markdown; `mode="fast"` (default) or `"accurate"` (VLM pipeline, PDF only, `vlm` extra)
- `andamentum.vision_critique` — bounded, schema-checked vision critique of a rendered figure
- `andamentum.proofread` — deterministic readability and style checks; no LLM, pure function
- `andamentum.typeset` — standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules for rendering
- `andamentum.agentic_dialect` — the house style for agentic graph systems, as data plus portable conformance checks; a leaf module
- `andamentum.forge` — a natural-language brief → a runnable agentic system built in the dialect; LLM-written code runs only in a sandbox
- `andamentum.llm_judge` — LLM-as-judge: `judge_score` / `judge_compare`; the `model` argument's type sets the mode (one id = fast, a list = panel)
- `andamentum.references` — reference checking for a draft in any citation style: find, decide by evidence, check, Crossref lookup; the package's only citation reader
- `andamentum.compose` — a task in, a finished document out (email, memo, report, article, howto, proposal, grant): brief → evidence → outline → draft → independent checks → revise, no approval step; the result is a folder bundle plus rendered outputs; never sends anything. `compose.bundle` is its model-free subpackage: a document as a plain folder bundle (markdown sections with frontmatter), rendered to `.docx` / `.md` / `.html` / `.txt`; `article` scaffold only
- `andamentum.core` — shared model-resolution, `AgentRunner`, embedding infrastructure (`core.embeddings`) and local decision models (`core.decisions`: a checked Ollama handle that answers typed questions; a site given one uses it instead of its LLM call) used by all sub-modules

**Each module's full notes** — design decisions, measured numbers, pitfalls, layering reasons — live in `src/andamentum/<module>/CLAUDE.md`, which Claude Code loads when it reads files in that directory. Read it before you change the module.

Everything ships in one distribution. The default dependencies are the flat union of what the sub-modules need. Three optional extras add heavier, self-contained capabilities with graceful fallback when absent: `html-articles` (trafilatura, Apache-2.0 — falls back to docling), `pdf` (WeasyPrint, BSD-3 — `typeset` raises a clear ImportError without it), and `rerank` (sentence-transformers, Apache-2.0 — `document_store.rerank` raises an ImportError naming the extra; nothing in the store calls it, so an install without it behaves identically). A fourth, `vlm`, is documented under harvest.

## Commands

Use `uv run` for everything Python-related — `uv run python`, never plain `python` or `python3`, including one-off `-c` snippets and throwaway checks.

**Never write scratch artefacts into the repo.** Screenshots, rendered previews, sample outputs, temp scripts, benchmark dumps and anything else generated while working goes in the session scratchpad directory, not in the working tree. That includes tool output directories such as `.playwright-mcp/` — pass an absolute scratchpad path when a tool asks where to save. The only files that should appear in `git status` after a task are the ones the task was actually asked to produce.

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

The canonical green state: **pyright 32 errors (test-only typing noise — pydantic-graph generic variance in `test_topology.py` and in the `_Ctx` stand-in that several `v3` node tests pass to `run()`, `Decomposition` dict-form fixtures in several test files; zero errors in shipped `src/` code. The count moves by one for every node test added, which is why it is a baseline to re-measure rather than a target), `uv run ruff check src/` clean (note: `ruff format` still wants to reformat ~110 pre-existing files — a known cleanup, run `uv run ruff format` once and commit), pytest 4291 passing (34 skipped, 75 deselected)**. Don't claim completion until you've run these three and seen that state.

Run pytest from the **repo root**. `testpaths` is relative, so invoking it from inside a sub-package silently collects only that subtree — which looks like a green run against a fraction of the suite.

Two caveats on that line. **Scope `ruff check` to `src/`**: repo-wide it reports ~76 findings, all pre-existing style issues in `paper/scifact/` (the SciFact benchmark harness, adopted into the repo 2026-08-09 and never linted); they are unrelated to the library and cleaning them is a separate chore. **The pytest number drifts constantly and this line goes stale** — it read 2755 while the suite was actually at 3043, so re-measure rather than trusting the figure if it looks off. The most recent move is +85 from whetstone checklist mode. (History: the count moved 2075 → 2152 across the whetstone-v3 consolidation — v2 deletion removed ~380 v2-only tests, v3 added ~460 — then release-prep hardening added the SSRF-redirect / typeset-escaping / size-cap / partial-coverage / panel-confidentiality regression tests to reach 2234. The `llm_judge` landing contributed 148, ~85 of them regression tests pinning the tie-machinery, normalisation, boundary-validation, and model-call-layer bugs found in its pre-merge audit.)

## CLIs

The scripts installed by the package. Run `--help` on any binary for the exhaustive flag reference; the table below is the at-a-glance map of what to reach for.

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-epistemic` | Formal-epistemology pipeline. Two modes: `ask "<question>"` (research mode — system attempts decomposition, falls back to open research if the question doesn't decompose) or `verify "<claim>"` (single-claim verification, SciFact-style — seeds exactly one Claim from the provided text, skips decomposition) | required |
| `andamentum-research` | Web-research pipeline (search → fetch → extract → verify → synthesise) over a local SearxNG | required |
| `andamentum-whetstone` | Review of your own draft → markdown / HTML / tracked-changes .docx. Subcommands: `review` (default), `panel`, `checklist`, `references`, `proofread`, `apply-patches`, `verify-provenance`. Details: `src/andamentum/whetstone/CLAUDE.md` | required for `review`, `panel`, `checklist`; optional for `references`; none for the rest |
| `andamentum-figures` | Publication-quality scientific figures (10 chart types, 7 journal palettes and the andamentum report palette, journal-matched sizing; `--journal report --style andamentum --svg-inline` for a report figure) | none |
| `andamentum-chunker` | Verifiable semantic chunking of long markdown into 2k–10k char self-contained units | optional (stage 3 only, `--judge-model`) |
| `andamentum-harvest` | Any source → markdown (PDF / HTML / DOCX / PPTX / Markdown / plain). `--mode fast` (default) or `--mode accurate` (VLM pipeline, PDF only, `uv sync --extra vlm`); `--no-ocr`; `--figure-model` | none for `fast`; `accurate` needs local vision models |
| `andamentum-vision-critique` | Vision-critique of a rendered figure → bounded JSON (label overlap, legibility, legend placement, aspect-ratio, suggested fixes from a fixed set, confidence). Validated local model: `ollama:qwen3.8:27b-mlx` | required (multimodal) |
| `andamentum-proofread` | Deterministic readability + style check (SMOG, Flesch-Kincaid, weasel words, passive voice, duplicate words, weak openers, adverb density). Accepts URLs, PDF, DOCX, HTML, PPTX, Markdown, plain text (via `harvest.extract`) — or `-` for stdin. `--format human` (default) or `--format json` | none |
| `andamentum-agentic-dialect` | The agentic-systems house style: `laws` / `law <id>` (the 9 laws), `roles` / `role <role>` (the prompt-slice for a job), `checklist`, `skeleton`, `check <path> [--strict]` (portable conformance gates, nonzero exit on fail; `--strict` adds the opt-in swallowed-exception gate) | none |
| `andamentum-forge` | Meta-system: a natural-language brief → a recipe/dialect-validated `SystemSpec` → rendered, agent-authored, sandbox-audited runnable package. `build` (full pipeline: design → render → author code → audit) or `design` (validated spec only). `--stop-after {render,build,audit}`, `--sandbox {podman,subprocess}`, `--verbose` (live progress dashboard) | required |
| `andamentum-llm-judge` | LLM-as-judge → JSON on stdout (or `-o FILE`). `score <output>` (per-criterion, against `--criteria` or the built-in six) or `compare <output_a> <output_b>` (both presentation orders, always). `--model` once = fast single judge; repeated or comma-separated = sequential panel with an agreement gate. Every result carries `confidence`, `doubt`, and `needs_review` | required |
| `andamentum-compose` | A task → a finished document. `write "<task>" --out FILE.docx [--into FILE.docx] [--results DIR] [--library FILE] [--zotero] [--template FILE] [--offline]` drafts into Word as tracked changes, comments, verified citations and bound numbers (the `--into` file is never edited); `write "<task>" --out DIR` is the bundle path (`--genre`, `--source`, `--research`, `--format`, `--no-provenance`); `--model` or per-role `--drafter/--critic/--judge/--reader`, `--max-rounds`. Word verbs, each writing a copy: `check FILE.docx` (citations, retractions, URLs, cross-references, statistics, abstract numbers; `--results DIR --model M` numbers, `--decision-model M` pre-gate; exit 0/1/2/3), `bind-numbers FILE.docx --results DIR --model M`, `refresh FILE.docx --results DIR` (exit 0/1), `respond FILE.docx [--reviews FILE] --model M` (exit 0/4), `find-citation QUERY`, `insert-citation FILE.docx --paragraph-id ID --at N --doi DOI`, `mcp` (stdio MCP server). Bundle verbs: `run DIR`, `status DIR`, `check-bundle DIR`, `render DIR` (`--raw --format F --out FILE`), `genres`, `eval`, and the no-model `new`, `sections`, `show`, `edit`, `add`, `validate`. Exit 0 done, 4 open issues, 3 crash, 2 usage. Guide: `docs/compose.md`; details: `src/andamentum/compose/CLAUDE.md`, `src/andamentum/compose/harness/CLAUDE.md` | required for the writing verbs; `check` needs a model only with `--results`/`--decision-model` |
| `andamentum-docstore` | Deferred ingestion for the document store: `ingest [--defer]`, `ingest-source`, `status`, `process-pending`, `retry-failed` | only for `process-pending` and the non-deferred paths |

`andamentum-epistemic`, `andamentum-research` and `andamentum-forge` resolve their LLM via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`, routed through `core.models.resolve_model_from_args`, which `sys.exit(1)`s if neither is set — no hidden defaults. `andamentum-whetstone`, `andamentum-vision-critique`, and `andamentum-llm-judge` take `--model` directly (any pydantic-ai model id; vision-critique requires a multimodal one, and llm-judge accepts the flag more than once to form a panel). `andamentum-figures`, `andamentum-harvest`, and `andamentum-proofread` have no LLM dependency.

## Architectural conventions

**Explicit model argument, no hidden defaults.** Every public function that calls an LLM takes `model=` as a keyword-only argument. There is no shared config module, no silent fallback, no ambient default. When adding new LLM-calling code, match this pattern — don't reach for a global config.

**Confusable parameters are keyword-only.** Any parameter that is one of two or more same-typed, semantically confusable values — model names, identifiers, paths, source/target pairs — goes behind `*`. This applies to **private helpers**, not just the public API. The reason is that transposing two `str` parameters is invisible to pyright: the values simply swap and the failure surfaces far from the call site. Behind a `*` the same mistake is a `reportCallIssue` error at every call site, checked without needing a test to exercise that path.

This is not a blanket rule — genuinely positional signatures (`cosine_similarity(a, b)`, `_pct(numerator, denominator)`) stay as they are. It applies where the confusable values are *passed onward* rather than consumed locally, so a swap produces wrong behaviour instead of an immediate error. Retrofit opportunistically when touching a file; don't sweep the repo.

Worked example — `document_store/public.py:_preflight` took `(database, embedding_model, model=None, *, auto_repair=True)`. A caller passing `(database, model, embedding_model)` handed the embedding service an LLM id, breaking every `search()` call, and neither pyright nor the test suite could see it. The `*` now sits directly after `database`.

**Core module** (`andamentum.core`) — shared infrastructure for model resolution, agent execution, embeddings (`core.embeddings`), and SSRF-safe URL fetching (`core.url_safety`). All sub-modules import from core instead of maintaining independent implementations. When adding LLM-calling code, use `core.agents.AgentRunner` or `core.agents.run_agent_with_fallback()` — they provide model resolution (ollama, bedrock, passthrough) and PromptedOutput fallback for free.

**Layering:**
- `core` is the shared base — no sub-module imports from another sub-module via core's surface (core itself depends on neither).
- `document_store` is foundational; `epistemic` may depend on it directly (e.g., `EpistemicRepository` wraps a `DocumentStore`).
- `deep_research` is foundational for evidence gathering; `epistemic.evidence_gathering` may depend on it.
- `document_store` and `deep_research` MUST NOT depend on `epistemic` or on each other.
- `whetstone` depends on `core`, `typeset`, `chunker`, `harvest` and `references`; on `deep_research`, `vision_critique` and `llm_judge` only behind opt-in features (novelty check, goal checks). It MUST NOT depend on `epistemic` or `document_store`.
- `compose.bundle` depends only on `typeset`, `python-docx` and `pyyaml` (no `pydantic_ai`, `pydantic_graph`, `core`, `whetstone`, `llm_judge`, nor compose code outside `bundle/`; enforced by a test). `figures` depends only on matplotlib + numpy + pydantic; `figures.bundle_glue` is the ONLY place that imports `compose.bundle`.
- `chunker`, `vision_critique`, `llm_judge` and `references` depend on `core` plus their own third-party libraries and on no other sub-module. `llm_judge` MUST NOT import `pydantic-graph`.
- `compose` depends on `core`, `whetstone`, `proofread`, `llm_judge`, `harvest` and `figures`; `deep_research` only behind `--research` (deferred import). Nothing depends on `compose` except `figures.bundle_glue`, which imports only the model-free `compose.bundle`. It MUST NOT depend on `epistemic` or `document_store`.
- Leaf modules that import no other sub-module: `harvest`, `agentic_dialect`, and `proofread` (only `proofread.cli` imports `harvest`).
- `epistemic` MAY depend on `llm_judge`, one way only. `forge` depends on `core` and `agentic_dialect`.
- The full rules, with their reasons, are in each module's `CLAUDE.md`; conformance tests enforce several of them.

**Public API lives in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal. `document_store` also re-exports its public surface from `public.py`.

## Design language (andamentum visual system)

The visual design system: `docs/design/` is the specification, `src/andamentum/typeset/styles.py` the live CSS; details in `src/andamentum/typeset/CLAUDE.md`.

When working on HTML output, CSS, or any new visual surface, **read `docs/design/CLAUDE.md` and `docs/design/DESIGN.md` first.** Match the system rather than inventing new selectors, colours, or spacing. The system uses CSS custom properties (`--am-*`) for every colour, size, and gap — never write raw hex codes or pixel values when a token exists.

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
- `@pytest.mark.rerank` is deselected for the same reason as `vlm`: those tests download a real cross-encoder (~2 GB) and need the `rerank` extra. Run them with `uv run pytest -m rerank`. Every other reranking test stubs the model, and `select_relevant` is tested with no model at all — it must keep working on an install without the extra.
- `@pytest.mark.vlm` is deselected too: those tests run a local vision-language model for `harvest` accurate mode, downloading weights and holding over a GB. Run them explicitly with `uv run pytest -m vlm`. Keep new harvest tests out of that marker by stubbing the backend — the default suite has to stay cheap enough to run while other models are loaded on the same machine.
- WeasyPrint's flex layout breaks when `<p>` tags are inside flex children. The typeset renderer strips `<p>` wrapping from item and reference bodies as a workaround.
