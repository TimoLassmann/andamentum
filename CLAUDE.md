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
- `andamentum.deep_research` — the research loop: plan, then per sub-question queries → search → choose → read → notes → reflect, one write, a code-checked report that cites verified findings, over web (an isolated Chrome in podman, or SearxNG) and scholarly sources, with a research store and a novelty check
- `andamentum.document_store` — one SQLite file of documents: fixed source tables (every revision in full, original bytes), disposable indexes rebuilt from them, and a fixed async API (`Store`: put / get / update / delete / list / count / search / read / sync). Default search: BM25 + vectors over chunks, RRF top 50, cross-encoder rerank; `AgentSearcher` adds a model-driven tool loop on top. Enrichers write metadata outside the store
- `andamentum.whetstone` — criterion-cascade review of your own drafts → markdown / HTML / tracked-changes `.docx`; reviewer profiles, checklist and panel modes, and `revise` (review until a stated condition holds)
- `andamentum.figures` — publication-quality scientific figures: 10 chart types, journal palettes and the andamentum report look
- `andamentum.chunker` — deterministic, model-free chunking of markdown: `chunk(text, max_chars=N)` splits at headings, then paragraphs, then sentences; fenced code, tables and front matter are never cut; each `Chunk` carries source offsets and its heading path. Standard library only.
- `andamentum.harvest` — any source (PDF / HTML / DOCX / PPTX / Markdown / plain) → markdown; `mode="fast"` (default) or `"accurate"` (VLM pipeline, PDF only, `vlm` extra)
- `andamentum.vision_critique` — bounded, schema-checked vision critique of a rendered figure
- `andamentum.proofread` — deterministic readability and style checks; no LLM, pure function
- `andamentum.typeset` — standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules for rendering
- `andamentum.agentic_dialect` — the house style for agentic graph systems, as data plus portable conformance checks; a leaf module
- `andamentum.forge` — a natural-language brief → a runnable agentic system built in the dialect; LLM-written code runs only in a sandbox
- `andamentum.llm_judge` — LLM-as-judge: `judge_score` / `judge_compare`; the `model` argument's type sets the mode (one id = fast, a list = panel)
- `andamentum.references` — reference checking for a draft in any citation style: find, decide by evidence, check, Crossref lookup; the package's only citation reader
- `andamentum.compose` — a task in, a finished document out (email, memo, report, article, howto, proposal, grant): brief → evidence → outline → draft → independent checks → revise, no approval step; the result is a document folder (markdown sections) plus rendered outputs; never sends anything. Working on an existing Word file is the separate `andamentum.word` tool (check, respond, bind and refresh numbers, citations, draft into Word), implemented in `compose.harness`. `compose.bundle` is its model-free subpackage: a document as a plain folder bundle (markdown sections with frontmatter), rendered to `.docx` / `.md` / `.html` / `.txt`; `article` scaffold only
- `andamentum.core` — shared model-resolution, `AgentRunner`, embedding infrastructure (`core.embeddings`) and local decision models (`core.decisions`: a checked Ollama handle that answers typed questions; a site given one uses it instead of its LLM call) used by all sub-modules

**Each module's full notes** — design decisions, measured numbers, pitfalls, layering reasons — live in `src/andamentum/<module>/CLAUDE.md`, which Claude Code loads when it reads files in that directory. Read it before you change the module.

Everything ships in one distribution. The default dependencies are the flat union of what the sub-modules need. Two optional extras add heavier, self-contained capabilities with graceful fallback when absent: `html-articles` (trafilatura, Apache-2.0 — falls back to docling) and `pdf` (WeasyPrint, BSD-3 — `typeset.render_pdf` raises `PdfUnavailableError`, an `ImportError` naming the extra, without it). A third, `vlm`, is documented under harvest. sentence-transformers (Apache-2.0) is a core dependency, because the document store's default searcher reranks every query.

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

The canonical green state: **pyright 32 errors (test-only typing noise — pydantic-graph generic variance in `test_topology.py` and in the `_Ctx` stand-in that several `v3` node tests pass to `run()`, `Decomposition` dict-form fixtures in several test files; zero errors in shipped `src/` code. The count moves by one for every node test added, which is why it is a baseline to re-measure rather than a target), `uv run ruff check src/` clean (note: `ruff format` still wants to reformat ~110 pre-existing files — a known cleanup, run `uv run ruff format` once and commit), pytest 6135 passing (29 skipped, 79 deselected; the known environment failure `forge/tests/test_sandbox_podman.py::test_network_run_reaches_the_net` needs a podman network)**. Don't claim completion until you've run these three and seen that state.

Run pytest from the **repo root**. `testpaths` is relative, so invoking it from inside a sub-package silently collects only that subtree — which looks like a green run against a fraction of the suite.

Two caveats on that line. **Scope `ruff check` to `src/`**: repo-wide it reports ~76 findings, all pre-existing style issues in `paper/scifact/` (the SciFact benchmark harness, adopted into the repo 2026-08-09 and never linted); they are unrelated to the library and cleaning them is a separate chore. **The pytest number drifts constantly and this line goes stale** — it read 2755 while the suite was actually at 3043, so re-measure rather than trusting the figure if it looks off. The most recent move is +85 from whetstone checklist mode. (History: the count moved 2075 → 2152 across the whetstone-v3 consolidation — v2 deletion removed ~380 v2-only tests, v3 added ~460 — then release-prep hardening added the SSRF-redirect / typeset-escaping / size-cap / partial-coverage / panel-confidentiality regression tests to reach 2234. The `llm_judge` landing contributed 148, ~85 of them regression tests pinning the tie-machinery, normalisation, boundary-validation, and model-call-layer bugs found in its pre-merge audit.)

## CLIs

The scripts installed by the package. `andamentum list` prints every tool (`--json` for agents) from `andamentum.TOOLS` (`src/andamentum/tools.py`, the one list; a new command is added there and in `pyproject.toml`), and `andamentum <tool> ...` forwards to the tool's own command. **Every command and every main Python function follows `docs/interface.md`**: positional input with `-` for stdin, `-o/--out` (a folder when a command writes several files; nothing written anywhere else), `--json`, `--model` required with no environment variable or default, secondary models named by job with no default, exit codes 0 success / 1 negative result / 2 usage / 3 input unreadable / 4 run failed, `error:` messages, `build_parser()` + `main(argv) -> int`, `--help`/`--version` that load no heavy library; in Python one exported main function per tool, keyword-only after the input, typed results and one base error per module. **Presets on the command line, every setting in Python:** no tuning numbers as flags (a default or one named preset such as `research --depth`), a mode is one flag with named values, one flag name per idea (`--offline` everywhere). **Every `--help` ends with one to three examples, and the user guide `docs/guide.md` shows each tool with commands to copy;** a new command or a changed flag updates both, and the test parses every example in each. `src/andamentum/tests/test_interface.py` checks the mechanical rules; follow the rest by hand when adding a flag or a command. Run `--help` on any binary for the exhaustive flag reference; the table below is the at-a-glance map of what to reach for.

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-epistemic` | Answer a question or check a claim by weighing evidence for and against it (web evidence needs `--searxng-url` or `--browser`). `ask "<question>"` (decomposes, falls back to open research) and `verify "<claim>"` (one claim, skips decomposition) need `--model` and `--embedding-model`; `--out DIR` keeps the database and `report.html` (without it the run uses a temporary folder), `--json` prints the typed result. `stage NAME --out DIR`, `inspect DIR`, `confidence DIR`, `preflight`, `run`, `agents`. Python: `await andamentum.epistemic.ask(q, model=, embedding_model=, searxng_url=)` / `verify(...)` → `AskResult`. Exit 0 / 1 negative / 2 usage / 3 input / 4 failed | required |
| `andamentum-research` | Research a question over the web and scholarly sources; the report cites verified findings by id. `andamentum-research [ask] "QUESTION" --mode {web,scholarly,both} --model M [--depth {quick,standard,deep}] [--mailto EMAIL] [--store PATH] [--cache PATH] [--searxng-url URL \| --browser] [--local-corpus PATH] [--tdm-host HOST]`, `--json`, `-o DIR` (without it the report goes to stdout and no file is written; with it the folder gets `report.md` or `report.json` plus `log.md` and `log.jsonl`), `--quiet`. `--depth` is the one tuning flag: a named `Budget` (`Budget.for_depth`; `standard`, the default, is `Budget()`; `quick` and `deep` are starting points, not measured). The scholarly sources are OpenAlex, Europe PMC and arXiv; `--local-corpus` adds your own documents. Every finer setting (each `Budget` field, the source list, Semantic Scholar, results per query) is an argument of the Python `research()`. Other subcommands: `mcp` (stdio MCP server over the store), `cache info` / `cache prune` (HTTP cache upkeep). Exit 0 clean, 1 degraded (result still written), 2 usage, 3 `--local-corpus` unreadable, 4 run failed. `ask` is the default command (the word is optional). Web mode (`--mode web\|both`) needs `--browser` (an isolated Chrome in a podman container the run starts and removes; needs podman) or a SearxNG address. Details: `src/andamentum/deep_research/CLAUDE.md` | required |
| `andamentum-whetstone` | Review of your own draft → markdown / HTML / tracked-changes .docx. Seven commands, each its own subparser: `review` (default when the first argument is a file; `--brief TEXT\|@FILE`, `--reviewer A[,B...]`, `--no-editor`, `--no-references`, `--offline` (refuses a web address as the input), `--check-novelty`), `revise DRAFT (--until TEXT \| --until-clean {structure,citations,figures}) [--figure LABEL=FILE --figure-model M]` (review → edit → review until the condition holds twice running; one reviewer; the editor runs every round; at most 4 rounds, `RevisionSpec(max_rounds=)` in Python), `panel` (needs `--i-am-the-author`), `checklist`, `reviewers`, `apply-patches INPUT.docx --patches P.json -o OUT.docx`, `verify-provenance [--json]`. Presets on the command line; every setting (criteria, editor criteria, gap-loop cap, revise round cap, novelty depth and store) stays a Python argument. Python: `await review_document(src, *, model)`, `await revise_document(doc, spec=RevisionSpec(...), model=)` → `RevisionResult`. `-o/--out FILE` is repeatable and the only place output goes. Exit 0 / 1 negative (`revise` stopped before its condition held, no provenance marker, patches not applied) / 2 usage / 3 input unreadable / 4 run failed. The reference check and the style check are the separate tools `andamentum-references` and `andamentum-proofread`. Details: `src/andamentum/whetstone/CLAUDE.md` | required for `review`, `revise`, `panel`, `checklist`; none for the rest |
| `andamentum-references` | Citations against the reference list in any citation style, then each entry against Crossref: `DRAFT [--model M] [--decision-model M] [--offline] [--mailto EMAIL] [--json] [-o FILE]`; `-` reads markdown from stdin, Word / PDF / HTML go through `harvest`. Exit 1 when problems are found, 3 unreadable input, 4 run failed. Details: `src/andamentum/references/CLAUDE.md` | optional (`--model` / `--decision-model` only for what the text leaves open) |
| `andamentum-figures` | Publication-quality scientific figures (10 chart types, 7 journal palettes and the andamentum report palette, journal-matched sizing; `[plot] DATA -o FILE` (`plot` is the default command; `-o` required; the format follows the extension, raster output is 300 dpi, nothing is written elsewhere), `--json` prints the result; `--journal report --style andamentum --svg-inline` for a report figure, `--journal showcase` for large type on slides; also `palettes`, `journals`; exit 0 / 2 usage / 3 data unreadable / 4 failed) | none |
| `andamentum-chunker` | Deterministic chunking of markdown by headings, paragraphs and sentences (`INPUT --max-chars N [-o FILE] [--text-only]`; `-` reads stdin; exit 0 / 2 usage / 3 input unreadable) | none |
| `andamentum-harvest` | Any source → markdown (PDF / HTML / DOCX / PPTX / Markdown / plain). `--mode fast` (default) or `--mode accurate` (VLM pipeline, PDF only, `uv sync --extra vlm`); `--no-ocr`; `--preserve-superscripts`; `-o FILE`; `--figure-model` (no default; required with `--mode accurate`) | none for `fast`; `accurate` needs local vision models |
| `andamentum-vision-critique` | Vision-critique of a rendered figure → bounded JSON (label overlap, legibility, legend placement, aspect-ratio, suggested fixes from a fixed set, confidence). Validated local model: `ollama:qwen3.8`. `IMAGE --model M [--context TEXT] [-o FILE]`; PNG/JPEG/WEBP/GIF only. Exit 0 / 1 issues found / 2 usage / 3 image unreadable or unsupported / 4 failed | required (multimodal) |
| `andamentum-proofread` | Deterministic readability + style check (SMOG, Flesch-Kincaid, weasel words, passive voice, duplicate words, weak openers, adverb density). Accepts URLs, PDF, DOCX, HTML, PPTX, Markdown, plain text (via `harvest.extract`) — or `-` for stdin. `--json`, `-o FILE`, `--slides [--allow ACRONYMS]` (a .pptx deck; `--allow` is an error without `--slides`). Exit 0 / 2 usage / 3 input unreadable / 4 failed | none |
| `andamentum-typeset` | Render markdown to a styled HTML page or PDF: `SOURCE` (or `-`) `[-o FILE.html|FILE.pdf] [--style article|cv|report] [--title T] [--footer F]`; HTML to stdout without `-o`; PDF needs the `pdf` extra (`pip install 'andamentum[pdf]'` plus pango/cairo). Exit 0 / 2 usage / 3 input unreadable / 4 render failed | none |
| `andamentum-agentic-dialect` | The agentic-systems house style: `laws` / `law <id>` (the 9 laws), `roles` / `role <role>` (the prompt-slice for a job), `checklist`, `skeleton`, `check <path> [--strict]` (portable conformance gates; `--strict` adds the opt-in swallowed-exception gate); `--json` on `check` and the listings. Exit 0 / 1 violations found / 2 unknown law or role / 3 path unreadable | none |
| `andamentum-forge` | Build a small runnable function package from a plain-language brief, tested in a sandbox; refuses apps, agents and services. `build BRIEF --model M --out DIR` (design → render → write the code → sandbox test; `--stop-after {render,build,audit}`, `--sandbox {podman,subprocess}`, `--verbose`) or `design BRIEF --model M` (validated design only, writes nothing); `--json`. Python: `await run_forge(brief, *, model, out=None, stop_after=...)`. Exit 0 / 1 a check failed / 2 usage or refused brief / 4 failed | required |
| `andamentum-llm-judge` | LLM-as-judge → JSON on stdout (or `-o FILE`). `score TEXT` (per-criterion, against `--criteria` or the built-in six) or `compare TEXT_A TEXT_B` (both presentation orders, always); a text of `-` reads stdin. Python: `judge_score(text, *, model, ...)`, `judge_compare(text_a, text_b, *, model, ...)`. Exit 0 / 2 usage / 3 criteria file unreadable / 4 failed. `--model` once = fast single judge; repeated or comma-separated = sequential panel with an agreement gate. Every result carries `confidence`, `doubt`, and `needs_review` | required |
| `andamentum-compose` | A task → a finished document in a folder (no approval step, never sends anything). `write "<task>" -o DIR --model M [--genre G] [--source FILE_OR_URL]... [--research {auto,web,scholarly,both}] [--searxng-url URL \| --browser] [--format F]... [--no-provenance]`: one `--model` does every job (drafter, critic, judge, reader); `--research auto` lets the genre choose where to search; up to 4 check-and-revise rounds. Then `continue DIR --model M` (repeats the run with the settings `write` recorded in the folder; a folder without them exits 3; `--searxng-url`/`--browser`/`--mailto` only when web research did not finish), `status DIR`, `check DIR --model M` (checks once), `render DIR [--format F] [--plain -o FILE]`, `genres`, `eval -o DIR --model M`, and the no-model folder commands `new -o DIR --title T`, `sections`, `show`, `edit`, `add`, `validate DIR`; `--json` on the ones that print a result. Python: `await andamentum.compose.write_document(task, *, out, roles)` / `continue_document` → `ComposeResult` (per-role models through `Roles`, `max_rounds=` there); errors derive from `ComposeError`. Exit 0 goal met / 1 goal not met or `validate` error / 2 usage / 3 input unreadable / 4 failed. Guide: `docs/compose.md`; details: `src/andamentum/compose/CLAUDE.md` | required for `write`, `continue`, `check`, `eval`; none for the folder commands |
| `andamentum-word` | Work on an existing Word file; the input is never edited and every command that writes puts the edited copy, the report and `state/` into `-o/--out DIR` only. `check FILE.docx -o DIR` (citations, retractions, URLs, cross-references, statistics, abstract numbers; `--results DIR --model M` numbers, `--decision-model M` pre-gate), `respond FILE.docx -o DIR [--reviews FILE] --model M`, `bind-numbers FILE.docx -o DIR --results DIR --model M`, `refresh FILE.docx -o DIR --results DIR`, `draft "<task>" -o DIR --model M [--into FILE.docx] [--results DIR] [--library FILE] [--zotero] [--store PATH] [--research {auto,web,scholarly,both}]` (tracked changes, comments, verified citations and bound numbers; one model for every job; up to 2 revise rounds; after drafting, a cite loop finds, reads and cites papers for uncited background sentences, library first, and keeps every paper it read in the document store `--store PATH`, default `state/papers.db`), `find-citation QUERY` (prints), `paragraphs FILE.docx` (prints the paragraph ids `insert-citation` needs), `insert-citation FILE.docx -o DIR --paragraph-id ID --at N --doi DOI`, `mcp` (stdio MCP server; a tool call names its `out` file); `--json` prints the result. Python: `await andamentum.word.check_document(path, *, out)` and `respond_to_reviews`, `refresh_numbers`, `bind_numbers`, `draft_document`, `find_citation`, `list_paragraphs`, `insert_citation`; errors derive from `WordError`. Exit 0 / 1 errors found or items left for the authors / 2 usage / 3 input unreadable / 4 failed. Guide: `docs/word.md`; details: `src/andamentum/compose/harness/CLAUDE.md` | required for `respond`, `bind-numbers`, `draft`; `check` needs a model only with `--results`/`--decision-model` |
| `andamentum-docstore` | The document store from a shell: `ingest DB SOURCE...` (harvest → markdown, original bytes kept), `sync DB --embedding-model M` (build the index; Ctrl-C and rerun to continue), `search DB QUERY --embedding-model M --reranker-model R`, `get`, `list`, `delete`, `enrich DB NAME` (identifiers, facts, bibliographic, extract, classify); `--json` on every command that prints a result; exit 1 when `sync` could not index some documents, 3 for a missing store or source, 4 for a model or network error | none for `ingest`/`get`/`list`/`delete` and the identifiers/facts enrichers; `sync` embeds, `search` embeds and reranks; `enrich extract` needs `--model`, `enrich classify` a decision model |

Every command that calls a model takes `--model` on the command line; there is no environment variable and no default (`core.models.resolve_model_from_args` raises `ValueError` for a missing one, and a command makes `--model` an argparse `required=True` flag). `andamentum-llm-judge` accepts the flag more than once to form a panel; vision-critique requires a multimodal model. `andamentum-figures`, `andamentum-harvest`, and `andamentum-proofread` have no LLM dependency.

## Architectural conventions

**Explicit model argument, no hidden defaults.** Every public function that calls an LLM takes `model=` as a keyword-only argument. There is no shared config module, no silent fallback, no ambient default. When adding new LLM-calling code, match this pattern — don't reach for a global config.

**Confusable parameters are keyword-only.** Any parameter that is one of two or more same-typed, semantically confusable values — model names, identifiers, paths, source/target pairs — goes behind `*`. This applies to **private helpers**, not just the public API. The reason is that transposing two `str` parameters is invisible to pyright: the values simply swap and the failure surfaces far from the call site. Behind a `*` the same mistake is a `reportCallIssue` error at every call site, checked without needing a test to exercise that path.

This is not a blanket rule — genuinely positional signatures (`cosine_similarity(a, b)`, `_pct(numerator, denominator)`) stay as they are. It applies where the confusable values are *passed onward* rather than consumed locally, so a swap produces wrong behaviour instead of an immediate error. Retrofit opportunistically when touching a file; don't sweep the repo.

Worked example — a former document-store helper, `_preflight`, took `(database, embedding_model, model=None, *, auto_repair=True)`. A caller passing `(database, model, embedding_model)` handed the embedding service an LLM id, breaking every `search()` call, and neither pyright nor the test suite could see it. The `*` now sits directly after `database`.

**Core module** (`andamentum.core`) — shared infrastructure for model resolution, agent execution, embeddings (`core.embeddings`), and SSRF-safe URL fetching (`core.url_safety`). All sub-modules import from core instead of maintaining independent implementations. When adding LLM-calling code, use `core.agents.AgentRunner` or `core.agents.run_agent_with_fallback()` — they provide model resolution (ollama, bedrock, passthrough) and PromptedOutput fallback for free.

**Layering:**
- `core` is the shared base — no sub-module imports from another sub-module via core's surface (core itself depends on neither).
- `document_store` is foundational; `epistemic` may depend on it directly (e.g., `EpistemicRepository` wraps a `Store`). `document_store` depends on `core`, `chunker`, `references` (enrichers) and `harvest` (`pipeline.py` only).
- `deep_research` is foundational for evidence gathering; `epistemic.evidence_gathering` may depend on it.
- `document_store` and `deep_research` MUST NOT depend on `epistemic` or on each other.
- `whetstone` depends on `core`, `typeset`, `chunker`, `harvest` and `references`; on `deep_research`, `vision_critique` and `llm_judge` only behind opt-in features (novelty check, `revise` checks). It MUST NOT depend on `epistemic` or `document_store`.
- `compose.bundle` depends only on `typeset`, `python-docx` and `pyyaml` (no `pydantic_ai`, `pydantic_graph`, `core`, `whetstone`, `llm_judge`, nor compose code outside `bundle/`; enforced by a test). `figures` depends only on matplotlib + numpy + pydantic; `figures.bundle_glue` is the ONLY place that imports `compose.bundle`.
- `chunker` depends on nothing but the standard library.
- `vision_critique`, `llm_judge` and `references` depend on `core` plus their own third-party libraries and on no other sub-module; the one exception is `references/cli.py`, which imports `harvest` (lazily) to read a Word or PDF file, like `proofread/cli.py`. `llm_judge` MUST NOT import `pydantic-graph`.
- `compose` depends on `core`, `whetstone`, `proofread`, `llm_judge`, `harvest`, `references` and `figures`; `deep_research` behind `--research` and in the Word draft's cite loop, `document_store` in the cite loop only (the papers it reads go into a store), both through deferred imports. Nothing depends on `compose` except `figures.bundle_glue`, which imports only the model-free `compose.bundle`, and `andamentum.word`, the public face of `compose.harness` (the Word-file tool). It MUST NOT depend on `epistemic`.
- Leaf modules that import no other sub-module: `harvest`, `agentic_dialect`, and `proofread` (only `proofread.cli` imports `harvest`).
- `deep_research` depends on `core`, `references` and `harvest` (its page extractor) and on no other sub-module.
- `epistemic` MAY depend on `llm_judge` and on `references` (identifier extraction and PubMed record parsing live in `references`), one way only. `forge` depends on `core` and `agentic_dialect`.
- The full rules, with their reasons, are in each module's `CLAUDE.md`; conformance tests enforce several of them.

**Public API lives in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal.

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
- `@pytest.mark.rerank` is deselected for the same reason as `vlm`: those tests download a real cross-encoder (~2 GB). Run them with `uv run pytest -m rerank`. Every other reranking test, including the `search()` rerank tests, stubs the model, and `select_relevant` is tested with no model at all.
- `@pytest.mark.vlm` is deselected too: those tests run a local vision-language model for `harvest` accurate mode, downloading weights and holding over a GB. Run them explicitly with `uv run pytest -m vlm`. Keep new harvest tests out of that marker by stubbing the backend — the default suite has to stay cheap enough to run while other models are loaded on the same machine.
- WeasyPrint's flex layout breaks when `<p>` tags are inside flex children. The typeset renderer strips `<p>` wrapping from item and reference bodies as a workaround.
