# andamentum.compose — notes for Claude

Claude Code loads this file when it reads files in this directory.

## Overview

`await write_document("<task>", out=DIR, roles=Roles.all(model))` turns a task (and optional sources) into a finished document with no approval step. The brief is compiled, frozen and written into the bundle so it can be read afterwards; nothing waits on a person. compose NEVER sends anything: no SMTP, no mail API, no HTTP posting. The email genre writes `out/email.txt`.

Pipeline (one pydantic-graph graph, `graph.py`):

    Ingest -> CompileBrief -> Research -> Evidence -> Plan -> Draft -> Check -> (Revise -> Check)* -> Finish

- **Ingest** harvests sources into `evidence/sources/` (confidentiality gate first; `--confirm-own-sources` overrides).
- **Research** (only with `--research`, after the brief because the genre picks the mode) runs `deep_research.research` once and adds each work that has a finding as a source of kind `research`. See "Research" below.
- **Evidence** extracts facts per source (sequential); a fact whose quote is not in the source text is dropped and logged. A research source needs no extraction: its facts are its findings. The drafter marks fact use as `[[f12]]`.
- **Plan** makes the outline (one retry on failed outline checks, then proceeds with warnings) and lays out empty sections in the bundle.
- **Draft** writes section by section, each call seeing the first two sentences of the earlier sections (deterministic, no model).
- **Check** (`assess.py`) runs the deterministic checks, grounding (one judge call per changed section), the reader test and the critic (whetstone), then routes findings to per-section work items.
- **Revise** rewrites each named section once per round; outline-level items go to `amend_outline`, at most `MAX_OUTLINE_AMENDMENTS` (1) per run.
- **Finish** renders `genre.outputs` to `out/`, writes `report.md`, returns `ComposeResult`.

Four roles name the model for each job: `drafter`, `critic`, `judge`, `reader`. `Roles` has no defaults; the CLI refuses a missing role with exit 2. Using one model for all four works but weakens the independence of the checks.

## Why these decisions

- **The bundle is the store (dialect L1).** `brief.json`, `outline.json`, `rounds.json`, `evidence/{sources,facts}.json`, `sections/*.md`, `log.jsonl`, `report.md` all live in the bundle folder. Resume is driven by which files exist, so a run that stopped can be continued with `run DIR` and a finished node is never redone. A section still `planned` is drafted; anything else is kept.
- **Goal met twice running.** A round meets the goal when no error-level check failed (warnings never count), the reader got every question right and the critic reported no `major` finding (`moderate` is allowed). It is accepted only when the next check pass over the same text meets it too. This is the whetstone goal loop's rule: the reader and critic are sampled model calls, and one lucky pass is not evidence. Grounding is cached by section hash, so the second pass pays only for reader and critic calls. With `max_rounds=1` the goal can therefore never be reported met.
- **Stop reasons.** `goal_met`; `no_progress` (a revise step changed no section, or the document hash equals one of the two previous rounds); `cap_reached` (`max_rounds`, counted per run, not across resumes); `unresolvable` (no section got any text; no outputs are rendered). Model errors raise; nothing is swallowed. A crash exits 3.
- **Style never drives idea revision.** The only style rule that can fail a check is the genre's readability ceiling (`max_grade_level`). Weasel words and the like are reported as info. Warnings (`section_words`, `to_fill`, `style`) never count against the goal. `[TO FILL: ...]` markers are collected into the report's open questions.
- **Grant is generic.** No funder-shaped template. Funder specifics enter only through the call text the user passes as a source; the brief compiler puts it in `brief.guidelines`, and the planner may rename, add or drop sections to follow it. `provenance_required` is on, so `--no-provenance` is refused before any model call.
- **One code path.** Genres are data (`genres.py`); nothing branches on a genre name. The email txt format is chosen by `brief.email_subject` being set, not by name. A genre can be loaded from a YAML path (`genre_for("x.yaml")`; stored absolute in the bundle metadata).
- **Sequential model calls only.** Never `asyncio.gather` over model calls (local Ollama serialises them anyway).
- **MCP server.** `andamentum-compose mcp` serves the harness verbs and a few native tools (open a document, show a section, apply edits through the safe writer) to Claude Code and Claude for Word. Every tool writes a copy and hashes its input files before and after; `apply_edits` cannot write citations, which go through the verifying `insert_citation`. The bundle pipeline (`write_document`) is reached through `write` with a non-`.docx` `--out`. See `harness/CLAUDE.md`.

## Research (`research.py`)

`--research` consumes `deep_research` findings and passages, never the engine's report prose.

- **Mode per genre**, as data: `Genre.research_mode` (`web`, `scholarly`, `both`; unset means `both` for a genre that cites and `web` otherwise). The packs: `article`, `grant`, `report` scholarly; `memo`, `email`, `howto` web; `proposal` both. `ResearchSetup.mode` (CLI `--research-mode`) overrides it. The reading of "both where the genre cites" is `proposal`, the one cited genre that is not an academic one. Web and both need `ResearchSetup.sources.searxng_url`; without one `ResearchSetupError` is raised before any search.
- **The question is the user's task text.** Nothing derived from a source file (the brief, a fact, a source's text) is sent to a search engine, because a source can be an unpublished draft. The engine plans its own sub-questions from the task. A model-written query from the brief would be better targeted; it would also leak, so it is not done.
- **The store is `<bundle>/research.db`** (with `blobs/` beside it), opened and closed by `research()`; one project, one store, so a resumed or repeated run reuses pages, works and answers. The whole `ResearchResult` is also saved as `evidence/research.json`, which is what makes the node idempotent: a resumed run loads it and never searches again, with no settings needed. A run that was started with research and crashed inside it has no `research.json`; `compose_bundle` then needs the research settings again and raises `ResearchSetupError` naming `--research` if it has none. `meta["research"]` records that research was asked for.
- **A source per work with a finding** (`research_sources`): kind `research`, `Source.work_id` the store id, text = title plus the passages the findings point into (verbatim, in the order they stand in the work), `label` = authors, year, title, venue and DOI or URL (`citation_label`), `path` = a link, never a machine path. The Sources list prints the label.
- **A fact per finding** (`research_facts`): the engine's quote is located again in the source text (`evidence.find_quote`) so `start`/`end` index the file the grounding check reads; a quote that cannot be placed is dropped and logged (`finding_dropped`; a bug signal, since the engine checked it). A finding the engine could not verify (`quote_verified` false: its quote was not found in the page) is never a fact; one note per source counts them ("N unverified quotes not used"). The statement is the claim written as another work's (`attribute`: "Another published work (T) reports, not this study: ..."), because the drafter sees only statements and would otherwise write a published result up as the author's. The harness's literature facts use the same function.
- **Old bundles**: a source of kind `research` with no `work_id` is the legacy report text; it is extracted by the model like any file.
- **Cost accounting**: the engine's token counts are logged as one `Research` call under the drafter role (so `ComposeResult.usage` includes them); Degraded runs and empty runs become open questions in the report. `evidence/research.json` holds the result without its run log (the log, which carries every page's text, stays in `research.db`); `store.load_research` also reads a file written with one.
- `deep_research` is imported only inside `run_research` and `store.load_research` (conformance test); `ResearchSetup` is on the public surface (`from andamentum.compose import ResearchSetup`).

## Layering

compose depends on `core`, `compose.bundle`, `whetstone` (public API), `proofread`, `llm_judge`, `harvest`, `figures`; `deep_research` only behind `--research` (deferred import). Nothing depends on compose except `figures.bundle_glue`, which imports the model-free `compose.bundle` only. It MUST NOT depend on `epistemic` or `document_store`. Only `graph.py` imports `pydantic_graph`; workers are plain async functions with a keyword-only `model=`. `tests/test_dialect_conformance.py` enforces this and runs the dialect check in strict mode.

The writing harness (`harness/`: a format-neutral kernel, `.docx` and markdown adapters, deterministic and model-assisted checks, the `check FILE.docx` verb and the planted-defect benchmark) is documented in `harness/CLAUDE.md`; the CLI's `check-bundle DIR` is the bundle's own one-pass check, `check FILE.docx` is the harness verb. Nothing outside compose imports `compose.harness`.

Files: `schemas.py`, `genres.py`, `brief.py`, `evidence.py`, `plan.py`, `write.py`, `revise.py` (workers); `assess.py` (one check pass, shared by the Check node, `check-bundle` and `eval`); `checks/` (deterministic, grounding, reader, critic); `route.py`; `render.py`; `steps.py` (deterministic bundle helpers and `report.md`); `store.py` (JSON artefacts); `log.py` and `_call.py` (action log and the one model-call seam); `research.py` (research sources, facts and the setup); `graph.py`; `api.py`; `cli.py`; `evals/`; `bundle/` (see below).

## bundle

`compose/bundle/` is the folder-bundle document layer (markdown sections with YAML frontmatter, figures, references; renders to `.docx` / `.md` / `.html` / `.txt`). It is model-free: no `pydantic_ai`, `pydantic_graph`, `core`, `whetstone`, `llm_judge` or compose imports outside `bundle/`, enforced by `tests/test_bundle_boundary.py`. It depends only on `typeset`, `python-docx` and `pyyaml`. Public API: `from andamentum.compose.bundle import Bundle, Section, Reference, ValidationIssue, BundleError, StaleRevisionError`; `Bundle` and `Section` are also on `andamentum.compose`. The no-model CLI verbs (`new`, `sections`, `show`, `edit`, `add`, `validate`, `render --raw`) are in `cli.py`. Full notes (layout, revisions, markdown subset, rendering, validation): `bundle/CLAUDE.md`.

## Pitfalls

- **whetstone sectionises by size.** A rendered docx under about 6000 characters becomes ONE whetstone section titled with the document title, so a critic finding cannot be mapped by section title there. `checks/critic.py` locates each finding from quote offsets (bundle section headings in the reviewed text) and falls back to the title; findings with neither become outline-level items. Do not map by title alone.
- **Token counts come from the model-run usage.** `core.agents.run_agent_with_usage` returns `(output, RunUsage)`; `_call.call_model`, the reader and the grounding judge log `input_tokens` / `output_tokens` from it. Calls made inside `llm_judge.judge_score` and whetstone (critic, reader judge) do not expose usage, so they log null and add 0 to `ComposeResult.usage`; the report's per-role token figure is therefore a lower bound for the `critic` and `judge` roles. When the tool-based attempt fails and the PromptedOutput retry answers, only the retry's tokens are counted.
- **Whetstone is used through its public API only** (`reviewer_profile`, `provenance_line`, `stamp_provenance`, `check_confidentiality`, `review_document`, ...); `test_dialect_conformance` forbids any `andamentum.whetstone.<submodule>` import.
- **The critic reviews what ships.** `render.shipped_view` builds the document a reader receives (citations, Sources list, or markers stripped); delivery and `checks/critic.py` both render through it. A finding that lands in the Sources list has no bundle section and becomes an outline-level item.
- **Sources show `Source.label`, never `Source.path`.** `path` is absolute so a resume works from any directory; the finished document prints the file name or URL (`label`), and for research a "Web research report (urls)" line. A test asserts no absolute path reaches `document.md`.
- **Length failures are section work.** A failed `total_words` check is routed to every section in proportion (`route._length_items`), because only a section revision changes drafted text; it falls back to an outline item only when no section can absorb it.
- **Sources section is on a throwaway copy.** `render_outputs` adds the numbered "Sources" section to a temporary copy of the bundle (rendered with `image_root=<real bundle>` so md/html figure links point at the real `figures/` folder), so the real bundle never gains one and checks never see it.
- **Figures survive revision.** `revise_section` rewrites a whole body; `steps.keep_figures` re-appends figure paragraphs it dropped. A figure the planner asked for with a bad column is reported as an open question and logged (`figure_failed`), not hidden and not fatal.
- **Source ids are `src1..srcN` by input order**; fact ids are `f1..fN` across sources; bundle section ids are `sNN`. They never collide.
- **Resume rounds.** `rounds.json` keeps earlier rounds, so a resumed run continues the numbering (and `reviews/round-NN/`). The no-progress hash history starts empty on resume.
- Facts are capped at 25 per source and only the first 60000 characters of a source go to the model (`source_truncated` is logged).

## Evaluation

`andamentum-compose eval --model M --out DIR [--genre NAME ...|all]` runs the 14 synthetic tasks under `evals/briefs/<genre>/<name>/` (`task.yaml` plus `sources/`; invented organisations only; they ship in the wheel). Arm A is `write_document`; arm B is `evals.baseline.single_call` (one call writes the whole document) in its own bundle, with arm A's brief, sources and facts copied in. Both final bundles get the same `check_bundle` pass; the one asymmetry is that arm A's checks also see its own outline. Arm B is given the same extracted facts and marker rules as arm A, so it cites and the grounding check judges both arms alike (without that, an arm that cites nothing would read as perfectly grounded). The table reports tokens per arm; judge and critic calls log no usage, so those columns are lower bounds. A judge then runs `judge_compare` (both orders). Output: a table, `eval.json`, `eval.md`. Sequential. Live runs cost tokens, so no live test runs by default; real-model tests carry the `compose_eval` marker, deselected like `cloud`.

## Tests

Default suite is offline: workers are stubbed at the model boundary (`tests/_stub.py`) or, for the graph, at the worker names in `graph.py`. `test_integration.py` runs the real workers through the real graph with only the model call and the critic scripted. Synthetic data only (Apollo Labs, Jane Doe, SaltLog, Harbor Foods).
