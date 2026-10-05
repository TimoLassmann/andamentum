# harness/draft — notes for Claude

`draft_document(task, *, out, roles, into=None, genre=None, sources=(), results=None, literature=None, template=None, max_rounds=2, offline=False, ...)` is the library behind `andamentum-compose write`. The CLI flags are wired in `verb.py` (the `write` verb: `--out FILE.docx` or `--into` comes here, any other `--out` is the bundle path; `run_write` builds the literature sources, maps `DraftUsageError` to exit 2 and `DraftResult.exit_code` to 0 or 4); this module takes explicit arguments and the four compose `Roles` (the drafter writes, the judge labels claims and runs the numbers check; critic and reader are unused here).

## What it reuses (and where it stops)

The compose workers, unchanged: `compile_brief`, `ingest_sources`, `extract_facts`, `make_outline`, `draft_section`, `revise_section`, the genre packs, `ActionLog` and the JSON store functions (for `brief.json`, `outline.json`, `evidence/*.json` in `<project>/compose/`). No pydantic-graph, no `Bundle`: the `.docx` is the output of record and the bundle's section store is not needed. The checks are the harness `check` verb (`run_check`), the writer is `DocxAdapter.apply`, number matching is the bindings code (`load_catalogue`, `parse_written`, `scale_for`, `NumberBinding`, `Ledger.record`), citations are `InsertCitation` / `UpdateBibliography`, provenance is whetstone's `stamp_provenance`. New code is only what none of those do: the mapping from a drafted body to Word, the sentence analysis, the repair loop's wiring, the literature adapter and the disclosure text.

Two names are imported across modules on purpose, because a second copy would drift: `check.comment_ops` (a diagnostic list as clamped `CommentOp`s) and `bindings.common.write_copy` (apply, then add the project id). If either is renamed, this module breaks loudly.

## Flow (`run.py`)

1. Project folder `<out stem>.harness/` (the harness store; its id is the document's when `--into` already carries one) with a scratch folder `compose/` for the workers. `out` must not exist; the input of `--into` and every source are only read.
2. Sources: the text of the document being extended (as one source), `--source` files and URLs, every csv/tsv/json under `--results` (the source's label is the path relative to the results folder), then verified literature works. One fact-extraction call per source, in order; a fact whose quote is not in the source is dropped by the worker. Source ids are `src1..`, fact ids `f1..` globally.
3. Outline (figures are not drawn into Word and are noted in the report), then targets (below), then one draft per target in outline order, each seeing the first two sentences of the sections before it (written this run or already in the document).
4. `claims.analyse_section` per section (parse, sentences, kinds, number bindings), then `assemble.build_round`, then the check, then the repair loop.
5. `_finish`: remaining diagnostics become comments, the AI statement is added, the file is written to `out` with the project id and provenance stamp, ledger and report are written.

Every model call is sequential. The workers log every call to `<project>/compose/log.jsonl`; `draft_step` events in the project's `events.jsonl` record step, role, model and section and are what the AI statement is built from.

## Targets into an existing document (`choose_targets`)

Into a document the sections written are, in order of precedence: those the task **names** (a planned title or an existing heading that appears in the task as a whole word, matched case-folded with a leading section number removed); else the planned sections that are **missing or empty** in the document. A named heading the outline lacks is added to the outline with a bare plan. A named section that already has text is **appended to** (new paragraphs after its last one; its own text is never touched). Nothing to write is a usage error. A new document writes every planned section.

Placement (`place_sections`): a new section goes after the last paragraph of the nearest earlier outline section that is in the document, else just before the nearest later one, else at the end; sections placed at the same spot keep outline order. Heading level is the document's top heading level (1 in a new document). Styles (`resolve_styles`): `Heading N`, else the style of an existing heading of that level, else an error; body text copies the commonest style of the document's body paragraphs; `List Bullet` / `List Number` when present, else the marker stays in the text.

## The build (`assemble.py`)

The draft is written as **tracked insertions**, also in a new document (Accept All gives the plain draft, Reject All the empty template). Three transactions per build, each applied to a copy by the one safe writer:

1. `InsertParagraph` for every heading and paragraph, text clean (markers cut out). Ids of new paragraphs are unknown before this step; their positions follow from the writer's rule (each new paragraph directly after its anchor, in operation order), computed in `_inserted_ids` and **verified against the text read back**, so a wrong position raises.
2. `Bind` (kind `number`) for bound numbers and `CommentOp` for `[TO FILL: ...]` ranges, interpretation sentences and document-level questions.
3. `InsertCitation` per cite position plus `UpdateBibliography` in the **same** transaction (the writer lists the works of the transaction).

A citation is inserted verbatim by the writer, so the paragraph is written with a space opened at each cite position (`PlacedParagraph.gaps`; `start` / `end` map drafted offsets, a range that ends where a space was opened leaves it outside). If a citation is refused the space stays; the refusal is in `DraftResult.refused` and the exit code is 4. A refused citation adds nothing to the bibliography: the writer plans again without it.

A round is a pure function of the drafted bodies and the base document, so the repair loop **rebuilds from the base** instead of editing its own output. `carry_ids` maps paragraphs of the second file to the third (which only grew), by order and character-subsequence containment, to keep section ownership across the bibliography insertion.

Drafted body to paragraphs (`body.py`): blank-line blocks, one paragraph per list item, a pipe table flattened to one paragraph per row (noted in the report), inline markdown removed (inserted paragraphs are plain text), markers cut out. A cite goes before a sentence's closing punctuation.

## Claims (`claims.py`) and the ledger

Per sentence: markers on a result-file fact make a `result` claim and bind its numbers; markers on a literature fact make it `background`; every other sentence gets its kind from **one judge call per section** (four labels; unlabelled means `interpretation`, logged as `label_missing`: the safe side is to ask the author). Interpretation sentences get a comment and status `needs-author`. A number binds only to a value of a result file one of the sentence's facts came from, equal at the written precision; when several values match and the facts' quotes do not pick one, it is **left unbound and reported**. A placeholder sentence is a question, not a claim; table rows are not claims.

`claims.yaml` (project store, shared with the bindings code): this module appends entries `d1, d2, ...` (id, text, kind, section, evidence, strength, status, approved_by null, anchors) and keeps every entry already in the file; bound numbers go through `Ledger.record` (entries with a `binding` block, which `refresh` reads). Statuses: `supported`, `unsupported` (no evidence), `needs-author`. Evidence pointers: `{type: fact, fact_id, source, quote[, work][, file]}`.

## Typed citations

The harness inserts a citation only from a `[[fN]]` marker on a verified work. Citation-shaped text the drafter typed itself ("Smith (2020)", "(Smith, 2020)", "[3]"; the pattern `TYPED_CITATION` in `kernel/csl.py`, shared with `respond/edits.py`) names a work nothing checked, so it never stands silently: `claims.analyse_section` finds it in the cleaned body text (`SectionAnalysis.typed`), the repair loop sends the section to the reviser with the string named, and what is still there after the last round is an error comment ("[error] Typed citation, not verified") on its range, an entry in `open_errors` (so exit 4) and a round-note error. Its sentence is `unsupported` in the ledger with strength "typed citation, not verified", whatever fact it also quotes. The drafting rules in `write.py` also tell the model not to type citations. The `citations` check separately warns (`unlisted`) about any plain-text author-year citation no reference-list entry names, so a document that already carries one is not clean.

## Check and repair

After every build the harness `check` runs on the file (all checks whose needs are met: tier-0 always, network ones unless `offline`, `numbers` with `results`, using the judge model). Errors located in drafted paragraphs go to `revise_section` as work items per section (whole-section rewrite, one call per section per round); unchanged text or `max_rounds` stops the loop (`max_rounds=0` checks only). Warnings are never repaired. What remains: errors and warnings in drafted paragraphs become comments signed "andamentum compose" (`[error] check: message`); in a new document every diagnostic counts, in `--into` only those located in drafted paragraphs (errors in the author's own text are in the report counts, not in the exit code).

**Exit code:** `DraftResult.exit_code` is 0 when nothing is left for the authors and 4 otherwise: a check error left in the drafted text, a refused operation, or an author question (`[TO FILL]`, an interpretation claim, an unverified work). The report lists each; the questions are also comments. Usage errors raise `DraftUsageError` (the CLI maps it to 2).

## Literature

`draft` never searches or verifies: a `LiteratureSearch` (async callable `(query, *, limit) -> LiteratureFinding`) is passed in; `search.literature_search(sources=, http=)` builds one from `harness.literature` (`find_works`, then `resolve` per work; unverified works become one line each). One query per key message (at most six, sequential), at most 8 works with an abstract; a work with no abstract is verified but unused and listed in the report notes. Only verified works are ever cited; their facts come from title plus abstract. Unverified lines become comments on the first paragraph the draft inserted. Genres with `cites=False` ask nothing.

**Limit that matters:** a verified work backs a fact only through its CSL `abstract`. A library file or Zotero item that carries one works; Crossref search asks for the `abstract` field, but many records have none; a verified work without one is unused and listed in the report notes.

## AI statement (`disclosure.py`)

One paragraph under "Statement on the use of AI", built only from logged events (`draft_step`, `check_run`, `draft_run`); a sentence without its events is left out. The log is cumulative per project, so a second pass names both passes. Into a document that already has the heading the paragraph is appended under it.

## Limits and pitfalls

- Tracked insertions only: nothing existing is rewritten (`Replace` is not used by `write`).
- Inline formatting, tables and figures are not written; a table is flattened, a planned figure is dropped, both reported.
- Anchors for unverified works and other document-level questions are the first inserted paragraph, not a section.
- The confidentiality gate of `ingest_sources` also reads the existing document text; it raises unless `confirm_own_sources` is set.
- An `--into` document's project id must equal that of an existing `<out stem>.harness/`.
- The numbers check and the claim labels use the judge model and add tokens that `DraftResult.usage` (which sums the worker log) does not include for the check.
- Not measured against a real model: tests stub the model boundary and the network. The first live run is the acceptance smoke.

## Tests

`tests/test_draft_body.py` (parsing), `test_draft_claims.py` (kinds, numbers, ledger), `test_draft_assemble.py` (placement, styles, build), `test_draft_disclosure.py`, `test_draft_search.py` (the literature adapter over a fake Crossref), `test_draft_run.py` (end to end: new document, into, Zotero document, repair loop, usage errors); `tests/_draft_helpers.py` has the synthetic SaltLog project and `Script`, a model stub that answers by output type.
