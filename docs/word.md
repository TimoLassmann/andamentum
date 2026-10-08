# andamentum-word: working on a Word file

`andamentum-word` drafts, checks and maintains a document as a Word file. It writes new text, citations and numbers as tracked changes and comments in a copy of your `.docx`; the file you give it is never modified. It checks citations (they exist, match their record, are not retracted), cross-references, statistics and the numbers in the text against your result files; it keeps those numbers in sync when the results change; and it works through collaborators' and reviewers' comments, replying in the thread and changing the text where a change is asked for. Claude Code and Claude for Word can drive the same operations over MCP. Nothing is ever sent anywhere by the tool itself.

(Writing a finished document from a task into a folder of markdown sections is the separate tool `andamentum-compose`; see `docs/compose.md`.)

Every command that writes puts its files into the folder named by `-o/--out DIR` and nowhere else: the edited copy of your file, the report, and a `state/` folder with what a later run into the same folder picks up. Nothing is written next to the input, in the current folder or in your home folder. A command that only prints (`find-citation`) writes nothing.

All examples below use the synthetic SaltLog project that ships with the package (an invented pilot study; every name and DOI is made up).

## Install

```bash
uv tool install andamentum          # or: uv add andamentum
andamentum-word --help
```

The commands that call a model take `--model` (any pydantic-ai model id, for example `anthropic:claude-haiku-4-5` or an Ollama model); there is no hidden default. `check` needs no model for its default checks.

In Python each command is one async function in `andamentum.word`: `check_document`, `respond_to_reviews`, `refresh_numbers`, `bind_numbers`, `draft_document`, `find_citation` and `insert_citation`. They take the file first, `out=` (the folder) and the options by keyword, and return a typed result; the errors derive from `WordError`.

```python
import asyncio
from andamentum.word import check_document

run = asyncio.run(check_document("collaborator.docx", out="checked", results="results"))
print(run.report.counts, run.output)
```

## Exit codes

| code | meaning |
|---|---|
| 0 | success: nothing left for you |
| 1 | the run finished and the result is negative: errors found in the document, items left for the authors, a work that did not verify. The output is still written |
| 2 | usage error: a wrong or missing argument |
| 3 | an input could not be read: a missing file or folder, a file that is not a readable `.docx` |
| 4 | the run failed: a model or network error, a crash |

Errors go to stderr as one line that starts with `error:`. `--json` on a command prints its result as JSON instead of text.

## The example project

```bash
uv run python -m andamentum.compose.harness.examples saltlog-demo
cd saltlog-demo
ls
# collaborator.docx  notes.md  refs.json  results/  reviews.txt
```

| file | what it is |
|---|---|
| `notes.md` | the author's working notes: design, results, open points |
| `results/summary.csv`, `results/intake.csv` | the numbers the paper reports |
| `refs.json` | three invented works as CSL JSON (a reference library) |
| `collaborator.docx` | a draft with Zotero citations, a threaded comment, a resolved comment, two open questions, and a hand-edited number (48 participants changed to 52) |
| `reviews.txt` | two invented reviewers' numbered points |

## The everyday commands

`andamentum-word --help` lists every command with an example, and each command's `--help` shows its flags.

### check

Run the checks on a Word file and get a commented copy plus a report.

```bash
andamentum-word check collaborator.docx --results results --out checked
# commented copy: checked/collaborator.checked.docx
# report: checked/collaborator.checked.md and checked/collaborator.checked.json
```

The default checks are deterministic and need no model: cross-references, statistics (recomputes p from the stated test statistic), abstract numbers against the body, citations (existence, metadata, retraction notices, URL health) and structure limits from `state/checks.yaml`. `--results DIR --model M` adds the numbers check; `--decision-model M` adds the uncited-claim check (advisory). `--offline` drops everything that needs the network. On the example, the report flags that `52` appears in the abstract and nowhere else: the hand edit.

Exit 0 no errors, 1 errors found.

### draft

Draft a document, or sections of one, into Word.

```bash
andamentum-word draft "Results and discussion for the SaltLog pilot" \
  --out drafted --genre article \
  --source notes.md --results results --library refs.json \
  --model anthropic:claude-haiku-4-5
```

The document is `drafted/draft.docx`. Without `--into` it creates a new `.docx` (from `--template journal.dotx` when given). With `--into collaborator.docx` it drafts the sections named in the task, or the missing ones, as tracked changes in a copy and leaves existing text untouched. Facts that came from a literature work become citations (a Zotero field when the document or `--zotero` says so, otherwise formatted text plus a References section); numbers from `--results` are bound to their result file so `refresh` can keep them current. Every work that enters the document must resolve to a real, non-retracted DOI; one that does not is never cited and becomes a question to you. After drafting it runs `check`, revises the sections with errors (at most 2 rounds) and anchors the rest as comments. A closing paragraph under "Statement on the use of AI" says which sections were drafted by which model and which checks ran.

The works in the example `refs.json` have invented DOIs, so a live `draft` finds them unverifiable and cites nothing; it lists each as a question. A library with real DOIs is cited. `--zotero` reads your library from the local Zotero API (`http://127.0.0.1:23119/api/`, read only); the tool never writes to Zotero. `--offline` makes no network call besides the model, so nothing can be verified and nothing is cited; a web address as `--source` is refused with it. `--confirm-own-sources` is needed when the text of `--into` or a source looks confidential. Only one model is needed for Word: `--model M` drafts, labels the claims and checks the numbers. Exit 0 when nothing is left for you, 1 when the file is written but a check error, a refused operation, or a question remains (an unverified work, a `[TO FILL]` placeholder, an interpretation claim). The output folder must not already hold a `draft.docx`: the tool never overwrites a file.

#### Research

`--research WHERE` runs a literature or web search for the task (`deep_research`) and lets the draft cite what it finds. `--research auto` lets the genre pick where it looks: scholarly databases (OpenAlex, Europe PMC, arXiv) for `article`, `grant` and `report`; the web (your SearxNG, `--searxng-url`, or `--browser`, an isolated Chrome in a podman container that the research starts and removes) for `memo`, `email` and `howto`; both for `proposal`. `--research web`, `--research scholarly` or `--research both` names the place instead. The language model plans the search, chooses the pages to read and notes findings; only findings whose quote the engine found in the page are used, and the report says how many were left out.

What the draft gets is not a summary. Every finding is a claim with a quote copied word for word from a passage of a real work, checked by code against the text that was fetched; the quote is looked up again in the source file the draft reads, so a `[[f12]]` marker leads to words on a page. Each work that has a finding becomes a source, and its citation is the label in the Sources list. In Word a work is cited only when it has a DOI that resolves in Crossref, matches its record and is not retracted (the same check as a `--library` work); a work with no DOI (a web page) is listed in the report notes and not cited. With `--research` the new works come from the search and not from a Crossref text search; `--library` and `--zotero` are still searched first. `--research` cannot be combined with `--offline`.

The search sends your task text (what you typed, never text from your sources) to the sources, and keeps everything it learns in `state/research.db` in the output folder with a `blobs/` folder beside it: a repeat run into the same folder reuses pages and answers. Contact addresses and keys come from the environment (`OPENALEX_EMAIL`, `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, read from `.env`), `--mailto` sets the address. A keyless OpenAlex account has 1000 free credits a day and one scholarly run can use around half of them; a free key lifts the limit.

### refresh

Bring the numbers in a document up to date after the results change.

```bash
andamentum-word refresh bound/collaborator.bound.docx --results results --out bound
```

Every bound number is recomputed from the result files. A changed value becomes a tracked change in the original precision and units, and the claim that cites it gets a comment ("value changed 48 -> 52; check the claim still holds"). A change that flips a significance statement (p crossing .05, a confidence interval crossing 0 or 1) is flagged as an error. Numbers that are not bound are left alone. The report lists every change. Writes `collaborator.bound.refreshed.docx` and a report. The bindings are read from `state/` in the output folder, so give the folder that `bind-numbers` or `draft` wrote them to. Exit 0 no errors, 1 when a change flips a significance statement or a bound number cannot be found any more.

### bind-numbers

Bind the numbers in an existing document to your result files, so `refresh` can maintain them.

```bash
andamentum-word bind-numbers collaborator.docx --results results --out bound \
  --model anthropic:claude-haiku-4-5
```

A model proposes which sentence number comes from which result file and key; code then checks that the value matches at the written precision before anything is bound. Every other number of the paragraphs it looks at is listed as not bound, with the reason; none is guessed. Writes `bound/collaborator.bound.docx` (the numbers sit in hidden content controls), a report and the ledger in `bound/state/`.

### respond

Work through the comments in a document, and through a reviews file.

```bash
andamentum-word respond collaborator.docx --reviews reviews.txt --out answered \
  --model anthropic:claude-haiku-4-5
```

Each unresolved comment thread (the tool's own comments excluded) and each numbered reviewer point is sorted into one of four kinds: `edit` (a concrete change is requested: the text is revised as a tracked change, checked, and the thread gets a reply saying what changed and why), `answer` (a reply with the answer and where in the document or evidence it comes from), `author` (a reply "Needs the authors: ..." and an entry in the report: judgement or new work is required) and `done` (already addressed). It never resolves or closes a human comment. `--model` is always needed when there is something to answer. `--decision-model M` uses a small local Ollama decision model to choose the kind instead of the language model. When it chooses `author` or `done` the language-model triage call is saved (the reply is templated; a `done` points at the best matching sentence of the document, or becomes `author` when none matches); `edit` and `answer` still use the language model for the quote and the text. A reviews file (`.txt`, `.docx` or `.pdf`) also produces `response-letter.docx` and `response-letter.md` in the output folder (the command refuses to run when either already exists, because you may have edited it; move or rename it first): one entry per point, with the point quoted, the response and where the tracked change is. It also re-checks paragraphs a collaborator changed since the last snapshot in `state/`. Writes `collaborator.responded.docx` and a report. Exit 0 everything handled, 1 written but items for the authors remain.

### find-citation, paragraphs and insert-citation

Find a work for a claim and cite it, with the work verified first.

```bash
andamentum-word find-citation "smartphone food diaries reporting accuracy" --library refs.json
andamentum-word insert-citation collaborator.docx --paragraph-id p9-10000009 --at 105 --key lindqvist2022 --library refs.json --out cited
```

`find-citation` prints the verified works, which may be cited, and the others with the reason they may not (`--json` for the full records); it writes nothing. `insert-citation` verifies each work (it exists in Crossref, its details agree, it is not retracted), then writes the citation and its reference-list entry as tracked changes in `cited/collaborator.cited.docx`. If any work does not verify, or the writer refuses the place, nothing is written (exit 1); an `--at` outside the paragraph is a usage error (exit 2). A reviewer point that asks for a new citation is left to the authors by `respond`: cite with `insert-citation`, which verifies first. The citation goes in verbatim at character offset `--at` of the paragraph text, so you choose the spacing; paragraph ids come from `andamentum-word paragraphs FILE.docx` (`--json` for programs; it reads the file, writes nothing and needs no model). In Python: `await list_paragraphs(path)`.

### mcp

Let Claude Code or Claude for Word drive the same operations.

```bash
andamentum-word mcp
```

This serves nine tools over stdio: `open_document`, `show_section`, `apply_edits` (the safe writer; it does not write citations), `check`, `draft`, `refresh`, `respond`, `find_citation` and `insert_citation`; the last report is a resource. Every tool writes to a copy, and a call that would change its input file fails. In a tool call you name the copy with `out` (a `.docx` path); the tools keep `state/` beside it unless you pass `project`.

## Connect Claude Code

```bash
claude mcp add word -- andamentum-word mcp
```

Then, in a Claude Code session, ask in plain words: "Open collaborator.docx, check it against the results folder, and fix what you can as tracked changes." Claude calls the tools above; you review the result in Word.

## Output files

Everything is in the folder named by `--out`.

| file | made by | what it is |
|---|---|---|
| `<name>.checked.docx` | check | copy of your file with one comment per finding, signed "andamentum check" |
| `<name>.checked.md`, `<name>.checked.json` | check | the report: counts, checks run, skipped checks and why, network calls, tokens, time |
| `draft.docx` | draft | the new document: every paragraph a tracked insertion, comments signed "andamentum compose", a "Statement on the use of AI" |
| `<name>.bound.docx` | bind-numbers | copy with the numbers bound, signed "andamentum bind-numbers" |
| `<name>.refreshed.docx` | refresh | copy with changed numbers as tracked changes and comments signed "andamentum refresh" |
| `<name>.responded.docx` | respond | copy with tracked edits and replies signed "andamentum respond" |
| `<name>.cited.docx` | insert-citation | copy with the citation and bibliography entry as tracked changes |
| `<name>.bound.md`, `<name>.refreshed.md`, `<name>.responded.md` (and `.json`) | bind-numbers, refresh, respond | the report of that run |
| `response-letter.docx`, `response-letter.md` | respond (with `--reviews`) | one entry per reviewer point |
| `state/` | all | what a later run into the same folder picks up: `project.yaml` (id), `events.jsonl` (what ran, with which model), `claims.yaml` (the claims ledger: text, kind, evidence, strength, status; interpretation claims wait for you), `checks.yaml` (optional limits you write yourself), `cache/` (checks and HTTP answers), `view/` (a private git history of the document text, used to spot collaborators' edits) |

One output folder belongs to one document: `state/` keeps the identity of the document it was started for, and a different document is refused. The `state/` folder uses its own git history; it never reads or writes your global git configuration.

## What leaves the machine

- **Model calls**, only for the commands and flags that take a model: the text of the paragraphs or comments being worked on goes to the model you name. Use a local (Ollama) model to keep it on the machine. `check` with no `--model` makes no model call.
- **Crossref** (through `andamentum.references`): the titles, authors and DOIs of cited works, and DOIs for retraction notices. Your own text is not sent. A contact address goes only with `--mailto`. `--offline` turns this off. `draft` also sends the key messages of the brief as search queries.
- **Research** (`draft --research` only): your task text goes to the search engine you name (SearxNG, which forwards it to public search engines, or the Chrome you name, which searches DuckDuckGo and Bing) and to OpenAlex, Europe PMC and arXiv; the works found are fetched. Nothing from your sources or your draft is sent. Run it on a task you are willing to publish.
- **URL probes**: the links already in your document, requested to see if they still work (private and internal addresses are refused). `--offline` turns this off.
- **Zotero**: the local API on your own machine; nothing is sent out, and nothing is ever written to Zotero.
- **MCP**: stdio only; the server listens on no network port.

Answers from Crossref and the URL probes are cached in `state/`, and research keeps its own store there (`research.db`), so repeat runs into the same folder make fewer requests.

## Limits

- Word desktop and Word on the web have not been tested with the hidden content controls that bind numbers and claims; if Word drops one, that number is no longer bound and `refresh` will not update it. The file is checked by re-reading it after every write, with Word-independent tools (python-docx, LibreOffice).
- Zotero fields are written for Zotero; EndNote fields are read (for checking) but not written.
- `statistics` recomputes p for t, F, chi-square, r and z tests only; not regression coefficients, tables or corrections.
- `numbers` and `bind-numbers` find a value only if it is stored in a `.csv`, `.tsv` or `.json` result file at the written precision; derived values (a difference the text computes) are not found. At most 80 paragraphs per run.
- A citation check says the cited work exists and matches its record. It does not say the work supports the sentence; that stays your judgement.
- `needs_citation` is advisory: it flags a sentence when the decision model's probability of "yes" is at least 0.5, and nobody has measured how often that is right.
- Research has not been run against live sources and models in this build; its tests use an invented world. Findings are only as good as the passages the engine read: an abstract is a short passage, and a work whose full text is behind a paywall is read from its abstract.
- Interpretation claims are never approved by the tool: they stay marked "needs author".
