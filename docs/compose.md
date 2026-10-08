# andamentum-compose: a task in, a finished document out

`andamentum-compose` writes a document from a task. You give it what to write (and optionally files or URLs to draw on); it compiles a brief, extracts evidence from the sources, plans an outline, drafts section by section and revises against independent checks until the goal holds twice running. There is no approval step and nothing is sent anywhere: an email is written as a text file.

The result is a plain **document folder** in the folder you name with `-o/--out`: markdown sections you can read and edit, the evidence, a report, and the rendered files (`.docx`, `.md`, `.html`, `.txt`) in `out/`. The tool writes only into that folder.

Working on an existing Word file (checking it, answering reviewers' comments, refreshing numbers, inserting citations) is the separate tool `andamentum-word`; see `docs/word.md`.

## Install

```bash
uv tool install andamentum          # or: uv add andamentum
andamentum-compose --help
```

Every command that calls a model takes `--model` (any pydantic-ai model id, for example `anthropic:claude-haiku-4-5` or an Ollama model); there is no hidden default. Four jobs are done by a model: the `drafter` writes, the `critic` reviews, the `judge` checks that claims rest on the evidence, and the `reader` answers questions about the finished text. `--model M` serves all four; `--drafter`, `--critic`, `--judge` and `--reader` override it for one job. Using one model for all four works but weakens the independence of the checks.

## Write a document

```bash
andamentum-compose write "A one-page memo on the SaltLog pilot for the steering group" \
  --out memo --genre memo --source notes.md \
  --model anthropic:claude-haiku-4-5
```

`--genre` picks the kind of document: `email`, `memo`, `report`, `article`, `howto`, `proposal` or `grant` (list them with `andamentum-compose genres`, or pass your own `.yaml` file). Without it, the brief compiler chooses. `--source` is repeatable and takes a file (PDF, Word, HTML, markdown, text) or a URL. A source that looks confidential is refused unless you pass `--confirm-own-sources`. The folder named by `--out` must not exist or must be empty.

What you get in `memo/`:

| path | what it is |
|---|---|
| `out/` | the finished document, in the genre's formats (`--format docx --format md ...` to choose) |
| `report.md` | what happened: rounds, the checks, open questions (`[TO FILL: ...]` markers and unmet checks) |
| `sections/*.md` | the sections, one markdown file each, with revision and status in the front matter |
| `brief.json`, `outline.json`, `rounds.json` | the frozen brief, the outline and the check rounds |
| `evidence/` | the sources as read, and the facts extracted from them (each with its verbatim quote) |
| `log.jsonl` | every model call: role, model, tokens |
| `bundle.yaml` | the folder's manifest |

The drafter marks each use of a fact as `[[f12]]`; the finished document turns these into numbered citations and a Sources list. Every fact carries a quote that the code found in the source; a fact whose quote is not in the source is dropped. Provenance (which model wrote which sections, which checks ran) is stamped into the output; `--no-provenance` leaves it out, but is refused for genres that require it (`grant`).

Exit 0 when the goal was met, 1 when the run finished but the goal was not (the stop reason and the open questions are in `report.md`; the files are written), 2 usage error, 3 an input could not be read, 4 the run failed.

### The goal and when a run stops

A round meets the goal when no error-level check failed, the reader got every question right and the critic reported no major finding. It is accepted only when the next check pass over the same text meets it too. The run stops at `goal_met`, at `no_progress` (a revision changed nothing), at `cap_reached` (`--max-rounds`, default 4) or at `unresolvable` (no section got any text).

### Research

`--research` runs a literature or web search for the task and lets the document cite what it finds. The genre picks where it looks: scholarly databases for `article`, `grant` and `report`; the web for `memo`, `email` and `howto`; both for `proposal`. `--research-mode web|scholarly|both` overrides that. Web search needs `--searxng-url ADDRESS` or `--browser` (an isolated Chrome in a podman container that the run starts and removes). Every finding is a claim with a quote copied word for word from a real passage and checked by code; the findings become sources. The search sends your task text (never text from your sources) to the sources and keeps what it learns in `research.db` in the document folder. Contact addresses and keys come from the environment (`OPENALEX_EMAIL`, `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, read from `.env`); `--mailto` sets the address.

## Continue, check, render

```bash
andamentum-compose continue memo --model anthropic:claude-haiku-4-5   # pick up where a run stopped
andamentum-compose status memo                                          # where does it stand?
andamentum-compose check memo --model anthropic:claude-haiku-4-5        # the checks once, no revision
andamentum-compose render memo --format docx                            # render the sections again into memo/out/
```

`continue` skips finished steps and resumes the checks; it takes the models recorded in the folder unless you name others. `--max-rounds` counts the rounds of this run. `check` prints each check as pass or FAIL and exits 1 when the goal is not met. `render` needs no model call; it names the model in the provenance from the folder (or `--model`). `render --plain --format docx -o notes.docx` renders the bare text, with no citations, Sources list or provenance, to one file.

## Read and edit a folder without a model

These commands call no model; an agent or a person can use them to work on the sections directly.

```bash
andamentum-compose new --out paper --title "SaltLog pilot" --scaffold article   # an empty folder with the article sections
andamentum-compose sections paper                    # id, revision, status, words, title
andamentum-compose show paper s01                    # one section in full
andamentum-compose edit paper s01 --from body.md --revision 1   # replace a body (or --from - for stdin)
andamentum-compose add paper --title "Limitations" --after s03
andamentum-compose validate paper                    # structural checks; exit 1 on an error
```

`edit` names the revision you read and is refused if the section has moved on, so two editors cannot overwrite each other.

## Compare with a single model call

```bash
andamentum-compose eval --genre memo --out eval-out --model anthropic:claude-haiku-4-5
```

Runs the synthetic tasks that ship with the package, once through compose and once as one model call, checks both with the same checks and has a judge compare them. Writes `eval.json` and `eval.md` into the folder. It calls a model many times; use a cheap one.

## Python

```python
import asyncio
from andamentum.compose import Roles, write_document

result = asyncio.run(
    write_document("A one-page memo on the SaltLog pilot", out="memo", roles=Roles.all("anthropic:claude-haiku-4-5"))
)
print(result.goal_met, result.outputs)
```

`write_document(task, *, out, roles, ...)` and `continue_document(folder, *, roles, ...)` are async and return a `ComposeResult` (`folder`, `outputs`, `goal_met`, `stop_reason`, `rounds`, `open_questions`, `usage`). The errors derive from `ComposeError`; a bad argument raises `ComposeUsageError`.

## What leaves the machine

- **Model calls**: the text of your task, the sources and the sections goes to the models you name. Use a local (Ollama) model to keep it on the machine.
- **URL sources** you pass are fetched; private and internal addresses are refused.
- **Research** (`--research` only): your task text goes to the search engine you name and to OpenAlex, Europe PMC and arXiv. Run it on a task you are willing to publish.
- Nothing is ever sent by the tool: no mail, no upload.

## Limits

- The checks are only as independent as the models behind them; use different models for `--drafter` and `--critic` when it matters.
- A citation in the finished document says the fact has a verbatim quote in the source. It does not say the source is right.
- Judge and critic calls do not report their tokens, so the usage figures in the report are a lower bound for those roles.
- The `grant` genre is generic: funder specifics enter only through the call text you pass as a source.
