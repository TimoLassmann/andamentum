# Using andamentum

This guide shows what each andamentum tool does and how to start it, with commands you can copy. Every command's `--help` ends with the same examples, and a test checks that each of them is accepted by the command.

## Install

```bash
uv tool install "andamentum @ git+https://github.com/timolassmann/andamentum.git"
andamentum list
```

`andamentum list` prints every tool, what it does and whether it needs a language model. The optional extras (`pdf`, `html-articles`, `vlm`) are explained in the [README](../README.md#installation).

## Find the tool for your job

| I want to ... | Command |
|---|---|
| turn a PDF, Word file, slide deck or web page into markdown | [`andamentum-harvest`](#harvest-any-file-to-markdown) |
| split markdown into pieces for a search index | [`andamentum-chunker`](#chunker-split-markdown) |
| keep a collection of documents and search it | [`andamentum-docstore`](#docstore-store-and-search-documents) |
| research a question on the web and in the literature | [`andamentum-research`](#research-research-a-question) |
| check a claim or answer a question by weighing evidence | [`andamentum-epistemic`](#epistemic-weigh-evidence-for-a-claim) |
| check the citations and reference list of my draft | [`andamentum-references`](#references-check-citations) |
| get a review of my own draft, with tracked changes | [`andamentum-whetstone`](#whetstone-review-your-own-draft) |
| revise my draft until a condition holds | [`andamentum-whetstone revise`](#revise-until-a-condition-holds) |
| check readability and style, without a model | [`andamentum-proofread`](#proofread-readability-and-style) |
| write a new document (email, memo, report, article, proposal, grant) from a task | [`andamentum-compose`](#compose-write-a-new-document) |
| work on a Word file I already have: check it, answer reviewers, refresh numbers, add citations | [`andamentum-word`](#word-work-on-an-existing-word-file) |
| score a text, or compare two texts, against criteria | [`andamentum-llm-judge`](#llm-judge-score-or-compare-texts) |
| draw a chart from a table | [`andamentum-figures`](#figures-charts-from-a-table) |
| check a chart image for layout problems | [`andamentum-vision-critique`](#vision-critique-check-a-chart-image) |
| turn markdown into a styled HTML page or PDF | [`andamentum-typeset`](#typeset-markdown-to-html-or-pdf) |
| build or check an agentic system in the house style | [`andamentum-agentic-dialect`](#agentic-dialect-the-house-style) and [`andamentum-forge`](#forge-a-small-package-from-a-brief) |

`andamentum <tool> ...` is the same as `andamentum-<tool> ...`.

## What every tool has in common

- **Input** is the first argument: a file, a web address, a text or a question. Where the input is text, `-` reads it from stdin.
- **Output.** A tool with one result prints it, and `-o FILE` writes it to a file instead. A tool that makes several files writes all of them into the folder you name with `-o DIR`. No tool writes anywhere else, not into the current folder and not into your home folder.
- **`--json`** prints the result as JSON, for scripts and agents.
- **Models.** A tool that calls a language model needs `--model MODEL`. There is no default model and no environment variable for it. Any [pydantic-ai](https://ai.pydantic.dev/models/) model id works, for example `anthropic:claude-haiku-4-5` or a local Ollama model such as `ollama:gemma4:31b`. A second model has a name that says its job: `--embedding-model`, `--reranker-model`, `--decision-model`, `--figure-model`.
- **API keys** come from the environment, as each provider documents (for example `ANTHROPIC_API_KEY`). They are never flags.
- **`--offline`** means the same in every tool that has it: make no network call besides the model.
- **Presets, not knobs.** The command line gives a few named presets (`--depth quick|standard|deep`, `--reviewer methods`, `--genre memo`, `--journal nature`). Every finer setting is a keyword argument of the tool's Python function.
- **Exit codes** are the same in every tool:

  | Code | Meaning |
  |---|---|
  | 0 | success |
  | 1 | the tool ran and the result is negative: problems found, a condition not met, a check failed; the output is still written |
  | 2 | usage error: a wrong or missing argument |
  | 3 | the input could not be read: missing file, unreachable address, unsupported format |
  | 4 | the run failed: a model or network error, or a crash |

- **Errors** go to stderr as one line that starts with `error:` and says what to do.

The full rules are in [interface.md](./interface.md).

## Sources and search

### harvest: any file to markdown

Reads PDF, HTML, Word, PowerPoint, markdown and plain text, and finds the kind itself. No model in the default `fast` mode. `--mode accurate` (PDF only) also describes the figures; it needs the `vlm` extra and a vision model.

```bash
andamentum-harvest paper.pdf -o paper.md
andamentum-harvest https://example.org/article -o article.md
andamentum-harvest paper.pdf --mode accurate --figure-model ollama:qwen3.8 -o paper.md
```

Python: `andamentum.harvest.extract(source)`.

### chunker: split markdown

Splits markdown at headings, then paragraphs, then sentences. Code blocks, tables and front matter are never cut. No model.

```bash
andamentum-chunker notes.md --max-chars 2000 -o chunks.json
andamentum-chunker notes.md --max-chars 2000 --text-only
```

Python: `andamentum.chunker.chunk(text, max_chars=2000)`.

### docstore: store and search documents

One SQLite file holds your documents, every revision of them and the original bytes. `ingest` and `list` need no model. `sync` builds the search index with an embedding model, and you can stop it with Ctrl-C and start it again to continue. `search` needs the embedding model and a reranker.

```bash
andamentum-docstore ingest store.db paper.pdf notes.md
andamentum-docstore sync store.db --embedding-model embeddinggemma-2:latest
andamentum-docstore search store.db "what limits the method?" \
    --embedding-model embeddinggemma-2:latest --reranker-model BAAI/bge-reranker-v2-m3
```

Also: `get`, `list`, `delete` and `enrich` (add identifiers, facts, bibliographic data, extracted fields or labels). Design and measurements: [document_store/overview.md](./document_store/overview.md). Python: `andamentum.document_store.Store`.

### research: research a question

Plans the question, searches, reads, takes notes and writes a report. Every finding in the report is a quote that the code found in the page. `--mode` chooses where to search: `scholarly` (OpenAlex, Europe PMC, arXiv), `web` or `both`. Web search needs `--browser` (an isolated Chrome in a podman container) or a SearxNG address (`--searxng-url`). `--depth quick|standard|deep` sets how much it reads; `standard` is the default.

```bash
andamentum-research "How much carbon do salt marshes store?" --mode scholarly --model anthropic:claude-haiku-4-5
andamentum-research "How much carbon do salt marshes store?" --mode both --browser --model anthropic:claude-haiku-4-5 \
    --depth deep -o marsh
```

Without `-o` the report goes to stdout. With `-o DIR` the folder gets the report and a log. `--store PATH` keeps what the run found for later runs. Design: [deep_research/overview.md](./deep_research/overview.md). Python: `andamentum.deep_research.research(question, mode=..., model=...)`.

### epistemic: weigh evidence for a claim

`ask` answers a question and `verify` checks one claim. Both collect evidence for and against, and keep the database and report in `-o DIR`. Web evidence needs `--browser` or `--searxng-url`.

```bash
andamentum-epistemic verify "Restored salt marshes store more carbon than bare mudflats." \
    --model anthropic:claude-haiku-4-5 --embedding-model embeddinggemma-2:latest \
    --searxng-url http://127.0.0.1:4070 --json
andamentum-epistemic ask "Does salt marsh restoration increase carbon storage?" \
    --model anthropic:claude-haiku-4-5 --embedding-model embeddinggemma-2:latest --browser -o marsh
andamentum-epistemic inspect marsh
```

Details: [epistemic/overview.md](./epistemic/overview.md).

## Your own drafts

### references: check citations

Checks the citations of a draft against its reference list, in any citation style, then looks up each entry in Crossref: does the work exist, and do the title, year and first author match? Only the reference entries go to Crossref, never the draft. A model is used only for cases the text leaves open. Exit 1 when it finds problems.

```bash
andamentum-references draft.md
andamentum-references draft.md --offline --json -o report.json
andamentum-references draft.docx --model anthropic:claude-haiku-4-5
```

Python: `andamentum.references.check_references(text)`.

### whetstone: review your own draft

Reviews a draft that you wrote. The default run reviews, writes the suggested edits as tracked changes, checks the references and lists what the draft claims as new. The format of each `-o` file comes from its extension: `.md`, `.html` or `.docx`, and you can give `-o` more than once. whetstone refuses material that someone shared with you in confidence, such as a manuscript you review for a journal.

```bash
andamentum-whetstone review draft.docx --model anthropic:claude-haiku-4-5 \
    -o draft.reviewed.docx
andamentum-whetstone review draft.docx --model anthropic:claude-haiku-4-5 \
    --no-editor --offline -o review.md -o draft.reviewed.docx
andamentum-whetstone review draft.md --model anthropic:claude-haiku-4-5 \
    --reviewer methods --brief @guidelines.md -o review.md
```

- `--reviewer` picks a named reviewer: `default`, `claims`, `methods`, `framing`, `reader`, `adversary` or `editor`. `andamentum-whetstone reviewers` lists them. A comma-separated list runs them one after another.
- `--brief TEXT` (or `@FILE`) says what to review against, for example a journal's guidelines.
- `--no-editor` gives comments only, `--no-references` skips the reference check, and `--offline` keeps the reference check but skips the Crossref lookup.
- Other commands: `checklist` (a fixed set of checks and a coverage table that you can compare between versions), `panel` (several simulated experts), `apply-patches` and `verify-provenance` (no model).

Full guide: [whetstone.html](./whetstone.html).

### revise: until a condition holds

`revise` reviews, edits and reviews again until a condition holds, for at most 4 rounds. Give the condition in your own words with `--until`, or choose a fixed check with `--until-clean structure|citations|figures`. A condition counts as met only when it passes two rounds in a row. Exit 1 when the run stops before the condition holds; the revised draft is still written.

```bash
andamentum-whetstone revise draft.docx --model anthropic:claude-haiku-4-5 \
    --until "the introduction motivates the study" -o draft.revised.docx
andamentum-whetstone revise draft.docx --model anthropic:claude-haiku-4-5 \
    --until-clean structure -o draft.revised.docx
```

`--until-clean structure` holds when the review finds no major or moderate problem; pair it with `--reviewer reader` for the flow of the text. `--until-clean citations` holds when every citation matches the reference list. `andamentum-whetstone revise --help` also shows the figure check.

### proofread: readability and style

Readability scores (SMOG, Flesch-Kincaid) and style checks (weasel words, passive voice, duplicate words, weak openers, adverbs). No model, and the same text always gives the same result.

```bash
andamentum-proofread draft.md
andamentum-proofread draft.md --json -o report.json
andamentum-proofread talk.pptx --slides --allow MEK,RAS
```

Python: `andamentum.proofread.analyze(text)`.

## Writing

### compose: write a new document

Give a task and get a finished document. compose writes a brief, collects the evidence from your sources, plans an outline, drafts each section and revises against independent checks. There is no approval step and nothing is sent anywhere: an email is written as a text file. The result is a folder of markdown sections plus `.docx`, `.md`, `.html` and `.txt` files.

```bash
andamentum-compose write "A memo on the SaltLog pilot" -o memo --model anthropic:claude-haiku-4-5
andamentum-compose write "A grant application for the Apollo study" -o grant --genre grant --research scholarly --model anthropic:claude-haiku-4-5
andamentum-compose genres
```

`--genre` is the preset: `email`, `memo`, `report`, `article`, `howto`, `proposal` or `grant`. `--research auto|web|scholarly|both` adds a research run. Full guide: [compose.md](./compose.md).

### word: work on an existing Word file

Every command writes a copy into `-o DIR`. Your file is never changed. `check` looks at citations, retractions, web addresses, cross-references, statistics and numbers. `respond` answers comments and reviewer reports. `bind-numbers` and `refresh` keep the numbers in the text the same as your result files. `draft` writes new text into the file as tracked changes, then cites it: each uncited background sentence gets a paper that was read and supports it, and the papers read are kept in a document store (`--store PATH`).

```bash
andamentum-word check draft.docx -o checked
andamentum-word respond draft.docx --reviews reviews.txt -o answered --model anthropic:claude-haiku-4-5
andamentum-word draft "A methods paragraph on the SaltLog assay" --into draft.docx -o drafted --model anthropic:claude-haiku-4-5
```

`andamentum-word mcp` gives the same commands to Claude Code and Claude for Word. Full guide: [word.md](./word.md).

### llm-judge: score or compare texts

Scores one text against criteria (your own in a JSON file, or six built-in ones), or compares two texts in both orders. One `--model` is one judge; give `--model` more than once for a panel that must agree. Every result says how confident it is and whether a person should look.

```bash
andamentum-llm-judge score "SaltLog records salinity every ten minutes." --model anthropic:claude-haiku-4-5
andamentum-llm-judge compare "Paris is the capital of France." \
    "The capital of France is Lyon." --context "What is the capital of France?" --model anthropic:claude-haiku-4-5
```

More: [llm_judge.html](./llm_judge.html).

## Figures and pages

### figures: charts from a table

Draws a chart from a CSV or JSON table. No model. `--kind` is the chart type (`bar`, `line`, `scatter`, `box`, `violin`, `histogram`, `heatmap`, `strip`, `swarm`, `stacked`; `auto` by default). `--journal` sets size and font for `nature`, `science`, `cell`, `plos`, `showcase` (slides) or `report`. The format follows the `-o` extension.

```bash
andamentum-figures results.csv --kind bar --x group --y value -o figure.svg
andamentum-figures results.csv --x group --y value --journal showcase -o slide.png
andamentum-figures plot results.csv --x group --y value --journal report --style andamentum --svg-inline -o figure.svg
```

`andamentum-figures palettes` and `andamentum-figures journals` list the presets.

### vision-critique: check a chart image

A vision model looks at a rendered figure and reports overlapping labels, legibility, legend placement and aspect ratio, with fixes from a fixed list. Validated local model: `ollama:qwen3.8`.

```bash
andamentum-vision-critique figure.png --model ollama:qwen3.8
andamentum-vision-critique figure.png --model ollama:qwen3.8 \
    --context "a two-column journal figure" -o critique.json
```

### typeset: markdown to HTML or PDF

Renders markdown in one of three styles: `article`, `cv` or `report`. The format follows the `-o` extension. PDF needs the `pdf` extra.

```bash
andamentum-typeset notes.md -o notes.html
andamentum-typeset notes.md --style report -o notes.pdf
```

## Building agentic systems

### agentic-dialect: the house style

The house style for systems built on pydantic-graph and pydantic-ai: nine laws, a short brief for each kind of step, a checklist, a skeleton to copy, and `check`, which tests code against the laws (exit 1 when it finds a violation).

```bash
andamentum-agentic-dialect check src/mypipeline
andamentum-agentic-dialect role worker
andamentum-agentic-dialect law L7
```

### forge: a small package from a brief

Turns a plain-language brief into a small, runnable function package in the house style and tests it in a sandbox (podman by default). It refuses apps, agents and services.

```bash
andamentum-forge build "Count the words in a text and return the ten most common." \
    --model anthropic:claude-haiku-4-5 -o wordcount
andamentum-forge design "Count the words in a text and return the ten most common." \
    --model anthropic:claude-haiku-4-5 --json
```

Design: [forge-architecture.html](./forge-architecture.html).

## From Python

Each tool has one main Python function, which `andamentum list` names. The first argument is the input, everything else is a keyword argument, and a function that calls a model needs `model=`. Python has every setting; the command line has the presets.

```python
import asyncio
from andamentum.whetstone import review_document

result = asyncio.run(
    review_document("draft.md", model="anthropic:claude-haiku-4-5", confirm_own_draft=True)
)
print(result.summary)
```
