# andamentum.deep_research

An evidence-first research loop over the web and the scholarly literature. A question goes in; a report comes out in which every sentence cites findings by id, and every finding is a claim with a verbatim quote that code found in the page it came from.

When a language model researches a question it can cite pages it never read, quote words the page does not contain, and report a retracted paper as support. This sub-module keeps the model on the decisions (what is missing, what to read, whether the question is answered) and does the checking in code: the searches and fetches, de-duplication of works by identifier, quote location, retraction look-ups and budgets.

## The loop

The model plans (one sub-question when the question asks one thing, at most four when it has parts) and then, per sub-question, repeats a round until it judges the question answered or the round cap (three by default) is reached:

1. it writes search queries;
2. the engine searches the web (a Chrome you start with `--remote-debugging-port`, or a SearxNG instance) and, in scholarly mode, OpenAlex, Europe PMC and arXiv, and merges the results by work identity;
3. the model chooses which results to read from their titles, sites and snippets;
4. each page is read (through the same Chrome when the plain fetch fails), cut to its most relevant characters, and the model notes findings, each a claim with a verbatim quote;
5. the model judges whether the findings answer the question.

One call then writes the report from all findings, committing to the best-supported answer and citing findings as `[F1]`. Code checks that every citation exists, the quotes are still in their passages, the links resolve and no cited work is retracted. A finding whose quote cannot be found is kept and marked `[quote not verified]`.

## Quick start

```python
import asyncio

from andamentum.deep_research import SourceConfig, research


async def main() -> None:
    result = await research(
        "Who wrote the novel Middlemarch?",
        mode="web",
        model="ollama:gemma4:31b-cloud",
        sources=SourceConfig(browser=True),
    )
    print(result.report_markdown)
    print(result.stop_reason, len(result.findings), "findings")


asyncio.run(main())
```

`mode` is `web`, `scholarly` or `both`. Web mode needs the isolated browser (`browser=True`: a Chrome in a podman container the run starts and removes; podman is required and the image is built on first use) or a SearxNG instance (`searxng_url`; the library does not start it). `model` is the one language model the engine uses. A runnable version is `examples/deep_research/quickstart.py`.

## What it provides

- **`research()`** and **`ResearchResult`**: sub-questions, works, passages, findings, the report, the run log and the run's costs. Every bound of a run is a field of **`Budget`**.
- **`ResearchStore`**: one SQLite file and a `blobs/` folder you name (or `ResearchStore.in_memory()`): the HTTP cache every source reads through, the works found, the passages and each run's records. A second run starts from the first one's answers.
- **`SourceConfig`**, **`build_sources()`**, **`SourceClient`**: the adapters (OpenAlex, Europe PMC, arXiv, Semantic Scholar, Crossref, the web, a local corpus) and the shared, rate-limited, cached HTTP client.
- **`run_novelty_check()`**: whether a claim has prior work in the scholarly literature, judged from findings that quote the work; it returns a verdict of novel, not novel or undetermined, never a guess.
- **`Asker`**, **`MODEL_ERRORS`**, **`locate_quote()`**, **`merge()`**: the model-call wrapper, the quote locator and the work de-duplication the other parts are built from.

## CLI

```bash
andamentum-research ask "Who wrote the novel Middlemarch?" --mode web \
  --model ollama:gemma4:31b-cloud --browser
```

`--model` is required. `andamentum-research ask --help` lists every flag: `--store`, `--cache`, `--searxng-url`, `--browser`, `--sources`, `--local-corpus`, the limit flags, `--json`, `-o DIR`. Without `-o` the report goes to stdout and no file is written; with `-o DIR` the folder gets `report.md` (or `report.json`), `log.md` and `log.jsonl`. `andamentum-research mcp --store PATH` serves the engine and its store to an agent over MCP; `cache info` and `cache prune` look at and trim an HTTP cache. Exit codes: 0 clean, 1 finished degraded (result still written), 2 usage error, 3 an input could not be read, 4 the run failed.

With `-o DIR` a run also writes a Markdown log and a JSONL log of what happened at each step, including the text the model saw. `uv run python -m andamentum.deep_research.bench experiment losses ...` reads those logs and says, for each wrong answer, at which stage the right one was lost.

Notes for contributors: `src/andamentum/deep_research/CLAUDE.md`.
