# andamentum.deep_research

Structured web research with source verification, circuit breakers, and novelty checking.

When language models research topics on the web, they face three reliability problems: hallucinated sources (citing URLs never accessed), cascading failures (one service going down takes everything with it), and unverified novelty claims. This sub-module provides data contracts, verification tools, and a complete research pipeline to address these problems.

## What it provides

- **`ResearchState`** — typed state object tracking query, iterations, phase, and completion
- **`verify_sources()`** — checks which cited sources were actually accessed during research, reporting hallucination rate
- **`CircuitBreaker`** / **`CircuitOpenError`** — prevents cascading failures by opening the circuit after repeated service errors
- **`check_novelty()`** — assesses whether a claim has prior art, returning a `NoveltyAssessment` with confidence and similar works
- **`run_research()`** — full pipeline: iterative web search, evidence extraction, synthesis, and source verification

## Installation

```bash
pip install andamentum
```

## Quick start

```python
from andamentum.deep_research import ResearchState, CircuitBreaker, verify_sources
```

For the full research pipeline (requires a running SearXNG instance and an LLM):

```python
from andamentum.deep_research.orchestrator import run_research

result = await run_research(
    "What is spaced repetition and does it work?",
    model="anthropic:claude-haiku-4-5",
    max_iterations=3,
)
report = result["output"]
print(report.evidence_summary)
print(f"Verification rate: {result['verification']['verification_rate']:.0%}")
```

## CLI

```bash
andamentum-research "What is spaced repetition?" --model anthropic:claude-haiku-4-5
```

Or set `$ANDAMENTUM_MAIN_LLM_MODEL` to avoid passing `--model` on every call.

See [`examples/deep_research/quickstart.py`](../../examples/deep_research/quickstart.py) for runnable examples that work without an LLM or SearXNG.

## The research loop (`deep_research`)

`andamentum.deep_research` is the newer engine. The model plans (one sub-question when the question asks one thing, at most four when it has parts) and then, per sub-question, repeats a round until it judges the question answered or the round cap (three by default) is reached: it writes search queries, the engine searches the web (a Chrome you start with `--remote-debugging-port`, or SearxNG) and, in scholarly mode, OpenAlex, Europe PMC and arXiv; the model chooses which results to read from their titles, sites and snippets; each page is read (through the same Chrome when the plain fetch fails), cut to its most relevant twelve thousand characters, and the model notes findings, each a claim with a verbatim quote that code checks against the page; the model then judges whether the findings answer the question. One call writes the report from all findings, committing to the best-supported answer and citing findings as `[F1]`; code checks that every citation exists, the quotes are still in their passages, the links resolve and no cited work is retracted. A finding whose quote cannot be found is kept and marked `[quote not verified]`.

```bash
andamentum-research v2 "Who wrote the novel Middlemarch?" --mode web \
  --model ollama:gemma4:31b-cloud --browser-cdp-url http://127.0.0.1:9222
```

Every run writes a Markdown log and a JSONL log of what happened at each step, including the text the model saw. `uv run python -m andamentum.deep_research.bench experiment losses ...` reads those logs and says, for each wrong answer, at which stage the right one was lost. Notes for contributors: `src/andamentum/deep_research/CLAUDE.md`.
