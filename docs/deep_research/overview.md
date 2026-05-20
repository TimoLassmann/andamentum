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
