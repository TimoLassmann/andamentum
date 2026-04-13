# andamentum

Composable agentic systems for scientific automation.

Andamentum is a single Python package containing three tightly-scoped sub-modules
for building agentic reasoning pipelines:

- **`andamentum.epistemic`** — formal epistemology, claim analysis, multi-agent verification
- **`andamentum.deep_research`** — web research pipeline with iterative search, verification, and synthesis
- **`andamentum.document_store`** — SQLite + FTS5 + vector search storage with automatic chunking and LLM metadata extraction

## Installation

```bash
pip install andamentum
```

Everything works out of the box. There are no optional extras to remember.

## Quickstart

```python
from andamentum.epistemic import EpistemicRepository
from andamentum.deep_research import ResearchState
from andamentum.document_store import ingest, search
```

Each public function takes `model=` as a keyword-only argument. The Python API has
no hidden defaults — you always specify which model to use.

```python
from andamentum.epistemic.runner import DefaultAgentRunner

runner = DefaultAgentRunner(model="anthropic:claude-haiku-4-5")
```

## Command-line tools

Two CLIs are installed with the package:

```bash
andamentum-epistemic --help
andamentum-research --help
```

Set `ANDAMENTUM_MAIN_LLM_MODEL` in your environment to avoid passing `--model` on
every invocation:

```bash
export ANDAMENTUM_MAIN_LLM_MODEL=anthropic:claude-haiku-4-5
```

## Documentation

See [`doc/`](./doc/) for module-level narrative documentation and
[`examples/`](./examples/) for runnable code demonstrating common workflows.

## License

MIT. See [`LICENSE`](./LICENSE).
