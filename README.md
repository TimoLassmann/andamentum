# andamentum

Composable agentic systems for scientific automation.

Andamentum is a single Python package containing three tightly-scoped sub-modules
for building agentic reasoning pipelines:

- **`andamentum.epistemic`** — formal epistemology, claim analysis, multi-agent verification
- **`andamentum.deep_research`** — web research pipeline with iterative search, verification, and synthesis
- **`andamentum.document_store`** — SQLite + FTS5 + vector search storage with automatic chunking and LLM metadata extraction
- **`andamentum.whetstone`** — sharpen your own drafts with editing, specialist review, or multi-expert panel feedback (track changes, HTML, or markdown output)

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

Eight CLIs are installed with the package. Run `--help` on any of them for the
full flag reference.

| Command | What it does |
|---|---|
| `andamentum-epistemic` | Formal-epistemology pipeline — ask a question, get a graph-evaluated answer with evidence, claims, and uncertainty tracking |
| `andamentum-research` | Web-research pipeline (search → fetch → extract → verify → synthesise) |
| `andamentum-whetstone` | Multi-lens review of your own draft → markdown / HTML / .docx with track changes. `--apply-patches PATCHES.json` applies a pre-built JSON patch list to a .docx (no LLM) |
| `andamentum-scribe` | Block-based document authoring backed by SQLite; renders to .docx |
| `andamentum-figures` | Publication-quality scientific figures (9 chart types, journal-matched sizing) |
| `andamentum-chunker` | Verifiable semantic chunking of long markdown into 2k–10k char self-contained units |
| `andamentum-harvest` | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain, auto-detected) |
| `andamentum-vision-critique` | Vision-critique a rendered figure → bounded JSON (label overlap, legibility, suggested fixes). Multimodal model required |

`andamentum-epistemic`, `andamentum-research`, `andamentum-whetstone`,
`andamentum-chunker`, and `andamentum-vision-critique` need an LLM. Set
`ANDAMENTUM_MAIN_LLM_MODEL` once to avoid passing `--model` on every invocation:

```bash
export ANDAMENTUM_MAIN_LLM_MODEL=anthropic:claude-haiku-4-5
```

`andamentum-scribe`, `andamentum-figures`, and `andamentum-harvest` have no LLM
dependency.

## Documentation

See [`doc/`](./doc/) for module-level narrative documentation and
[`examples/`](./examples/) for runnable code demonstrating common workflows.

## License

MIT. See [`LICENSE`](./LICENSE).
