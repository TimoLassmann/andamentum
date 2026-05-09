# Andamentum documentation

Andamentum is a single Python package of tightly-scoped sub-modules for building agentic reasoning pipelines. Each sub-module ships in the same distribution; dependencies are the flat union of what they need.

## Sub-modules

| Module | Role |
|---|---|
| [**epistemic**](./epistemic/overview.md) | Formal-epistemology pipeline: 23-node graph, multi-philosophical verification (Popper, Lakatos, Lipton, Reichenbach, Peirce), calibrated posteriors. The flagship — see [`epistemic_flow.html`](./epistemic/epistemic_flow.html) for the rendered architecture overview. |
| [**deep_research**](./deep_research/overview.md) | Web research pipeline (search → fetch → extract → verify → synthesize) over a local SearxNG instance. |
| [**document_store**](./document_store/overview.md) | SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal RRF search and LLM metadata extraction. |
| **whetstone** | Structured multi-lens review of your own drafts → markdown / HTML / .docx with track changes. |
| **scribe** | Block-based document authoring (paragraph, heading, figure, table) backed by SQLite, one-way render to .docx. |
| **figures** | Publication-quality scientific figure generation (9 chart types, 7 journal palettes, journal-matched sizing). |
| **chunker** | Structural-first semantic chunking of long markdown into 2k–10k char self-contained units. |
| **harvest** | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain). |
| **vision_critique** | Bounded vision critique of rendered figures via local multimodal models. |
| **typeset** | Standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules. |
| **core** | Shared model-resolution, `AgentRunner`, embedding infrastructure used by all sub-modules. |

## Design principles

- **Explicit model argument.** Every public function that calls an LLM takes `model=` as a keyword-only argument. No hidden defaults, no silent fallbacks, no shared config module.
- **One distribution, no extras.** Dependencies are the flat union of what the sub-modules need. There are no optional installs.
- **Public API in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal.
- **Tests next to code.** `pytest.ini_options.testpaths = ["src/andamentum"]` — tests live next to what they test, not in a top-level `tests/` directory.

## Imports

```python
from andamentum.epistemic.graph import run_epistemic_graph
from andamentum.deep_research import ResearchState
from andamentum.document_store import ingest, search
from andamentum.whetstone import review_document
from andamentum.scribe import Document
from andamentum.figures import figure
from andamentum.chunker import extract_units
from andamentum.harvest import extract
from andamentum.vision_critique import critique_figure
```

## CLIs

Eight scripts installed by the package. Run `--help` on any binary for full flag reference.

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-epistemic` | Formal-epistemology pipeline. Two modes: `ask "<question>"` or `verify "<claim>"`. | required |
| `andamentum-research` | Web research over local SearxNG. | required |
| `andamentum-whetstone` | Multi-lens draft review → markdown / HTML / .docx with track changes. | required (or `--no-llm`) |
| `andamentum-scribe` | Block-based document authoring; renders to .docx. | none |
| `andamentum-figures` | Publication-quality scientific figures. | none |
| `andamentum-chunker` | Structural-first semantic chunking. | required |
| `andamentum-harvest` | Universal source → markdown extraction. | none |
| `andamentum-vision-critique` | Vision-critique of a rendered figure → bounded JSON. | required (multimodal) |

`andamentum-epistemic`, `andamentum-research`, and `andamentum-chunker` resolve their LLM via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`. The CLI exits with a clear error if neither is set — no hidden defaults.
