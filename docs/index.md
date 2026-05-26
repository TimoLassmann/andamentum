# Andamentum documentation

Andamentum is a single Python package of tightly-scoped sub-modules covering the document-handling pipeline a researcher uses end-to-end. Each sub-module ships in the same distribution; dependencies are the flat union of what they need.

## Sub-modules

| Module | Role |
|---|---|
| **scribe** | Block-based document authoring (paragraph, heading, figure, table) backed by SQLite, one-way render to .docx. |
| **figures** | Publication-quality scientific figure rendering — deterministic plotting of your data with journal-matched sizing (9 chart types, 7 journal palettes). |
| **typeset** | Standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules. |
| **whetstone** | Criterion-cascade review of your own drafts → markdown / HTML / .docx with track changes. Subcommands: `review` (default), `panel`, `proofread`, `apply-patches`. |
| **proofread** | Deterministic readability + style checking (SMOG, Flesch–Kincaid, weasel words, passive voice). No LLM. |
| **vision_critique** | Bounded vision critique of rendered figures via local multimodal models. |
| **harvest** | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain). |
| **chunker** | Structural-first semantic chunking of long markdown into 2k–10k char self-contained units. |
| [**document_store**](./document_store/overview.md) | SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal RRF search and LLM metadata extraction. |
| [**deep_research**](./deep_research/overview.md) | Web research pipeline (search → fetch → extract → verify → synthesize) over a local SearXNG instance. |
| **core** | Shared model resolution, agent runners, fetch gate, audit log, and embedding clients. |

An additional experimental sub-module ships installed but is not yet publicly documented; see the *Pre-release / experimental* section of [`../README.md`](../README.md).

## Design principles

- **Responsible by construction.** Fetch gate (robots.txt + paywalled-publisher tripwire), whetstone confidentiality tripwire, AI-provenance watermarks, opt-in audit log, cloud-vs-local awareness — implemented as in-code refusals and stamps, not markdown disclaimers. See [`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).
- **Explicit model argument.** Every public function that calls an LLM takes `model=` as a keyword-only argument. No hidden defaults, no silent fallbacks, no shared config module.
- **No environment-variable configuration.** Configuration flows top-down from the CLI / caller through explicit keyword arguments. Env vars are ambient state that hides at the call site.
- **One distribution, no extras.** Dependencies are the flat union of what the sub-modules need. There are no optional installs.
- **Public API in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal.
- **Tests next to code.** `pytest.ini_options.testpaths = ["src/andamentum"]` — tests live next to what they test, not in a top-level `tests/` directory.

## Imports

```python
from andamentum.whetstone import review_document
from andamentum.scribe import Document
from andamentum.figures import figure
from andamentum.proofread import analyze
from andamentum.harvest import extract
from andamentum.chunker import extract_units
from andamentum.document_store import ingest, search
from andamentum.deep_research import ResearchState
from andamentum.vision_critique import critique_figure
```

## CLIs

Eight scripts installed by the package. Run `--help` on any binary for full flag reference.

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-scribe` | Block-based document authoring; renders to .docx. | none |
| `andamentum-figures` | Publication-quality scientific figure rendering (deterministic plotting). | none |
| `andamentum-whetstone` | Criterion-cascade draft review → markdown / HTML / .docx with track changes. Four subcommands: `review` (default), `panel`, `proofread`, `apply-patches`. | required for `review` + `panel` |
| `andamentum-proofread` | Deterministic readability + style check. | none |
| `andamentum-harvest` | Universal source → markdown extraction. | none |
| `andamentum-chunker` | Structural-first semantic chunking. | required |
| `andamentum-vision-critique` | Vision-critique of a rendered figure → bounded JSON. | required (multimodal) |
| `andamentum-research` | Web research over local SearXNG. | required |

The LLM-using CLIs resolve their model via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`. The CLI exits with a clear error if neither is set — no hidden defaults.
