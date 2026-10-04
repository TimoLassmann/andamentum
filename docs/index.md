# Andamentum documentation

Andamentum is a single Python package of tightly-scoped sub-modules covering the document-handling pipeline a researcher uses end-to-end. Each sub-module ships in the same distribution; dependencies are the flat union of what they need.

## Sub-modules

| Module | Role |
|---|---|
| **compose** | A task in, a finished document out (email, memo, report, article, how-to, proposal, grant): brief, evidence, outline, draft, independent checks, revision. No approval step; never sends anything. |
| **scribe** | A document as a plain folder bundle (markdown sections with frontmatter, figures, references), rendered to .docx / .md / .html / .txt. |
| **figures** | Publication-quality scientific figure rendering — deterministic plotting of your data with journal-matched sizing (10 chart types, 7 journal palettes and the andamentum report palette). |
| **typeset** | Standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules. |
| [**whetstone**](./whetstone.html) | Structured review of your own drafts → markdown / HTML / .docx with comments and tracked changes. The default run reviews, rewrites, checks the references and suggests a next step; flags turn parts off, and `--goal` loops until a condition you state holds. Three review shapes over one substrate: `review` (criterion cascade, default), `checklist` (probe bank + coverage table), `panel` (simulated experts); plus `references`, `proofread`, `apply-patches`, `verify-provenance`. |
| [**references**](./whetstone.html) | Reference checking in any citation style — numbered, superscript, author-year, footnotes, Pandoc keys: citations against the list, then each entry against Crossref. Evidence first; a model only for what the text leaves open. |
| **proofread** | Deterministic readability + style checking (SMOG, Flesch–Kincaid, weasel words, passive voice). No LLM. |
| **vision_critique** | Bounded vision critique of rendered figures via local multimodal models. |
| **harvest** | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain). |
| **chunker** | Structural-first semantic chunking of long markdown into 2k–10k char self-contained units. |
| [**llm_judge**](./llm_judge.html) | LLM-as-judge: score one output against criteria, or compare two. Derived confidence + doubt, and a panel agreement gate that flags shaky verdicts rather than blocking. |
| [**document_store**](./document_store/overview.md) | SQLite + FTS5 + sqlite-vec personal knowledge base with 4-signal RRF search and LLM metadata extraction. |
| [**deep_research**](./deep_research/overview.md) | Web research pipeline (search → fetch → extract → verify → synthesize) over a local SearXNG instance. |
| **core** | Shared model resolution, agent runners, fetch gate, audit log, and embedding clients. |

An additional experimental sub-module ships installed but is not yet publicly documented; see the *Pre-release / experimental* section of [`../README.md`](../README.md).

## Design principles

- **Responsible by construction.** Fetch gate (robots.txt + paywalled-publisher tripwire), whetstone confidentiality tripwire, AI-provenance watermarks, cloud-vs-local awareness — implemented as in-code refusals and stamps, not markdown disclaimers. (An opt-in cloud-call audit log and tiered cloud-call gates are planned — see [`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).)
- **Explicit model argument.** Every public function that calls an LLM takes `model=` as a keyword-only argument. No hidden defaults, no silent fallbacks, no shared config module.
- **No environment-variable configuration.** Configuration flows top-down from the CLI / caller through explicit keyword arguments. Env vars are ambient state that hides at the call site.
- **One distribution, no extras.** Dependencies are the flat union of what the sub-modules need. There are no optional installs.
- **Public API in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal.
- **Tests next to code.** `pytest.ini_options.testpaths = ["src/andamentum"]` — tests live next to what they test, not in a top-level `tests/` directory.

## Imports

```python
from andamentum.whetstone import review_document
from andamentum.scribe import Bundle
from andamentum.compose import write_document, Roles
from andamentum.figures import figure
from andamentum.proofread import analyze
from andamentum.harvest import extract
from andamentum.chunker import extract_units
from andamentum.document_store import ingest, search
from andamentum.deep_research import ResearchState
from andamentum.vision_critique import critique_figure
from andamentum.llm_judge import judge_score, judge_compare
```

## CLIs

Ten scripts installed by the package. Run `--help` on any binary for full flag reference.

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-compose` | A task in, a finished document out, in a bundle folder. | required |
| `andamentum-scribe` | Folder-bundle document authoring; renders to .docx / .md / .html / .txt. | none |
| `andamentum-figures` | Publication-quality scientific figure rendering (deterministic plotting). | none |
| `andamentum-whetstone` | Draft review → markdown / HTML / .docx with comments and tracked changes; the default run also checks the references. Seven subcommands: `review` (default), `panel`, `checklist` (probe bank + coverage table), `references`, `proofread`, `apply-patches`, `verify-provenance`. | required for `review`, `panel` + `checklist`; optional for `references` |
| `andamentum-proofread` | Deterministic readability + style check. | none |
| `andamentum-harvest` | Universal source → markdown extraction. | none |
| `andamentum-chunker` | Structural-first semantic chunking. | required |
| `andamentum-vision-critique` | Vision-critique of a rendered figure → bounded JSON. | required (multimodal) |
| `andamentum-research` | Web research over local SearXNG. | required |
| `andamentum-llm-judge` | LLM-as-judge → JSON. `score <output>` or `compare <a> <b>`. One `--model` = fast single judge; repeated = sequential panel with an agreement gate. | required |

The LLM-using CLIs resolve their model via `--model anthropic:claude-haiku-4-5` or `$ANDAMENTUM_MAIN_LLM_MODEL`. The CLI exits with a clear error if neither is set — no hidden defaults.
