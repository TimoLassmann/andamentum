# Andamentum documentation

Andamentum is a single Python package of tightly-scoped sub-modules covering the document-handling pipeline a researcher uses end-to-end. Each sub-module ships in the same distribution; dependencies are the flat union of what they need.

## Sub-modules

| Module | Role |
|---|---|
| **compose** | A task in, a finished document out (email, memo, report, article, how-to, proposal, grant): brief, evidence, outline, draft, independent checks, revision. No approval step; never sends anything. Its `compose.bundle` subpackage holds a document as a plain document folder (markdown sections with frontmatter, figures, references), rendered to .docx / .md / .html / .txt. |
| **word** | Work on an existing Word file: checks (citations, cross-references, statistics, numbers), answers to comments and reviewers, bound numbers that refresh from result files, verified citations, drafting into Word as tracked changes. The input file is never edited. |
| **figures** | Publication-quality scientific figure rendering — deterministic plotting of your data with journal-matched sizing (10 chart types, 7 journal palettes and the andamentum report palette). |
| **typeset** | Standalone typesetting system (7 visual atoms, 3 named styles, HTML + PDF output) used by other modules. |
| [**whetstone**](./whetstone.html) | Structured review of your own drafts → markdown / HTML / .docx with comments and tracked changes. The default run reviews, rewrites, checks the references and suggests a next step; flags turn parts off, and `revise` reviews, edits and reviews again until a condition you state (`--until`) or a fixed check (`--until-clean`) holds. Three review shapes over one substrate: `review` (criterion cascade, default), `checklist` (probe bank + coverage table), `panel` (simulated experts); plus `revise`, `reviewers`, `verify-provenance`. The reference check alone is `andamentum-references`; the style check alone is `andamentum-proofread`. |
| [**references**](./whetstone.html) | Reference checking in any citation style — numbered, superscript, author-year, footnotes, Pandoc keys: citations against the list, then each entry against Crossref. Evidence first; a model only for what the text leaves open. |
| **proofread** | Deterministic readability + style checking (SMOG, Flesch–Kincaid, weasel words, passive voice). No LLM. |
| **vision_critique** | Bounded vision critique of rendered figures via local multimodal models. |
| **harvest** | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain). |
| **chunker** | Deterministic markdown chunking by headings, paragraphs and sentences, with source offsets and heading paths. No model. |
| [**llm_judge**](./llm_judge.html) | LLM-as-judge: score one text against criteria, or compare two. Derived confidence + doubt, and a panel agreement gate that flags shaky verdicts rather than blocking. |
| [**document_store**](./document_store/overview.md) | One SQLite file of documents with a fixed API: hybrid search (BM25 + vectors, cross-encoder rerank), an opt-in search agent, metadata enrichers. |
| [**deep_research**](./deep_research/overview.md) | Evidence-first research loop over the web and scholarly sources: a report that cites findings, each a verbatim quote that code found in the page. |
| **core** | Shared model resolution, agent runners, fetch gate, audit log, and embedding clients. |

An additional experimental sub-module ships installed but is not yet publicly documented; see the *Pre-release / experimental* section of [`../README.md`](../README.md).

## Design principles

- **Responsible by construction.** Fetch gate (robots.txt + paywalled-publisher tripwire), whetstone confidentiality tripwire, AI-provenance watermarks, cloud-vs-local awareness — implemented as in-code refusals and stamps, not markdown disclaimers. (An opt-in cloud-call audit log and tiered cloud-call gates are planned — see [`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).)
- **Explicit model argument.** Every public function that calls an LLM takes `model=` as a keyword-only argument. No hidden defaults, no silent fallbacks, no shared config module.
- **No environment-variable configuration.** Configuration flows top-down from the CLI / caller through explicit keyword arguments. Env vars are ambient state that hides at the call site.
- **One distribution.** Dependencies are the flat union of what the sub-modules need. Three optional extras add heavier capabilities: `pdf` (PDF output from typeset), `html-articles` (better HTML article extraction) and `vlm` (harvest's accurate mode).
- **Public API in `__init__.py`.** Each sub-module's `__init__.py` defines `__all__` explicitly; everything not listed is internal.
- **Tests next to code.** `pytest.ini_options.testpaths = ["src/andamentum"]` — tests live next to what they test, not in a top-level `tests/` directory.

## Imports

```python
from andamentum.whetstone import review_document
from andamentum.compose.bundle import Bundle
from andamentum.compose import write_document, Roles
from andamentum.figures import figure
from andamentum.proofread import analyze
from andamentum.harvest import extract
from andamentum.chunker import chunk
from andamentum.document_store import Store, HybridIndex, HybridRerankSearcher
from andamentum.deep_research import research
from andamentum.vision_critique import critique_figure
from andamentum.llm_judge import judge_score, judge_compare
```

## CLIs

Sixteen tools are installed with the package, each with its own command. `andamentum list` prints this list (`andamentum list --json` for programs and agents), `andamentum <tool> ...` runs a tool, and every command follows the rules in [interface.md](./interface.md): `-o/--out`, `--json`, `-` for stdin, `--model` with no default, and the same exit codes. Run `--help` on any command for its options and examples; the [user guide](./guide.md) shows each tool with commands to copy.

| Command | What it does | Model? |
|---|---|---|
| `andamentum-harvest` | Convert a PDF, Word, PowerPoint, HTML page, URL or text file to markdown. | for some uses |
| `andamentum-chunker` | Split markdown into chunks at headings, paragraphs and sentences. | none |
| `andamentum-docstore` | Store documents in one SQLite file, index them and search them. | for some uses |
| `andamentum-research` | Research a question on the web and in scholarly sources; the report cites what it found. | required |
| `andamentum-epistemic` | Answer a question or check a claim by weighing evidence for and against it. | required |
| `andamentum-references` | Check the citations and reference list of a draft, and look the references up. | for some uses |
| `andamentum-whetstone` | Review your own draft and write the review as markdown, HTML or tracked changes in Word. | required |
| `andamentum-proofread` | Check readability and style of a text; no model. | none |
| `andamentum-compose` | Write a document (email, memo, report, article, proposal, grant) from a task into a folder. | required |
| `andamentum-word` | Work on an existing Word file: check it, answer reviewer comments, refresh numbers, insert citations. | for some uses |
| `andamentum-llm-judge` | Score a text, or compare two texts, against criteria with one model or a panel. | required |
| `andamentum-figures` | Draw a publication-quality chart from a CSV or JSON table. | none |
| `andamentum-vision-critique` | Check a rendered figure for overlapping labels, legibility and layout with a vision model. | required |
| `andamentum-typeset` | Render markdown to a styled HTML page or PDF. | none |
| `andamentum-agentic-dialect` | Show the house style for agentic systems and check code against it. | none |
| `andamentum-forge` | Build a small runnable function package from a plain-language brief, tested in a sandbox. | required |

Every LLM-using CLI takes `--model anthropic:claude-haiku-4-5` on the command line. There is no environment variable and no default model.
