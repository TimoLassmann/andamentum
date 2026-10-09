# andamentum

[![Tests](https://github.com/timolassmann/andamentum/actions/workflows/test.yml/badge.svg)](https://github.com/timolassmann/andamentum/actions/workflows/test.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Checked with pyright](https://img.shields.io/badge/type_checked-pyright-blue.svg)](https://github.com/microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A toolkit of small, composable, locally-runnable research tools.

Andamentum is a single Python package of tightly-scoped, independent tools for
working with documents and sources — extracting them, organising and searching
them, rendering figures from data, and reviewing your own drafts. Each tool does
one thing and stands on its own; there is no orchestrator and no "main" entry
point. Some are backed by a language model; several use none at all. Reach for
one, or compose several.

Two commitments shape the whole package:

- **Responsible by construction.** Confidentiality tripwires, robots.txt and
  paywalled-publisher gating, SSRF-safe fetching, and AI-provenance watermarks —
  in-code refusals and stamps, not markdown disclaimers. See
  [`RESPONSIBLE_USE.md`](./RESPONSIBLE_USE.md).
- **Explicit control, used sparingly.** Every public function that calls a
  language model takes `model=` as a keyword-only argument — no shared config,
  no ambient defaults — and models are used surgically: figure rendering,
  readability scoring, and chunk segmentation are deterministic, not
  model-driven. Point any of it at local (Ollama) models to run fully offline
  when data can't leave your machine.

## What's in the package

**Authoring**
- `andamentum.compose` — a task in, a finished document out (email, memo,
  report, article, how-to, proposal, grant); no approval step, never sends
  anything; its `compose.bundle` subpackage holds a document as a plain folder
  of markdown sections with frontmatter and renders to .docx / .md / .html / .txt
- `andamentum.word` — work on an existing Word file: check it, answer
  reviewers' comments, bind and refresh numbers, insert verified citations,
  draft into it as tracked changes; the input is never edited
- `andamentum.figures` — deterministic publication-quality figure rendering
  with journal-matched sizing (10 chart types, 7 journal palettes)
- `andamentum.typeset` — HTML / PDF typesetting used by other modules

**Reviewing your own drafts**
- `andamentum.whetstone` — review of your own drafts. One command reviews,
  rewrites as tracked changes, checks the references (with a Crossref
  lookup) and lists what the draft claims as new; flags turn parts off
  (`--offline` makes no network call besides the model). Also `revise`
  (review → edit → re-review until a condition you state, or a fixed check,
  holds), panel mode, checklist mode (a fixed bank of quota-driven probes →
  deterministic coverage table), and an opt-in web novelty check
- `andamentum.references` — reference checking in any citation style
  (numbers in brackets or parentheses, superscript, author-year, footnotes,
  Pandoc keys): citations against the list, then each entry against
  Crossref. A model only for what the text leaves open
- `andamentum.proofread` — deterministic readability + style checking (no LLM)
- `andamentum.vision_critique` — bounded vision critique of rendered figures
  via local multimodal models

**Sourcing and indexing**
- `andamentum.harvest` — universal source → markdown extraction (PDF / HTML /
  DOCX / PPTX / Markdown / plain)
- `andamentum.chunker` — deterministic markdown chunking by headings,
  paragraphs and sentences, with source offsets and heading paths
- `andamentum.document_store` — one SQLite file of documents with a fixed
  API; hybrid search (BM25 + vectors, cross-encoder rerank) and an opt-in
  search agent
- `andamentum.deep_research` — evidence-first research loop over the web (a
  Chrome or SearxNG) and scholarly sources; a report that cites verified findings

**Building agentic systems**
- `andamentum.agentic_dialect` — the house style for pydantic-graph + pydantic-ai
  systems as data: 9 laws, role briefs, a conformance checklist, and portable
  static gates (`check_code`) you can run against any system built in the dialect
- `andamentum.forge` — natural-language brief → validated, agent-authored,
  sandbox-audited runnable package; itself built in the dialect and held to it

**Shared infrastructure**
- `andamentum.core` — model resolution, agent runners, fetch gating, and
  embedding clients

## What andamentum is not

It is a set of building blocks you call from your own code or the command line —
not a chatbot, not an agent framework, and not a hosted service. whetstone, in
particular, refuses to run on material another author has shared with you in
confidence.

## Installation

andamentum is installed from GitHub. (The name `andamentum` on PyPI belongs to
an unrelated project, so `pip install andamentum` does not give you this
package.)

```bash
# the command-line tools, each on your PATH
uv tool install "andamentum @ git+https://github.com/timolassmann/andamentum.git"

# or as a library in your own project
uv add "andamentum @ git+https://github.com/timolassmann/andamentum.git"
```

Three optional extras add heavier capabilities. Name them in brackets, for
example `"andamentum[pdf] @ git+https://..."`:

- `pdf`: WeasyPrint, for PDF output from `andamentum-typeset`
- `html-articles`: trafilatura, for better extraction of HTML articles
- `vlm`: local vision-language models for `andamentum-harvest --mode accurate`
  (large; Apple silicon gets the fastest path)

## Quickstart

```bash
andamentum list                                    # every tool and whether it needs a model
andamentum-harvest paper.pdf -o paper.md           # any document to markdown, no model
andamentum-whetstone review draft.docx --model anthropic:claude-haiku-4-5 \
    -o draft.reviewed.docx                         # review your own draft as tracked changes
```

The [user guide](docs/guide.md) starts from the job you want done and shows
the commands for each tool. Every command's `--help` ends with examples.

The same review from Python:

```python
import asyncio
from andamentum.whetstone import review_document

result = asyncio.run(
    review_document(
        "draft.md",
        model="anthropic:claude-haiku-4-5",
        confirm_own_draft=True,
    )
)
print(result.summary)
```

Every public function that calls an LLM takes `model=` as a keyword-only
argument. Every command that calls a model takes `--model` on the command
line; there is no environment variable and no default model.

## Command-line tools

Sixteen tools are installed with the package, each with its own command.
`andamentum list` prints them (`--json` for programs and agents) and
`andamentum <tool> ...` runs one. Every command follows the same rules
([docs/interface.md](docs/interface.md)): `-o/--out` for output, `--json`,
`-` for stdin, `--model` with no default, `--offline` for no network call
besides the model, and one set of exit codes (0 success, 1 negative result,
2 usage, 3 input unreadable, 4 run failed). The [user guide](docs/guide.md)
has more examples for each tool; `--help` lists every flag.

| Command | What it does | Example | Model? |
|---|---|---|---|
| `andamentum-harvest` | Convert a PDF, Word, PowerPoint, HTML page or text file to markdown | `andamentum-harvest paper.pdf -o paper.md` | only for `--mode accurate` |
| `andamentum-chunker` | Split markdown at headings, paragraphs and sentences | `andamentum-chunker notes.md --max-chars 2000 -o chunks.json` | none |
| `andamentum-docstore` | Keep documents in one SQLite file, index and search them | `andamentum-docstore ingest store.db paper.pdf notes.md` | `sync` and `search` need an embedding model; `search` also a reranker |
| `andamentum-research` | Research a question on the web and in scholarly sources; the report cites what it found (`--depth quick\|standard\|deep`) | `andamentum-research "How much carbon do salt marshes store?" --mode scholarly --model M` | required |
| `andamentum-epistemic` | Answer a question (`ask`) or check a claim (`verify`) by weighing evidence for and against it | `andamentum-epistemic verify "CLAIM" --model M --embedding-model E --browser` | required |
| `andamentum-references` | Check a draft's citations against its reference list, then each entry against Crossref; exit 1 when it finds problems | `andamentum-references draft.md` | only for unclear cases |
| `andamentum-whetstone` | Review your own draft as markdown, HTML or tracked changes in Word; `revise` repeats until a condition holds | `andamentum-whetstone review draft.docx --model M -o draft.reviewed.docx` | required, except `reviewers`, `apply-patches`, `verify-provenance` |
| `andamentum-proofread` | Readability and style check, the same result every time | `andamentum-proofread draft.md` | none |
| `andamentum-compose` | Write a new document (email, memo, report, article, how-to, proposal, grant) from a task into a folder | `andamentum-compose write "A memo on the SaltLog pilot" -o memo --model M` | required to write; none for the folder commands |
| `andamentum-word` | Work on a Word file you have: check it, answer reviewers, refresh numbers, insert citations; your file is never changed | `andamentum-word check draft.docx -o checked` | required for `respond`, `bind-numbers`, `draft` |
| `andamentum-llm-judge` | Score a text, or compare two, against criteria with one judge or a panel | `andamentum-llm-judge score "TEXT" --model M` | required |
| `andamentum-figures` | Draw a publication-quality chart from a CSV or JSON table | `andamentum-figures results.csv --kind bar --x group --y value -o figure.svg` | none |
| `andamentum-vision-critique` | Check a rendered figure for overlapping labels, legibility and layout | `andamentum-vision-critique figure.png --model ollama:qwen3.8` | required (vision) |
| `andamentum-typeset` | Render markdown to a styled HTML page or PDF (`article`, `cv`, `report`) | `andamentum-typeset notes.md -o notes.html` | none |
| `andamentum-agentic-dialect` | Show the house style for agentic systems and check code against it | `andamentum-agentic-dialect check src/mypipeline` | none |
| `andamentum-forge` | Build a small runnable function package from a plain-language brief, tested in a sandbox | `andamentum-forge build "BRIEF" --model M -o wordcount` | required |

`M` is a model id such as `anthropic:claude-haiku-4-5` or `ollama:gemma4:31b`.

## Documentation

- [User guide](docs/guide.md): what each tool does, with commands to copy.
- [Writing a document from a task](docs/compose.md) (`andamentum-compose`).
- [Working on a Word file](docs/word.md) (`andamentum-word`: check, respond,
  refresh, citations, MCP).
- [Reviewing your own draft](docs/whetstone.html) (`andamentum-whetstone`).
- [The interface rules](docs/interface.md) that every tool follows.
- [`docs/`](./docs/) for the design of each module and
  [`examples/`](./examples/) for runnable Python examples.

## Intended use and limits

Please read [`RESPONSIBLE_USE.md`](./RESPONSIBLE_USE.md) before
publishing or submitting anything produced with andamentum.

The short version:

- **`whetstone` is for your own drafts.** It is not a peer-review
  tool. Do not run it on manuscripts, grant proposals, or other
  documents shared with you in confidence. Most publishers and
  funders currently prohibit AI in peer review.
- **AI disclosure is your responsibility.** Manuscripts, grant
  proposals, theses, peer reviews, and similar artifacts that
  incorporate output from andamentum's LLM-using sub-modules must
  disclose that use per ICMJE / NIH / NHMRC / ARC / COPE / your
  institution's rules.
- **`harvest` and `deep_research` consult `robots.txt`** before every
  external fetch and refuse paywalled academic publishers (Elsevier,
  Springer Nature, Wiley, IEEE, ACM, NEJM, JAMA, Cell Press, Nature,
  Science) unless the caller passes `tdm_allowed_hosts` (API) or
  `--tdm-host` (CLI). Bulk extraction without a TDM licence is
  contractually prohibited by these publishers.
- **`figures` plots data, it does not generate it.** Deterministic
  matplotlib wrapper. Auto-applied visual decisions are reported
  in `FigureResult.advisor_notes` — mirror them in your captions.
- **Cloud inference sends your content to the provider.** Use local
  Ollama models for inputs subject to ethics, NDA, MTA, DUA, or
  institutional data-classification rules.

`andamentum` is MIT-licensed and ships without warranty. The full
guidance lives in [`RESPONSIBLE_USE.md`](./RESPONSIBLE_USE.md).

## Pre-release / experimental

The package includes an additional sub-module that ships installed but is not
yet publicly documented:

- **`andamentum.epistemic`** — still under active development. The API is
  unstable and the published documentation does not yet describe it; if you
  discover it via `pip show -f andamentum` or `andamentum-epistemic --help`,
  treat it as experimental. A dedicated release will accompany the
  documentation when the module stabilises.

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgements

andamentum builds on a substantial body of open-source software and
publicly-funded data infrastructure.

### Software

- **[SearXNG](https://github.com/searxng/searxng)** — privacy-preserving
  metasearch; an instance you run, one of the two web-search options of
  `deep_research`. AGPL-3.0.
- **[trafilatura](https://trafilatura.readthedocs.io/)** — HTML article
  extraction.
  Barbaresi, A. (2021). "Trafilatura: A Web Scraping Library and
  Command-Line Tool for Text Discovery and Extraction", *ACL-IJCNLP
  2021 System Demonstrations*.
- **[Docling](https://github.com/DS4SD/docling)** (IBM Research) — PDF,
  DOCX, PPTX extraction.
- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** and
  **[pydantic-graph](https://github.com/pydantic/pydantic-graph)** —
  agent and DAG infrastructure used across the package.
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)** (Alex Garcia)
  — vector search inside SQLite, used by `document_store`.
- **[Ollama](https://ollama.com/)** and
  **[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)**
  (Google) — local embedding and LLM inference.
- **[textstat](https://github.com/textstat/textstat)** — readability
  metrics used by `proofread`.
- **[scikit-learn](https://scikit-learn.org/)** — used internally.
  Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python",
  *JMLR* 12: 2825–2830.

### Algorithms

- **Reciprocal Rank Fusion** (RRF) — search fusion in
  `document_store.searchers`.
  Cormack, G.V., Clarke, C.L.A., Büttcher, S. (2009). "Reciprocal
  Rank Fusion outperforms Condorcet and individual rank learning
  methods", *SIGIR '09*.
- **HDBSCAN** — density-based clustering.
  Campello, R.J.G.B., Moulavi, D., Sander, J. (2013).
  "Density-Based Clustering Based on Hierarchical Density
  Estimates", *PAKDD '13*. McInnes, L., Healy, J., Astels, S.
  (2017). "hdbscan: Hierarchical density based clustering",
  *JOSS* 2(11): 205.
- **BM25** — keyword scoring via SQLite FTS5.
  Robertson, S., Zaragoza, H. (2009). "The Probabilistic Relevance
  Framework: BM25 and Beyond", *Foundations and Trends in
  Information Retrieval* 3(4): 333–389.

### Data sources

Users running the web-research pipeline are bound by each provider's
terms of use and rate limits. Where the provider asks to be credited
in publications, please respect that.

- **PubMed / E-utilities** — NCBI / National Library of Medicine.
  Please set an `NCBI_API_KEY` for the 10 req/s rate, otherwise the
  pipeline throttles to 3 req/s.
  https://www.ncbi.nlm.nih.gov/books/NBK25497/
- **arXiv API** — 1 request per 3 seconds per the arXiv API manual.
  https://info.arxiv.org/help/api/tou.html
- **Europe PMC REST API** — please credit Europe PMC where data is
  shown.
  https://europepmc.org/Help#whyepmc
- **bioRxiv** — Cold Spring Harbor Laboratory.
  https://api.biorxiv.org/
- **ClinicalTrials.gov API v2** — NIH / National Library of
  Medicine.
  https://clinicaltrials.gov/data-api/about-api/api-policy
- **ChEMBL** — EMBL-EBI.
  https://chembl.gitbook.io/chembl-interface-documentation/web-services
- **Monarch Initiative** —
  https://monarchinitiative.org/about/data-and-services
- **Open Targets Platform** — EMBL-EBI / GSK / Sanger / others.
  https://platform-docs.opentargets.org/citation
- **OpenAlex** —
  https://docs.openalex.org/how-to-use-the-api/api-overview
- **Cochrane review abstracts** are accessed only via PubMed; full
  Cochrane reviews are © Wiley and are not retrieved by this code.
