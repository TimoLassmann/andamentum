# andamentum

Composable agentic systems for scientific automation.

Andamentum is a single Python package of twelve tightly-scoped sub-modules
for building agentic reasoning pipelines:

- **`andamentum.epistemic`** — formal epistemology, claim analysis, multi-agent verification
- **`andamentum.deep_research`** — web research pipeline with iterative search, verification, and synthesis
- **`andamentum.document_store`** — SQLite + FTS5 + vector search storage with automatic chunking and LLM metadata extraction
- **`andamentum.whetstone`** — sharpen your own drafts with editing, specialist review, or multi-expert panel feedback (track changes, HTML, or markdown output)
- **`andamentum.scribe`** — block-based document authoring (paragraph, heading, figure, table) backed by SQLite; renders to .docx
- **`andamentum.figures`** — publication-quality scientific figure rendering — deterministic plotting of your data with journal-matched sizing (9 chart types)
- **`andamentum.chunker`** — structural-first semantic chunking of long markdown into self-contained units
- **`andamentum.harvest`** — universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain)
- **`andamentum.vision_critique`** — bounded vision critique of rendered figures via local multimodal models
- **`andamentum.proofread`** — deterministic readability + style checking (no LLM)
- **`andamentum.typeset`** — typesetting system used by other modules for HTML / PDF output
- **`andamentum.core`** — shared model-resolution, `AgentRunner`, and embedding infrastructure

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

Nine CLIs are installed with the package. Run `--help` on any of them for the
full flag reference.

| Command | What it does |
|---|---|
| `andamentum-epistemic` | Formal-epistemology pipeline. Two modes: `ask "<question>"` (research mode — system attempts decomposition, falls back to open research if the question doesn't decompose) or `verify "<claim>"` (single-claim verification, SciFact-style) |
| `andamentum-research` | Web-research pipeline (search → fetch → extract → verify → synthesise) |
| `andamentum-whetstone` | Multi-lens review of your own draft → markdown / HTML / .docx with track changes. `--apply-patches PATCHES.json` applies a pre-built JSON patch list to a .docx (no LLM) |
| `andamentum-scribe` | Block-based document authoring backed by SQLite; renders to .docx |
| `andamentum-figures` | Publication-quality scientific figure rendering (9 chart types, journal-matched sizing). Deterministic plotting — no generative AI. |
| `andamentum-chunker` | Verifiable semantic chunking of long markdown into 2k–10k char self-contained units |
| `andamentum-harvest` | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain, auto-detected) |
| `andamentum-vision-critique` | Vision-critique a rendered figure → bounded JSON (label overlap, legibility, suggested fixes). Multimodal model required |
| `andamentum-proofread` | Deterministic readability + style check (SMOG, Flesch–Kincaid, weasel words, passive voice, weak openers, adverb density). Accepts URLs, PDF, DOCX, HTML, PPTX, Markdown, plain text |

`andamentum-epistemic`, `andamentum-research`, `andamentum-whetstone`,
`andamentum-chunker`, and `andamentum-vision-critique` need an LLM. Set
`ANDAMENTUM_MAIN_LLM_MODEL` once to avoid passing `--model` on every invocation:

```bash
export ANDAMENTUM_MAIN_LLM_MODEL=anthropic:claude-haiku-4-5
```

`andamentum-scribe`, `andamentum-figures`, `andamentum-harvest`, and
`andamentum-proofread` have no LLM dependency.

## Documentation

See [`docs/`](./docs/) for module-level narrative documentation and
[`examples/`](./examples/) for runnable code demonstrating common workflows.

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
- **`harvest` does not enforce publisher Terms of Service.** You
  are responsible for respecting `robots.txt`, publisher ToS, and
  any applicable TDM (text-and-data-mining) licensing for the URLs
  you fetch. Bulk extraction from paywalled academic publishers
  without a TDM licence is contractually prohibited.
- **`figures` plots data, it does not generate it.** Deterministic
  matplotlib wrapper. Auto-applied visual decisions are reported
  in `FigureResult.advisor_notes` — mirror them in your captions.
- **`epistemic` is research-stage software.** Its verdicts reflect
  what a single LLM-driven pipeline concluded; they are not
  statements of clinical, regulatory, or legal truth.
- **Cloud inference sends your content to the provider.** Use local
  Ollama models for inputs subject to ethics, NDA, MTA, DUA, or
  institutional data-classification rules. Tiered cloud-call gates
  for the CLIs are planned for a future release.

`andamentum` is MIT-licensed and ships without warranty. The full
guidance lives in [`RESPONSIBLE_USE.md`](./RESPONSIBLE_USE.md).

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgements

andamentum stands on a substantial body of open-source software and
publicly-funded data infrastructure. The full bibliography (including
the philosophy-of-science literature that informs the epistemic
pipeline) will be published alongside the epistemic module when that
system stabilises.

### Software

- **[SearXNG](https://github.com/searxng/searxng)** — privacy-preserving
  metasearch; self-hosted in `deep_research`. AGPL-3.0.
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
- **[scikit-learn](https://scikit-learn.org/)** — HDBSCAN clustering
  in `epistemic.dedup`.
  Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python",
  *JMLR* 12: 2825–2830.

### Algorithms

- **Reciprocal Rank Fusion** (RRF) — multi-signal search fusion in
  `document_store.chunks_search`.
  Cormack, G.V., Clarke, C.L.A., Büttcher, S. (2009). "Reciprocal
  Rank Fusion outperforms Condorcet and individual rank learning
  methods", *SIGIR '09*.
- **Dirichlet-Hawkes Process** (DHP) — temporal clustering in
  `document_store.dhp`.
  Du, N., Farajtabar, M., Ahmed, A., Smola, A., Song, L. (2015).
  "Dirichlet-Hawkes Processes with Applications to Clustering
  Continuous-Time Document Streams", *KDD '15*.
- **HDBSCAN** — density-based clustering used for evidence
  deduplication.
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

`epistemic` evidence providers query the following public APIs.
Users running the pipeline are bound by each provider's terms of
use and rate limits. Where the provider asks to be credited in
publications, please respect that.

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
