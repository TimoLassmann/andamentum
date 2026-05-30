# andamentum

Composable research tooling for scientific work.

Andamentum is a single Python package of tightly-scoped sub-modules. Each module
does one thing well; together they cover the document-handling pipeline a
researcher uses end-to-end — from extracting a PDF to drafting a paper to
critiquing the result.

The package is built with two design commitments that distinguish it from
generic AI tooling:

- **Responsible by construction.** Confidentiality tripwires, robots.txt and
  paywalled-publisher gating, AI-provenance watermarks, and explicit local-vs-
  cloud awareness — implemented as in-code refusals and stamps, not markdown
  disclaimers. See [`RESPONSIBLE_USE.md`](./RESPONSIBLE_USE.md).
- **No hidden defaults.** Every public function that calls an LLM takes
  `model=` as a keyword-only argument. There is no shared config module, no
  ambient defaults, no env-var-driven behaviour selection.

## What's in the package

**Authoring**
- `andamentum.scribe` — block-based document drafting (paragraph, heading,
  figure, table) backed by SQLite; renders to .docx
- `andamentum.figures` — deterministic publication-quality figure rendering
  with journal-matched sizing (9 chart types, 7 journal palettes)
- `andamentum.typeset` — HTML / PDF typesetting used by other modules

**Reviewing your own drafts**
- `andamentum.whetstone` — criterion-cascade review of your own drafts
  with track changes, panel mode, and optional novelty check
- `andamentum.proofread` — deterministic readability + style checking (no LLM)
- `andamentum.vision_critique` — bounded vision critique of rendered figures
  via local multimodal models

**Sourcing and indexing**
- `andamentum.harvest` — universal source → markdown extraction (PDF / HTML /
  DOCX / PPTX / Markdown / plain)
- `andamentum.chunker` — structural-first semantic chunking of long markdown
  into 2k–10k char self-contained units
- `andamentum.document_store` — SQLite + FTS5 + sqlite-vec personal knowledge
  base with 4-signal RRF search and LLM metadata extraction
- `andamentum.deep_research` — web research pipeline (search → fetch → extract
  → verify → synthesise) over a local SearXNG instance

**Shared infrastructure**
- `andamentum.core` — model resolution, agent runners, fetch gating, and
  embedding clients

## Installation

```bash
pip install andamentum
```

The core install works out of the box. Two optional extras add heavier,
self-contained capabilities:

```bash
pip install 'andamentum[html-articles]'   # trafilatura — best-in-class HTML article extraction
pip install 'andamentum[pdf]'             # WeasyPrint — PDF output for andamentum-typeset
```

## Quickstart

Review a draft you wrote yourself:

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
argument. Set `ANDAMENTUM_MAIN_LLM_MODEL` once to avoid passing `--model` on
every CLI invocation:

```bash
export ANDAMENTUM_MAIN_LLM_MODEL=anthropic:claude-haiku-4-5
```

## Command-line tools

Ten CLIs are installed with the package. Run `--help` on any of them for
the full flag reference.

| Command | What it does | LLM? |
|---|---|---|
| `andamentum-scribe` | Block-based document authoring backed by SQLite; renders to .docx | none |
| `andamentum-figures` | Publication-quality scientific figure rendering (9 chart types, journal-matched sizing). Deterministic plotting — no generative AI. | none |
| `andamentum-whetstone` | Criterion-cascade review of your own draft → markdown / HTML / .docx with track changes. Five subcommands: `review` (default), `panel` (multi-expert), `proofread` (no LLM), `apply-patches` (no LLM), `verify-provenance` (no LLM) | required for `review` + `panel` |
| `andamentum-proofread` | Deterministic readability + style check (SMOG, Flesch–Kincaid, weasel words, passive voice, weak openers, adverb density). Accepts URLs, PDF, DOCX, HTML, PPTX, Markdown, plain text. | none |
| `andamentum-harvest` | Universal source → markdown extraction (PDF / HTML / DOCX / PPTX / Markdown / plain, auto-detected) | none |
| `andamentum-typeset` | Render Markdown to typeset HTML / PDF in three styles (article / cv / report). PDF needs the `pdf` extra. | none |
| `andamentum-chunker` | Verifiable semantic chunking of long markdown into 2k–10k char self-contained units | required |
| `andamentum-vision-critique` | Vision-critique a rendered figure → bounded JSON (label overlap, legibility, suggested fixes). Multimodal model required | required (multimodal) |
| `andamentum-research` | Web-research pipeline (search → fetch → extract → verify → synthesise) over a local SearXNG instance | required |
| `andamentum-epistemic` | Formal-epistemology pipeline: `ask "<question>"` (decompose + research) or `verify "<claim>"` (single-claim verification) | required |

## Documentation

See [`docs/`](./docs/) for module-level documentation and
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
- **[scikit-learn](https://scikit-learn.org/)** — used internally.
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
