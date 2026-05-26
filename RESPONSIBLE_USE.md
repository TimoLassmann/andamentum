# Responsible use of andamentum

`andamentum` is research infrastructure for building agentic reasoning
pipelines in scientific work. Several sub-modules call LLMs, fetch
external content, or produce artifacts that look like submission-ready
scientific output. The tool does not enforce research-integrity rules;
the user does.

This document records the project's expectations of how the package
should and should not be used. The MIT license disclaims warranty —
see [`LICENSE`](./LICENSE). Reading this document and complying with
the guidelines below is your responsibility.

## Intended use

- Drafting and reviewing **your own** scientific writing.
- Rendering scientific figures from **your own** data.
- Building **personal** knowledge bases of materials you have rights to
  access.
- Searching and synthesising literature you have permission to read.
- Reproducible epistemic analysis where each step's provenance matters.

## Out-of-scope uses

The following uses are not supported. Some are technically possible
with the code — possibility is not endorsement.

- **Peer review of confidentially-shared manuscripts.** Whetstone is
  not a peer-review tool. Do not run it on documents you received as
  a journal reviewer, grant-panel member, examiner, editor, or under
  any other confidentiality agreement. Most publishers and funders
  currently prohibit sharing such documents with AI tools, including
  cloud LLMs. See [`src/andamentum/whetstone/RESPONSIBLE_USE.md`](./src/andamentum/whetstone/RESPONSIBLE_USE.md).
- **Clinical decision support, diagnostic, or therapeutic
  recommendation.** `andamentum-epistemic` is research-stage software;
  its verdicts reflect what a single LLM-driven pipeline concluded
  from the evidence it retrieved within its cycle budget. They are
  not statements of clinical truth.
- **Content used as primary evidence in regulatory or legal
  submissions.** Same reasoning.
- **Mass extraction of paywalled scholarly content** without a
  text-and-data-mining (TDM) licence. Harvest will not check
  publisher Terms of Service for you. The major academic publishers
  (Elsevier, Springer Nature, Wiley, IEEE, ACM and others) contract
  TDM separately and prohibit unlicensed automated access.
- **Multi-tenant content-redistribution services** built on top of
  `document_store` containing third-party copyrighted material.
- **Any workflow where un-disclosed AI assistance violates a
  contractual or legal obligation** — funder rules, journal policies,
  institutional codes, NDAs, MTAs, DUAs.

## AI-use disclosure

Manuscripts, grant proposals, theses, peer reviews, and similar
artifacts that incorporate output from andamentum's LLM-using
sub-modules (`epistemic`, `deep_research`, `whetstone`, `chunker`,
`vision_critique`, and `document_store`'s metadata extractor) must
disclose that use under the rules of your journal, funder, and
institution.

The policy landscape is evolving; **confirm the current version** of
each relevant policy before drafting your disclosure. Key sources to
check:

- **COPE** (Committee on Publication Ethics) — position statements on
  AI in peer review and AI authorship.
- **ICMJE** (International Committee of Medical Journal Editors)
  Recommendations — author responsibilities for AI-generated content.
  ICMJE-aligned journals are the largest cluster of medical journals.
- **WAME** (World Association of Medical Editors) — recommendations on
  chatbots and scholarly manuscripts.
- **NIH** Grants Policy Statement; NOT-OD-25-122 on AI in peer review.
- **NHMRC** "Use of generative AI" guidance (Australian funding).
- **ARC** AI use in peer review of ARC schemes.
- **UKRI** generative-AI guidance.
- **ERC / Horizon Europe** rules.
- **Australian Code for the Responsible Conduct of Research** (2018)
  and any AI-specific updates from your institution.
- **Your institution's research-integrity office.**
- **The current policy of the specific journal / publisher / funder**
  whose work you are reviewing or to whom you are submitting.

`andamentum` does not insert disclosure language for you. The
`scribe.validate()` pass surfaces `[ai-drafted]` / `[ai-edited]`
markers as warnings if you choose to use them in your draft, but
that is a convention, not enforcement.

## Source-access expectations

`harvest`, `deep_research`, and the URL-fetching path used by
`document_store` ingestion all retrieve HTTP(S) URLs.

- **`robots.txt`** is consulted before every external HTTP(S) fetch
  via `harvest` and `deep_research`. The host's `/robots.txt` is
  fetched on first encounter, cached in-process, and parsed by
  Python's stdlib `RobotFileParser`. A `Disallow` rule matching the
  request path raises `FetchError`; a missing or unreachable
  `robots.txt` is treated as permissive (RFC-aligned default).
- **Paywalled academic publishers** (Elsevier, Springer Nature,
  Wiley, IEEE, ACM, NEJM, JAMA, Cell Press, Nature, Science / AAAS)
  are gated by hostname suffix. Fetches to these hosts are refused
  unless the caller passes the host into the surface API's
  `tdm_allowed_hosts` argument or the CLI's `--tdm-host` flag —
  the caller's explicit attestation that they hold a TDM licence
  for that publisher. Bulk extraction without a TDM licence is
  contractually prohibited by these publishers.
- **User-Agent** identifies the tool by name and version, with a
  contact URL pointing at the project repository. Do not
  impersonate a browser.
- **Rate limiting and attribution** for source APIs (PubMed,
  arXiv, Europe PMC, ChEMBL, Open Targets, Monarch, ClinicalTrials.gov,
  OpenAlex) is the user's responsibility. See the per-provider notes
  in the README's "Acknowledgements" section.

## Data fabrication

`andamentum.figures` is a deterministic plotting wrapper around
matplotlib. It plots whatever numbers you give it. It does not
generate, modify, or "AI-enhance" data points.

- Auto-applied visual decisions (horizontal bar flip, log scale,
  sort order, x-tick rotation) are listed in
  `FigureResult.advisor_notes`. **Mirror these in your figure
  caption** so reviewers can see what was decided automatically.
- `andamentum.vision_critique` produces a small-model heuristic in
  bounded JSON. The `confidence` field is the model's self-report,
  not a calibrated probability. Do not cite vision_critique output
  as an authoritative figure-quality verdict in a peer review or
  correction.

## Cloud inference

Whenever you select a non-local model (`openai:*`, `anthropic:*`,
`bedrock:*`, `gemini:*`, etc.), the full document or query content
is sent to that provider. Consider whether the input contains
patient data, embargoed research data, NDA / MTA / DUA-covered
material, or anything else your institution's data-classification
policy restricts.

- **Local-only operation** is supported end-to-end via Ollama
  models (`ollama:*`). Use this for any input that should not leave
  your machine.
- Tiered cloud-call gates (interactive prompt for `andamentum-whetstone`;
  TTY-aware one-time stderr warning for `andamentum-research` and
  `andamentum-epistemic`; silent for the pipeline CLIs) are planned
  for a future release (see
  `docs/plans/2026-05-16-responsible-release.md`).

## Audit log (opt-in)

andamentum can record every cloud LLM call to a local file as a
paper trail for institutional questions. **The audit log is off by
default.** Enable it by setting:

```bash
export ANDAMENTUM_AUDIT_LOG=/path/you/choose/andamentum-audit.log
```

When enabled, every cloud call appends one line:

```
2026-05-16T14:23:01Z whetstone panel anthropic:claude-haiku-4-5 sha256:abc123 6234B
```

The file is created with `0o600` permissions on first write at the
exact path you specified. No XDG default, no hidden directory in
your home. Disable by unsetting the variable.

## Reporting misuse

The MIT license permits any use technically. Socially-responsible
projects benefit from hearing about misuse patterns. Please file
issues at the project repository (see [`CITATION.cff`](./CITATION.cff)
for the canonical URL once published).

## Module-specific responsible-use documents

- `src/andamentum/whetstone/RESPONSIBLE_USE.md` — whetstone-specific
  guidance: peer-review prohibition, data classification, novelty
  check warnings, `--apply-patches` provenance, suggested
  ICMJE-style disclosure wording.
