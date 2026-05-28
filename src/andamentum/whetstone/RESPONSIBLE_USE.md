# Responsible use of whetstone

Whetstone is for sharpening **your own drafts**. This document
explains who whetstone is for, what it is explicitly not for, and
the in-code protections that exist to make accidental misuse hard.

## Who whetstone is for

- Authors sharpening their own pre-submission manuscript drafts.
- Authors revising rebuttals or response-to-review documents (their
  own writing, reflecting on a received review).
- Supervisors and co-authors editing a draft with the author's
  knowledge.
- PIs auditing a lab's pre-submission output.
- Anyone running the `andamentum-whetstone proofread <source>`
  subcommand for deterministic style + readability checks on their
  own text (no LLM call — wraps `andamentum.proofread`).

## Who whetstone is NOT for

The following uses are out of scope. Some are technically possible
with the code — possibility is not endorsement.

- **Peer review of confidentially-shared manuscripts.** Most
  publishers currently prohibit sharing reviewer manuscripts with
  AI tools. Springer Nature, Elsevier, Cell Press, AAAS, PLOS, BMJ,
  Frontiers, eLife, Wiley, and most society journals have published
  policies covering both cloud LLMs and self-hosted ones. Check the
  policy of the specific journal whose review you accepted before
  running anything.
- **Grant-application peer review.** NHMRC, ARC, NIH, ERC and
  others have stated positions on AI use by grant reviewers.
  Check the policy of the specific funder.
- **Thesis examination.** Most universities have explicit
  examiner-confidentiality rules.
- **Editorial Office workflows** where AI processing of
  confidential editorial correspondence is prohibited.
- **Pre-rebuttal generation** — feeding a received review back
  through whetstone to draft author responses is a grey area; if
  you do it, disclose in the cover letter / response that the
  responses were AI-assisted.

## Confidentiality and data classification

When the configured `--model` is a cloud provider (`openai:*`,
`anthropic:*`, `bedrock:*`, `gemini:*`, etc.), **the full document
text is sent to that provider**. Even for your own draft, this may
be inappropriate if the draft contains:

- Patient-identifiable data, clinical narratives, or other
  human-subjects data subject to ethics-approval handling conditions.
- Unpublished genomic, transcriptomic, proteomic, or other research
  data covered by an embargo or sponsor agreement.
- Industry-partner data or anything covered by an NDA / MTA / DUA /
  CDA.
- Material subject to your institution's classified-data policy.

For these cases, **use a local Ollama model end-to-end**:

```bash
andamentum-whetstone draft.md \
    --model ollama:gemma3-12b \
    --out review.md
```

Whetstone supports local-only operation throughout the review
pipeline. The `proofread` subcommand involves no model calls at all.

## In-code protections — what whetstone does for you

These exist to make accidental misuse harder. None of them remove
your responsibility to follow your journal / funder / institution's
rules — they're a safety net, not enforcement.

### Confidentiality-marker tripwire

Before the review pipeline runs, whetstone scans the harvested text
for phrases that suggest the document is a peer-review submission
or editorial-office artifact: `"Manuscript ID:"`, `"MS#"`,
`"Submission ID:"`, `"Confidential — do not distribute"`,
`"Reviewer Instructions"`, `"Editorial Office"`, `"Decision Letter"`,
`"This manuscript is being considered"`, and similar. If any match,
whetstone refuses to proceed.

To override (for false positives in your own draft):

```bash
andamentum-whetstone draft.md --confirm-own-draft ...
```

### Panel-mode authorship affirmation

The `panel` subcommand produces output shaped exactly like a journal
peer-review report (3–5 fictional reviewer biosketches, scored
per-criterion, with an Accept/Minor/Major/Reject recommendation).
That format is the highest laundering risk in the tool, so panel
mode requires an explicit affirmation:

```bash
andamentum-whetstone panel draft.md \
    --model ... --i-am-the-author --out panel.md
```

Or pre-set `ANDAMENTUM_PANEL_OWN_AUTHOR=1` once per shell session.

### AI-author attribution lock

Track-changes in the produced `.docx` are attributed to
`"andamentum-whetstone (AI)"` by default. Overriding this attribution
requires `--allow-author-override` AND emits a stderr warning that
misrepresenting AI-generated edits as a human reviewer's may
constitute research misconduct. This applies to the `--apply-patches`
path too.

### Tiered watermarking

Every output artifact carries **invisible provenance metadata**:

- `.docx`: AI contributor and `andamentum:ai-generated` keyword in
  document properties (visible in Word's File → Info).
- HTML: `<meta name="andamentum:ai-generated" content="true">` and
  related tags in `<head>`.
- Markdown: HTML-comment provenance header at top of file.

For `.docx` outputs, a **second customXml provenance part** is
also embedded inside the .docx zip at
`customXml/andamentum-provenance.xml` (namespace
`urn:andamentum:provenance:v1`). It carries the same provenance
line — generator, version, model id, produced-at timestamp,
`ai-generated="true"` — and is anchored via a package-root
relationship so it survives a python-docx round-trip. The point
of this second layer is that the core-properties fields (author,
keywords, comments) are user-editable from Word's File → Info pane
and from one-click "clear metadata" workflows; the customXml part
requires manual zip surgery to remove. An editor or integrity team
can read the part out with the `verify-provenance` subcommand:

```bash
andamentum-whetstone verify-provenance suspicious.docx
andamentum-whetstone verify-provenance suspicious.docx --format json
```

Exit code: 0 if any provenance marker is found, 2 if readable
but no markers found, 1 if the file can't be read as a .docx zip.

For **review-report outputs** (the `.docx`/`.md`/`.html` summarising
the review) there is also a **visible "AI-generated review content"
banner** by default. For the `--apply-patches` path (which produces
your modified manuscript) the visible banner is OFF by default to
avoid polluting your submission — the invisible metadata and
customXml provenance part are still written. Override with
`--visible-watermark` / `--no-visible-watermark`.

### Novelty-check network surface

`--check-novelty` extracts your unpublished novelty claims and
issues search queries derived from them. The v3 novelty pipeline
deliberately does NOT persist a per-claim cache to disk — hashed
digests of unpublished claims should not sit on your filesystem
between runs. Repeated runs re-query deep_research.

**The search queries themselves leave your machine** through
deep_research → your local SearXNG → public search engines. Treat
`--check-novelty` as a network-leak surface; do not run it on
someone else's unpublished work.

## Suggested AI-disclosure wording

If whetstone-assisted suggestions made it into your submitted draft,
most journals now require disclosure. The exact wording depends on
the journal — confirm against the current version of their policy.
A common ICMJE-aligned form:

> *During the preparation of this work, the authors used
> [whetstone (andamentum vX.Y) configured with `<MODEL_ID>`] in
> order to [critically review an earlier draft / suggest editorial
> rewrites / check for inconsistencies]. The authors reviewed and
> edited the content as needed and take full responsibility for
> the final manuscript.*

For NHMRC / ARC submissions, follow the funder's current AI-use
disclosure form. For thesis writing, follow your institution's
research-integrity code.

If you only ran the `proofread` subcommand (no LLM, no model
calls), no AI-content disclosure is required.

## What whetstone does NOT and CANNOT verify for you

- The current AI policy of your target journal, funder, or institution.
- Whether the document you're reviewing is technically a
  "manuscript" or a "draft" under any specific policy.
- Your obligations under sponsored-research agreements, MTAs, DUAs,
  or NDAs.
- Your institution's data-classification policy for the content
  in the document.
- Whether your co-authors consented to AI-assisted editing of the
  shared draft.

These are your responsibility.

## Policy landscape pointers

The policies cited throughout this document evolve continually.
Confirm the current version of each before drafting your disclosure
or submitting work that incorporates whetstone-assisted material.

- **COPE** (Committee on Publication Ethics) — position statements
  on AI in peer review and AI authorship.
- **ICMJE** Recommendations — author responsibilities for
  AI-generated content (covers most medical journals).
- **WAME** — chatbots and scholarly manuscripts.
- **NIH** — NOT-OD-25-122 and the Grants Policy Statement on AI in
  peer review.
- **NHMRC** "Use of generative AI" guidance.
- **ARC** AI guidance for peer review of ARC schemes.
- **UKRI / Horizon Europe / ERC** equivalents.
- **Australian Code for the Responsible Conduct of Research** (2018).
- **Your institution's research-integrity office** — most have
  released specific AI guidance since 2024.
- **The current policy of the specific journal / publisher / funder**
  whose work you are reviewing or to whom you are submitting.

## Reporting concerns

The MIT license permits any use technically; socially-responsible
projects benefit from hearing about misuse patterns. Please file
issues at the project repository (see `CITATION.cff` for the
canonical URL once published).
