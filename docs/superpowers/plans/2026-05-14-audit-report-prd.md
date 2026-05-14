# PRD — Epistemic Audit Report v2

**Status:** draft
**Author:** Claude (with Timo)
**Date:** 2026-05-14
**Scope:** `src/andamentum/epistemic/audit_report.py` and related rendering paths

---

## 1. Goal

The audit report is not a UI nicety — it is the **load-bearing artefact** that demonstrates the central claim of the andamentum.epistemic system:

> Reasoning has been *externalised* from the LLM into a deterministic, inspectable, auditable trail. A reader does not have to trust the model's testimony; they can verify the chain themselves.

A "top-quality" report is one a reader can hold up next to Schneider (2025) and say: *this is what justified, externalised chatbot reasoning looks like, in concrete form.*

Every section of the report must earn its place against that claim.

---

## 2. Why this matters (positioning vs. the literature)

Schneider's "Chatbot Epistemology" (2025) lists the specific epistemic deficits that prevent LLM output from being justified knowledge:

| Schneider's deficit | What the report has to do about it |
| --- | --- |
| **Hallucinations** | Every claim must carry retrieved-evidence anchors with stable external IDs (DOI / PMID / NCT / URL). No claim may rest on unsupported LLM text. |
| **Black box** ("Experts cannot generally provide semantically intelligible 'beliefs' or 'reasoning steps' that led to the output") | The report **is** the semantically intelligible reasoning trace: gates passed/failed, IBE candidates with loveliness×likeliness, scrutiny verdict, adversarial probe, decomposition tree. |
| **Feedback sycophancy** | The adversarial probe must be visible at the same prominence as supporting evidence, with a count of contradicting items the system actively went looking for. |
| **Diachronic justification** (will the same input produce the same justification tomorrow?) | The report must be a frozen artefact — snapshot ID, timestamp, model ID, pipeline version, evidence IDs all stable and citable. |
| **Reliabilist demand** (was the process reliable?) | Process traceable: which providers were queried, with which intents, yielding which items; which gates the claim cleared. |
| **Internalist demand** (does the reader have access to the reasoning?) | Reader can re-trace every node of the decision: from decomposition → claim → evidence → judgement → integration → verdict. |

The current v13 report partially does this. The fixes below close the gaps.

Related framing from the rest of the folder:
- **Apple "Illusion of Thinking" (NeurIPS 2025)** — reasoning models collapse past a complexity threshold and produce confident-but-wrong chains. The report's gate-by-gate trail is the *anti-pattern*: it doesn't hide where the system was uncertain or where evidence ran out.
- **"Unstable Intelligence"** — LLM outputs vary unpredictably. The report shows the deterministic substrate (gates, IBE scoring, evidence retrieval) that wraps the LLM and constrains its variance.
- **Peirce** ("Fixation of Belief", "Illustrations of the Logic of Science") — the system implements Peircean inquiry (fixation through doubt, IBE, common-cause inference). The report must make this visible by *naming the moves*, not just rendering them.

---

## 3. Audience

Two readers, both important:

- **R1 — Domain expert reader (clinician, scientist, policy analyst).** Wants the answer in one glance, the strongest evidence at hand, and confidence they could defend it. Cares less about epistemology, more about "what should I do with this?".
- **R2 — Epistemologist / AI-safety reader (Schneider's audience).** Wants to see the reasoning machinery: was this system reliabilist-justifiable? Was the reader given internalist access? Are the deficits Schneider listed actually addressed?

Today's report serves R1 adequately above the fold and degrades for R2 below it (limitations dumped, gate trace partial). v2 must serve both, with R1 above the fold and R2's needs satisfiable by drilling down.

---

## 4. Non-goals

- Changing the classic `typeset_report.py` (separate artefact, preserved as-is).
- Modifying the `typeset` module itself or adding custom atoms — work within the 7 built-in atoms.
- Adding LLM-generated prose anywhere new — the value is *externalising* reasoning, not generating more of it.
- Designing for downstream consumption (Slack, mobile). Web HTML + docx parity.

---

## 5. Issues in v13 (concrete catalogue)

### Class A — Correctness bugs (these are wrong, not stylistic)

**A1. Claim-card badge contradicts the verdict.**
HCQ posterior 11.5%, verdict "No", 31 contradicting vs 9 supporting items — yet the card shows a green `supported` badge. Badge is sourced from `claim.stage` (a lifecycle state meaning "robust enough to integrate"), not the computed verdict. The badge label and the report verdict must come from one source.

**A2. Evidence counts disagree across sections.**
HCQ Summary-of-findings table shows 11 supporting, 31 contradicting, 58 no-bearing (=100). The card details show 9 supporting, 31 contradicting, 58 no-bearing (=98). One of these is wrong, and they cannot both be displayed.

**A3. Adversarial items mis-bucketed as supporting evidence.**
Statins "Strongest supporting evidence" items 1–2 begin with `"Adversarial (statistical):"` / `"Adversarial (generalization):"` — these are adversarial-probe outputs surfaced through the supporting list because of how `evidence.direction` is being set upstream. The judge label is in the text, so the misclassification is observable from inspection alone.

**A4. Summary section duplicates and contradicts itself.**
The Summary opens with `**Research Question:** … **Evidence Sources:** 33` then a `<blockquote>` re-states the same lines but with **Evidence Sources: 10** (the LLM-written summary's own count). Two evidence counts in three lines. The blockquote is the agent's narrative prefix being included as data.

**A5. `Limitations` section is mis-named.**
HCQ has 19 "Limitations" bullets, each of which is a re-rendering of an individual contradicting-evidence judgement string. These are not limitations of the conclusion; they are more contradicting evidence, presented as a separate section. The semantic label deceives the reader.

### Class B — Schneider-framework gaps (R2 reader cannot do the internalist check)

**B1. No named gates trace.**
The report says "Scrutiny: pass. Adversarial search: mixed results (balance: 0.56). Convergence: assessed. Deductive validation: passed. Computational verification: completed." in a one-line card detail. The reader cannot see:
- Which gates were applied for this question type (PRIMARY/SECONDARY/SKIP routing)?
- What threshold each gate required and what value was observed?
- Which gate failed first, if any (for refuted/insufficient claims)?

This is exactly the "semantically intelligible reasoning steps" Schneider says LLMs cannot provide. We provide them — but currently we whisper them.

**B2. No decomposition tree for research mode.**
The system supports research mode (claim decomposition into sub-investigations). The report renders sub-claims as flat cards but does not show:
- The decomposition tree (parent objective → sub-investigations → claims).
- Whether the decomposition was attempted and abandoned (open-research fallback).
- The "combined verdict" reasoning that fuses sub-investigation results.

R2 cannot audit a multi-claim research run without this.

**B3. No reproducibility footer.**
Currently: `2026-05-12 · openai:gpt-5.4-nano`. Missing:
- Pipeline version (andamentum git ref or version tag).
- Snapshot ID and Artefact ID (the persistence anchors).
- Per-claim cycle counts (how many Peirce cycles were used; was the cap hit?).
- Provider list with per-provider yield (already partially present in audit trail; should be summarised).
- Reproduction command — the literal CLI invocation that would re-run this.

This is the diachronic-justification answer. A reader six months from now must be able to point at the report and say "I can re-run that command and obtain a comparable artefact."

**B4. No interpretation guide for the posterior.**
"Posterior: 11.5%" without a legend is a UX trap. A reader who is not a Bayesian sees a low number and concludes "low confidence". The actual reading is "11.5% probability the claim is true → 88.5% confidence in No." The report must defuse this either by (a) labelling the field directionally ("Probability the claim is true: 11.5% → verdict: No"), or (b) giving a one-line legend with the threshold bands the system uses for `decisive`, `inconclusive`, etc.

**B5. No quality / strength flags on evidence.**
The HCQ "Strongest supporting evidence" list includes items the system *itself* (via Caveats) flags as observational, confounded, or single-arm. The Caveats are correct; the support-list rendering doesn't reflect them. A reader scanning the list sees five "supporting" papers and may not realise three of them are weak by the system's own assessment. The structured judgement on each evidence item already carries this signal (study design, sample size where extracted, the judge's caveats); the renderer must surface it.

**B6. Flat h2 hierarchy with no visual nesting.**
`Summary`, `Summary of findings`, `Detailed analysis`, then per-claim children `How this claim was investigated`, `Strongest supporting evidence`, `Adversarial probe`, then `Caveats`, `Limitations` — all at h2. R2 cannot tell what nests inside what; the document looks like 8–10 peer-level sections rather than a structured chain.

### Class C — Naming and language

**C1. "Detailed analysis" is generic.** Schneider's framing suggests `Reasoning trace` or `How the system reasoned` — names that announce the externalisation move.

**C2. "Strongest supporting evidence" / "Adversarial probe" pair.** Currently asymmetric — supports gets a glowing label, adversarial gets a clinical one. Either both clinical (`Supporting evidence`, `Contradicting evidence the system actively sought`) or both rhetorical. Asymmetry encodes a subtle confirmation bias.

**C3. "supports_refined" / "supports" / "robust"** appear as raw verdict tokens. These are internal labels; user-facing rendering should normalise to a small closed set (e.g. `Supported`, `Supported with refinement`, `Insufficient`, `Refuted`).

**C4. Provider names rendered as code pills** (`europepmc`, `pubmed`, `openalex`) — fine but unstyled. They should look like badges (small caps, neutral tone), and the system should display the *human-readable* provider name (`Europe PMC`, `PubMed`, `OpenAlex`) alongside or instead.

### Class D — Polish

**D1.** IBE candidates table truncates descriptions mid-clause with `…`. Convert to stacked cards (one per candidate, full text, with verdict + score badges).

**D2.** Supporting-evidence bullets often include the judge's internal reasoning prose. Show only the one-sentence judgement; put the verbose reasoning in the appendix.

**D3.** Card `Details` collapsible is now sparse (scope + one-line verification + counts). Either inline these or remove and put the data in the audit-trail section.

**D4.** Statins Q&A opens with bare "Yes." but the integrated verdict is `supports_refined` and the body adds caveats. Match the headline to the verdict label ("Yes, with refinement.").

**D5.** No table of contents / skip links. HCQ report's `Limitations` alone is huge; navigation matters.

**D6.** No print stylesheet validation — the audit trail and IBE candidates section in particular need page-break-inside hints.

---

## 6. Required changes (the v2 spec)

### 6.1 Document skeleton

```
H1  Claim / question (existing)
Meta line: date · model · pipeline version · snapshot ID (NEW)

H2  Answer in brief                              (was: implicit, now named)
    Q&A items panel (kept, expanded)
      What did we find?
      What was studied?
      What type of question?
      What's the confidence?     (NEW: directional phrasing + legend)
      How thorough?
      Reproduction               (NEW: snapshot ID + CLI hint)

H2  Summary                                      (kept, deduplicated)
    Narrative prose only. No agent-prefix lines. No blockquote echo.

H2  Evidence at a glance                         (renamed from "Summary of findings")
    Directional split table (kept).
    One-line per claim: claim → verdict badge (FIXED: from verdict, not stage)

H2  Reasoning trace                              (renamed from "Detailed analysis")
    Per claim:

      H3  Claim: <statement>
          Card with: verdict badge, scope, evidence counts (one source of truth)

      H3  How the system reasoned about this claim
          - Decomposition (if research mode)
          - Initial gather → providers queried, items yielded
          - Investigation rounds (if any) → intent + yield
          - Scrutiny gate → passed/failed, threshold, observed value
          - Convergence gate → independent sources counted
          - Adversarial probe → balance, items surfaced
          - IBE candidates (stacked cards: verdict, loveliness, likeliness, full text)
          - Integrated assessment → final verdict label
          - Gate trace summary table (NEW): one row per gate, status, threshold

      H3  Supporting evidence
          Each item: clickable ref → provider badge → one-line judgement
          → strength flag (study design, weight, weaknesses) (NEW)

      H3  Contradicting evidence
          Same shape as supporting. (Renamed from "Adversarial probe".)

H2  Caveats and limitations
    Reserved for *system-level* caveats: where evidence was thin, where the
    scope mismatch was unresolvable, where the model declined to commit.
    NOT a re-dump of per-evidence judgements. (FIX for A5)

H2  Appendix
    Full evidence trail with judgements (kept).
    Add: full IBE candidate rationales (un-truncated).
    Add: gate-trace JSON snippet (for tooling / replication).

Footer (NEW)
    Pipeline: andamentum vX.Y (git ref abc1234)
    Model: openai:gpt-5.4-nano
    Snapshot: <uuid>   Artefact: <uuid>
    Date: 2026-05-12T14:22Z
    Reproduce: `andamentum-epistemic verify "..." --model ...`
```

### 6.2 Specific behavioural requirements

**R1. One source of truth for verdict label.**
Compute a per-claim verdict from `claim.integrated_assessment` (or equivalent), map to a small closed set `{Supported, Supported with refinement, Insufficient, Refuted, Open}`, and use it for the badge, the Q&A panel headline, the Evidence-at-a-glance row, and the card. Remove the `claim.stage`-driven badge.

**R2. One source of truth for evidence counts.**
A single function returns `(supporting, contradicting, no_bearing, total)` keyed off `claim.evidence_links` (or the canonical source). All sections render from that. Add an invariant check in the renderer; raise loudly on mismatch (no silent failures, per project convention).

**R3. Evidence bucketing is checked against the judge label.**
If the judge text starts with `"Adversarial …"`, the item is contradicting / adversarial probe output, not supporting. Add an assertion in the bucketer and a test pinning this.

**R4. Summary section is the narrative only.**
Drop the `**Research Question:**` / `**Evidence Sources:**` prefix lines and the agent's `<blockquote>` echo. Those data points live in the meta line and the Q&A panel; they don't need to appear three times.

**R5. Reasoning trace must expose gates.**
Add a gate-trace table per claim:

  | Gate | Required | Observed | Status |
  |---|---|---|---|
  | Scrutiny (independent verifier) | pass | pass | ✓ |
  | Convergence (≥2 independent sources) | ≥2 | 3 | ✓ |
  | Adversarial balance | <0.7 | 0.56 | ✓ |
  | Deductive validation | n/a (routed: SKIP) | — | skipped |
  | Posterior decisive | ≥0.8 or ≤0.2 | 0.83 | ✓ decisive |

  Each gate's `STAGE_GATES` entry already encodes the requirement; the renderer needs to read it and the per-claim observed value, not invent its own labels.

**R6. IBE candidates as stacked cards, full text.**

  ```
  ┌─ Candidate A (runner-up) ────────────────────────────┐
  │ Verdict: insufficient   Loveliness: 0.35 · Likeliness: 0.78
  │
  │ The supporting evidence mainly comes from pooled
  │ analyses showing reductions in stroke/MI and composite
  │ outcomes, but several key effects for the claim's most
  │ central endpoint are not tightly demonstrated …
  └──────────────────────────────────────────────────────┘
  ```

  Selected candidate gets a left-bar accent or a `selected` badge. No truncation.

**R7. Evidence strength flags.**
Each evidence item carries either:
- a small `design` chip (`RCT`, `meta-analysis`, `observational`, `review`, `guideline`)
- a `weight` chip (`primary`, `secondary`, `peripheral`)
- a `weakness` chip if applicable (`single-arm`, `combination intervention`, `out-of-scope cohort`)

  Source: the existing per-evidence judge output already carries these signals in the text; the structured assessment layer should extract them on ingest. For v2, extract on render if not yet structured upstream.

**R8. Reproducibility footer.**
Render at the bottom of every document. Computed from objective/snapshot metadata. Stable across re-renders of the same artefact.

**R9. Posterior gets a directional, legended display.**

  ```
  How confident are we?
  → Probability the claim is true: 0.115  (verdict: No)
    Decisive thresholds: ≥0.80 supports · ≤0.20 refutes · else inconclusive.
  ```

**R10. Closed verdict vocabulary.**
Normalise raw tokens at the renderer boundary. Map `supports`, `supports_refined`, `contradicts`, `insufficient`, `open` to a one-word user-facing label. Keep the raw tokens accessible in the appendix / gate trace for R2 audit.

**R11. h3 nesting under per-claim sections.**
Per-claim sub-sections (`How the system reasoned …`, `Supporting evidence`, etc.) become h3 children of the claim's h2. The classic typeset CSS already supports h3; the styling reads naturally.

**R12. Caveats and Limitations merged with stricter semantics.**
`Caveats and limitations` is a single section reserved for: system-acknowledged scope gaps, evidence-set limits the system flagged independently of per-evidence judgements, and any explicit "we cannot answer X" admissions. Per-evidence contradicting bullets go in the Contradicting evidence section, full stop.

**R13. TOC / skip navigation.**
Anchor links already exist (`id="audit-…"`, `id="supports-…"`, etc.). Add a minimal top-of-document jump list using the existing `typeset-items variant-pairs` atom — no new CSS.

### 6.3 What stays the same

- The classic `typeset_report.py` path is untouched.
- The 7 typeset atoms and existing CSS — no new atoms, no module changes.
- The adaptive intro paragraph in `_render_audit_trail_for_claim` — the v13 fix is good.
- The clickable DOI/PMID/NCT/URL resolution.
- The neutral colour palette (no red/green tonal panels).

---

## 7. Out-of-scope but worth noting

- **Cross-claim synthesis view for research mode.** Today's report renders sub-claims linearly. A claim-graph visualisation (parent → children with verdicts and weights) would be powerful for multi-step research but is a separate feature.
- **Diff view between two snapshots.** "Did the verdict change when we re-ran on 2026-05-14?" is a strong Schneider-style diachronic check. Requires snapshot diffing infrastructure.
- **Embedding the gate-trace JSON as machine-readable.** A small `<script type="application/ld+json">` block on the page would make the report indexable by downstream evaluation harnesses. Worth doing later.

---

## 8. Acceptance criteria

A v2 report is "top-quality" when:

1. **R1 (correctness):** every count, badge, and verdict label in the document agrees with every other. An automated test pins this for both modes (verify, research).
2. **R1 (no mis-bucketing):** no adversarial item appears in the supporting list; no contradicting item appears as a caveat.
3. **R2 (Schneider answer):** a reader can, for any claim in the report, identify
   (a) which gates the claim cleared and which it failed,
   (b) which providers were queried with which intents,
   (c) which alternative explanations the system considered and why one was selected,
   (d) the snapshot ID and reproduction command,
   without leaving the document.
4. **R2 (no orphan jargon):** every term-of-art (IBE, scrutiny, convergence, posterior, decisive threshold) is either defined inline on first use or linked to a glossary section.
5. **Visual:** the heading hierarchy nests visibly (per-claim sub-sections are clearly children of the claim, not peers of `Summary`).
6. **Manuscript-ready:** the rendered HTML is the figure (or figures) that goes into the manuscript. A reader of the manuscript who clicks through to the report sees the externalised reasoning Schneider says LLMs cannot provide.

---

## 9. Implementation order (suggested)

Do the correctness bugs first (Class A) — they undermine everything else.

1. **One source of truth for verdict + counts** (A1, A2). Includes the closed verdict vocabulary (R10).
2. **Adversarial bucketing fix** (A3) — pin with a test.
3. **Summary deduplication** (A4) — drop prefix lines and the agent's blockquote.
4. **Limitations re-scoping** (A5, R12) — merge Caveats and Limitations under stricter semantics; move per-evidence bullets to Contradicting evidence.
5. **Gate trace table** (B1, R5) — the central Schneider answer; biggest single payoff.
6. **Reproducibility footer** (B3, R8) — small change, large credibility return.
7. **Posterior legend** (B4, R9).
8. **Evidence strength flags** (B5, R7).
9. **IBE stacked cards** (D1, R6).
10. **h3 nesting + TOC** (B6, R11, R13).
11. **Naming pass** (C1–C4).
12. **Polish** (D2–D6).

Steps 1–4 are correctness fixes that should land as a single change; 5–6 are the Schneider-aligned epistemic-transparency additions; 7–12 are quality-of-presentation.

Each step has its own test; the rendered output for the two reference reports (`hcq`, `statins`) is the human-checked artefact at every stage.

---

## 10. Open questions for Timo

1. Is `andamentum-epistemic` reaching the point where the report is the primary user-facing artefact, or is the CLI output / JSON the primary and the HTML a derivative? (Affects how much polish vs how much machine-readability we invest in.)
2. The manuscript framing — is this report a figure inside the paper, an appendix artefact, or a live demo URL? (Affects width, print stylesheet, navigation.)
3. Are gate definitions stable enough to render directly, or will they churn? If churning, the gate-trace table needs an upstream contract first.
4. Should the report distinguish between "research mode" and "verify mode" with different top-level scaffolding (e.g. research mode opens with the decomposition tree), or keep one unified scaffold?
