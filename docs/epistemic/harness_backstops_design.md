# Harness backstops — moving philosophical weight off the LLMs

Status: **design approved in principle, revised after a critical review pass
(2026-07-07); not yet implemented.** Follow-up to
`docs/confidence_analysis.md` and the Tier 0/1 work (`docs/tier0_design.md`,
`docs/tier1_design.md`). Scope: `andamentum.epistemic` only.

## The diagnosis (from the 2026-07-07 audit)

The system's healthy pattern is *LLM proposes, deterministic code disposes*
(stage gates, adversarial caps, K-agreement, cycle caps). Three structural
findings identify where that pattern breaks:

1. The IBE selection and its loveliness/likeliness scorers — not the two
   "sanctioned" judges in `judge.py` — are the dominant confidence drivers,
   and they sit downstream of and bypass the counting machinery.
2. Every existing deterministic guard is a one-directional attenuator: it can
   shrink confidence or force `insufficient`, but nothing can catch a *wrong
   direction* the LLM committed to.
3. The validated entropy/one-hot signal (`judgment_signal.py`, Tier 0) is
   computed and stored on every Evidence but consumed nowhere.

Design discipline carried over from `confidence_analysis.md`: consume existing
signals before building new machinery; deterministic checks may *suspend*
judgment, never flip direction; any change at a convergence/aggregation site is
benchmark-gated, not merely unit-tested.

## When determinism is trustworthy (added after the 2026-07-07 critical review)

"Deterministic" is not a synonym for "correct" — a deterministic rule over
messy inputs is just a heuristic with a reproducible error rate. Determinism
earns trust in exactly three shapes, and every item below is sorted against
them:

1. **Pure math over already-structured signals** — entropy of a stored
   distribution, confidence-weighted mass, score gaps. No identity resolution
   involved.
2. **Exact identifiers** — DOI, PMID, registry keys, provider class
   attributes. Equality is meaningful.
3. **Conservative suspend-only floors** — rules that can only withhold or
   demand more work, never assert a direction or certify independence.

Determinism is NOT trustworthy for identity or semantic resolution — author
names across providers ("Timo Lassmann" vs "Lassmann T"), venue names, "does
this text bear on this claim". There, deterministic string rules have error in
both directions and lack the LLM's context; the right pattern is exact-match
floors beneath a bounded LLM judgment, not string heuristics dressed up as
ground truth.

## Item 1 — Direction-concordance tripwire (coherentism / consilience)

**Purely deterministic — zero new LLM calls.** All inputs already exist at
synthesis time: the IBE verdict + confidence (Claim), the diagnostic counting
posterior (`compute_posterior` computes it on every run from the stored
judgment distributions), and the adversarial balance band (Claim).

Rule shape: if the IBE direction disagrees with the confidence-weighted
evidence mass **and** adversarial balance sits in the opposing band, withhold
the directional verdict — force `insufficient` or emit a `Demand` for
re-scrutiny. This is not the removed counting blend: no numbers are mixed; it
is a disagreement detector with the same shape as the existing
no-certified-verdict gate (suspend, never assert the opposite direction).

**Honest caveats (from the 2026-07-07 critical review):**

- *The two legs are not equally independent.* The counting mass aggregates the
  same judge's verdicts that populated the IBE evidence brief — a systematic
  judge error fools both. The adversarial balance is the genuinely independent
  leg (different evidence pool, different agent). The conjunction is therefore
  load-bearing, not optional: counting disagreement alone must never fire the
  tripwire.
- *Partial redundancy with the existing cap.* `_adversarial_confidence_cap`
  already hard-caps confidence at 0.5 when balance is REFUTED — which still
  permits a directional contribution up to 0.5 + 0.5/2 = 0.75. The tripwire's
  marginal effect is exactly that residual 0.5–0.75 band (plus the CONTESTED
  band). Real, but narrower than "closes the gap" suggests.
- *When the signals disagree, IBE is by design the smarter one* — that is why
  the blend was removed. The tripwire is defensible only as suspend-only, and
  its false-positive rate (verdicts suspended that the benchmark says were
  right) is the primary benchmark metric, not an afterthought.
- Must operate **per claim** (balance and evidence mass are per-claim), not at
  the objective level.

Sites: `confidence.py` (or a small pure module it calls), threshold δ named in
`thresholds.py` with its philosophical grounding.

## Item 2 — Consume the stored entropy signal (Peirce, demand-gated) ✅ approved

Two deterministic consumers for `judgment_entropy` / `judgment_one_hot`:

- **(a) Paraphrase-flip tripwire (the Tier 2 design, made selective).** Fire a
  single re-judgment under a paraphrase only when a judgment is one-hot *and*
  load-bearing (pivotal to a near-threshold gate, or feeding a posterior near
  `POSTERIOR_DECISIVE_THRESHOLD`). Emitted as a `Demand`, bounded, never a
  blanket K× re-ask.
- **(b) Confidence-thresholded gate counting (Tier 1.5, benchmark-gated).**
  Promotion should not ride on high-entropy judgments — but fractional
  weighted counts would conflate two distinct things: *how sure each judgment
  is* with *how many independent evidence lines exist* (Reichenbach's ≥2
  requirement is about distinctness, and "3 sources at 0.67 confidence = 2.0"
  is not two independent sources). So the gate keeps discrete semantics: a
  source counts toward `count_supporting_sources` **iff** its judgment
  confidence clears a named threshold. One-hot judgments (the small-model
  degeneracy mode) always clear it ⇒ backward-compatible limiting case
  preserved.

## Item 3 — Compute the scrutiny verdict instead of asking for it ✅ approved

`epistemic_assess_evidence`'s strong/moderate/weak/conflicting categorical is
the sole determinant of the scrutiny verdict, yet every input it needs already
exists as data: per-evidence judgment distributions, quality scores, mass in
both directions. Deterministic mapping:

- `conflicting` — non-trivial confidence-weighted mass on both directions;
- `weak` — low quality-weighted supporting mass;
- `strong` / `moderate` — thresholds over quality-weighted mass (constants in
  `thresholds.py`).

This *removes* an ungrounded LLM categorical entirely. Verified feasible in
time: judgments are applied at claim creation (`operations/claims.py`), before
`Scrutinize` runs, and the scrutiny operation already ends in a deterministic
combiner — only its weight *input* is LLM today.

**Honest trade-off:** the assess agent reads the evidence *content* and can
mark methodologically weak evidence (small n, wrong population) that neither
`quality_score` (bibliometric, or absent for identifier-less sources) nor the
judgment distribution captures. Mitigation: the content-reading channel stays —
`epistemic_identify_single_issue` still runs per item, its blocking
uncertainties already downgrade the verdict deterministically, and that is
where methodological objections belong (as named issues, not as an opaque
weight). Also a cost: the mapping introduces 2–3 new thresholds ("non-trivial
opposing mass", "low quality-weighted mass") — exactly the constants-we-must-
justify problem, so they are benchmark-validated at introduction, not tuned by
feel. Site: `operations/scrutiny.py`.

## Item 4 — Independence floor (Reichenbach) ✅ approved, revised after critique

The deterministic domain classifier is stubbed (`domain_classifier.py` returns
a fixed low-confidence default), so independence currently rests on LLM
classification checked against itself. But the original "shared DOI / shared
authors / same venue ⇒ never independent" framing was too rosy — **identity
resolution is not deterministic**, and the floor must be tiered by how
trustworthy each signal actually is:

**Tier A — genuinely deterministic (build this):**
- shared normalized DOI / PMID (`identifiers`) ⇒ never independent. Note the
  literal same-paper-via-two-providers case is *already* handled: 
  `dedupe_evidence.py` merges by normalized `source_ref` and records
  `corroboration_count`. The floor extends this to the convergence layer.
- same provider `independence_group` ⇒ never independent. **Caveat found
  during review: `independence_group` is currently declared on the provider
  classes but consumed nowhere** — this item is "wire it in", not "already
  used". (The overview docs briefly claimed otherwise; corrected 2026-07-07.)

**Tier B — heuristic, NOT deterministic (treat accordingly):**
- Author matching. "Timo Lassmann" (arXiv) vs "Lassmann T" (PubMed) vs
  "Lassmann, T." — per-provider formats differ, plus transliteration, unicode,
  middle initials, and common surnames (many distinct "Y. Wang"s). Both error
  directions are harmful: a false merge suppresses convergence (blocks ROBUST
  systematically in fields dominated by common surnames); a false split defeats
  the floor. Rules: (i) use it only in the conservative direction — author
  overlap may *lower* independence, never certify it; (ii) high-precision
  matching only (unicode-normalized full-name equality, or ORCID when present —
  rare); surname + first-initial matching is too coarse to act on
  deterministically; (iii) anything softer stays with the LLM independence
  judge, now operating *on top of* the Tier A floor rather than alone.
- Venue matching has the same disease ("PNAS" vs "Proc Natl Acad Sci U S A")
  — skip unless a normalized identifier (ISSN) is available.

The honest summary: the same-paper case is already deterministic, the
same-corpus case becomes deterministic by wiring in `independence_group`, and
the same-research-group-different-paper case — the one author matching targets
— is precisely where deterministic rules are weakest. Sites:
`convergence_detector.py`, `domain_distance.py`, `dedupe_evidence.py`.

**Implementation note (2026-07-07): `independence_group` dropped as too coarse.**
Inspecting the actual values showed `independence_group` is *provider-corpus*
granularity — pubmed and europepmc both carry `biomedical_literature`, arxiv and
biorxiv both `preprint_archive`. Hard-declaring non-independence from a shared
group would collapse two *distinct* biomedical papers into one source and
systematically block ROBUST in literature-dense fields — the exact "false merge"
failure the critique warned about, one level up. So the shipped floor uses **only
exact DOI/PMID/arXiv identity** (`count_independent_sources`): two items provably
the same paper collapse to one; everything else counts separately. This is
unimpeachable and can only downgrade illusory convergence (same paper via ≥2
providers landing in ≥2 domain clusters). `independence_group` and author
matching are deliberately NOT used — the former is too coarse, the latter is the
Tier-B heuristic the critique said to leave to the LLM judge on top of the floor.

## Item 5 — Split Lipton's dimensions, with histogram-entropy elicitation ✅ approved

Lipton distinguishes **likeliness** (probabilistic fit to evidence —
computable) from **loveliness** (explanatory virtue — irreducibly judgment).
Current state: both are bare scalar floats from free LLM judgment
(`LovelinessScore.loveliness`, `LikelinessScore.likeliness`), and the code
concedes the selection argmax is noise-sensitive.

The validated construct from `experiments/dirichlet_confidence` (FINDINGS.md
finding 2) is **Shannon entropy of a single verbalized histogram over a closed
set** — one-call entropy matched or beat every multi-call method; the
multi-call Dirichlet band was refuted. Continuous scalars have no histogram, so
the elicitations are reshaped to closed sets:

- **Likeliness — anchored, not fully computed** (revised after critique). The
  original "computed, not elicited" overstated the case: candidates are
  *explanations*, not directions — several candidates share a direction while
  positing different mechanisms, and a direction-mass formula would score them
  identically, destroying within-direction discrimination and changing the
  framing-tie arithmetic. And "P(evidence | this mechanism)" genuinely requires
  reading content. Defensible version: compute the **direction-mass anchor**
  deterministically (confidence-weighted judged-evidence mass consistent with
  the candidate's direction) and bound the LLM's likeliness within a named band
  of it — the LLM discriminates mechanisms within the band; gross fit errors
  are structurally impossible.
- **Loveliness — ordinal histogram.** Per candidate, elicit belief over a small
  ordinal scale (weak / moderate / strong explanatory virtue, sum-to-100) using
  the exact Tier-0 recipe (reasoning-first field, per-field descriptions,
  calibration nudge). Score = expected value over bins; entropy = per-score
  reliability. `judgment_signal.py` math applies unchanged (class-count
  agnostic). **Validation caveat:** the entropy construct was validated on the
  3-way evidence-judgment task only; generalisation to virtue-rating is
  plausible (still closed-set) but unproven — run a Tier −1-style mini
  calibration on the canonical local models first (the
  `experiments/dirichlet_confidence` harness makes this cheap).
- **K-agreement gating — no new elicitation needed** (simplified after
  critique). The framing-tie gap is *already computed deterministically* from
  the candidate scores (`_framing_tie_cap` returns `ft_gap`), so gating the
  K-agreement re-run does not require eliciting a selection distribution: skip
  the re-run only when run 1's gap is decisively large **and** the claim is not
  near a load-bearing boundary (gate threshold or
  `POSTERIOR_DECISIVE_THRESHOLD`). The second condition matters because
  run-1's gap is itself score-noise — gating on it alone would partially
  undermine the reason K-agreement exists. Today K=2 unconditionally doubles
  IBE cost; gated, the re-run fires only where it can change an outcome.
  (Eliciting belief mass over candidates remains an option, but 4–6-way
  sum-to-100 distributions are harder for small models than the validated
  3-way, so it is not the first move.)

Sites: `agents/output_models.py`, `agents/integration.py`,
`operations/integration.py`, `judgment_signal.py` (generalise class labels if
needed), `thresholds.py`.

## Smaller approved-direction items (do after 1–5)

- Embedding-similarity **audit flag** under `epistemic_screen_relevance`
  (downgraded from "floor" after critique: embeddings measure topical
  similarity, not evidential bearing — same-topic-no-bearing text is common,
  so similarity must not override the screen). When the LLM drops evidence
  whose claim-similarity is high, log it and optionally demand one re-screen;
  never silently discard against strong deterministic signal (Carnap's
  total-evidence requirement), never auto-keep either.
- Tier-0 verbalized-distribution recipe for `epistemic_classify_question`;
  high-entropy classifications route to the **union** of the top-two types'
  tracks (routing uncertainty must over-verify, never under-verify).
- Pollock defeater typing in `Counterargument.compute_weight()` — treat
  undercutting defeaters (attack the evidence's warrant; the
  replication-failure override is already a special case) differently from
  rebutting defeaters (attack the claim).

## Explicitly NOT building

Re-introducing the counting blend; a full Dung argumentation framework;
multi-call ensembles on every judgment (the Dirichlet band was refuted — one
verbalized histogram per judgment is the validated unit).

## Validation gates

- Items 2b, 5 (posterior/IBE-adjacent): representative epistemic benchmark run
  on the canonical local models before merge — tests passing is not enough
  (standing rule: efficiency/aggregation knobs can regress TMS/IBE silently).
- Items 1, 3, 4: full deterministic suite + targeted regression tests; item 1
  additionally needs a benchmark run to confirm the tripwire's false-positive
  rate (how often it suspends a verdict the benchmark says was right).
- Every new threshold lands in `thresholds.py` with its philosophical
  grounding documented, per the module convention.

## Suggested implementation order

1. **Item 3** (scrutiny verdict) — removes an LLM call, pure win, easiest to
   test deterministically.
2. **Item 2a** (entropy tripwire) + **Item 1** (concordance tripwire) — both
   are suspend-only pure functions; benchmark together.
3. **Item 4** (independence floor) — data plumbing from `structured_data`.
4. **Item 5** (Lipton split) — largest change; benchmark-gated.
5. **Item 2b** (gate soft-counting) — after 5, so the benchmark isolates it.

## Implementation log

- **2026-07-07 — Item 3 DONE.** `scrutiny_weight.py` (new pure module)
  computes strong/moderate/weak/conflicting from per-evidence judgment
  distributions, quality scores, and corroboration counts; the soft-vote
  primitive was extracted to `judgment_signal.support_contradict_split` and
  `confidence._evidence_counting_vote` refactored onto it (behaviour-identical,
  posterior tests unchanged). The `epistemic_assess_evidence` agent, its prompt,
  `AssessEvidenceOutput`, and the conftest mock were deleted. Four thresholds
  added (`SCRUTINY_DEFAULT_QUALITY`, `SCRUTINY_STRONG_MASS_THRESHOLD`,
  `SCRUTINY_WEAK_MASS_THRESHOLD`, `SCRUTINY_CONFLICT_MINORITY_FRACTION`) —
  values initial/benchmark-pending, shape load-bearing. Full epistemic suite
  green (1182 passed); new `test_scrutiny_weight.py` + primitive tests. **The
  cutoff values still want a benchmark pass before they are relied on.**

- **2026-07-07 — Item 4 DONE (DOI/PMID floor only).** `count_independent_sources`
  (pure) + an optional `independent_source_keys` param on `detect_convergence`
  apply a post-computation floor: when evidence provably traces to fewer than 2
  distinct exact-identifier sources, `convergence_detected` is forced False and
  `CONVERGENT → PARTIAL`. Wired in `AssessConvergenceOperation` via the existing
  deterministic `extract_identifiers` (no Evidence data-model change). `independence_group`
  and author matching dropped (see note above). Conservative by construction —
  only downgrades illusory convergence. `test_independence_floor.py` added; the
  convergence/routing suites stay green. **Benchmark still wanted** to confirm the
  floor doesn't over-fire on real runs (should be rare — needs ≥2 items sharing a
  DOI across ≥2 domain clusters).

- **2026-07-07 — Item 2a DONE (mechanism + wiring; benchmark pending).**
  `reproducibility.py` (new): pure `warrants_reproducibility_check` (one-hot +
  directional selectivity), `verdicts_agree` (directional-flip detection), and a
  bounded, strictly-sequential `check_claims_reproducibility` driver. New generic
  `epistemic_paraphrase_claim` agent (`ParaphraseClaimOutput`) reused with the
  existing `epistemic_judge_evidence` — no divergent judge prompt, no runner
  infra change. Wired into `CheckSynthesisDemand` Gate 4: before finalising on a
  decisive posterior, the one-hot judgments underpinning it are re-judged under
  paraphrase (bounded, ≤4 checks); a directional flip converts satisfied →
  needs_more and loops back to `Scrutinize` (suspend-only; bounded by the
  existing `SCRUTINY_RESOLVE_CYCLE_CAP`, so it always terminates). Unit tests
  (`test_reproducibility.py`) + two integration tests (flip→loop-back,
  hold→finalize) added; full suite green (1201 passed). **This adds LLM calls in
  the synthesis loop and changes routing at a decisive gate — it is the most
  benchmark-sensitive of the shipped items and must get a benchmark pass (false
  loop-back rate, cost) before it is relied on.**

- **2026-07-07 — Item 1 DONE (concordance tripwire; benchmark pending).**
  `confidence.ibe_contradicted_by_independent_signals` (pure) fires only when the
  IBE direction is opposed by BOTH the confidence-weighted counting mass AND the
  adversarial band (`_claim_directional_masses` gives the per-claim mass; the
  adversarial leg is the genuinely-independent one, so the conjunction is
  load-bearing). Flagged claims are SUSPENDED — their directional verdict is
  treated as `insufficient` (neutral 0.5), never flipped — in both posterior
  paths: the rule-blind loop directly, and the rule-aware path via a new
  inert-by-default `suspended_ids` param on `combine_claim_verdicts`. Reuses the
  existing `ADVERSARIAL_REFUTED/SURVIVED` thresholds (no new knob).
  `test_concordance_tripwire.py` added; full suite green (1210 passed). **Its
  false-positive rate — verdicts suspended that were actually right — is the
  primary benchmark metric.**

- **2026-07-07 — Item 2b DONE (gate confidence floor; benchmark pending).**
  `gates.count_supporting_sources` now requires each counted supporting source's
  verbalized `judgment_confidence` to clear `GATE_MIN_JUDGMENT_CONFIDENCE` (0.6,
  benchmark-pending). Count stays DISCRETE (Reichenbach distinctness, not
  fractional confidence); one-hot judgments always clear it and distribution-less
  evidence is exempt, so the limiting case is unchanged. Applied to the
  supporting (promotion) count only — the contradict/refutation path is left
  alone for a bounded blast radius. `test_gate_confidence_floor.py` added; full
  suite green (1216 passed). **Threshold value wants a benchmark pass.**

## State after this session (2026-07-07)

Items **3, 4, 2a, 1, 2b** all implemented, unit/integration tested, ruff +
pyright clean (shipped src: 0 errors), full epistemic suite green (1216 passed;
the 2 live OpenAlex smoke tests are network-gated 429s, unrelated). Only **item
5** (Lipton split) remains — it needs the Tier −1 loveliness-histogram
calibration on local models before its elicitation is written.

## Benchmark A/B (2026-07-08) — SciFact dev30, gpt-5.4-nano

Ran the gold-standard SciFact benchmark (now moved into
`benchmarks/scifact/`) as a clean A/B: baseline (changes stashed) vs treatment
(changes applied), same model/cases/providers. Epistemic arm, 20 directional
cases: **AUC 0.83→0.95 (+0.12), SUPPORT recall 0.20→0.30 (+0.10), but
CONTRADICT recall 0.40→0.10, macro-F1 0.49→0.38, ECE 0.086→0.196** — a
trade-off, not a clean win. **All deltas are within huge CIs at n=20 — pattern,
not result.** Root cause traced (case 781): the tightened scrutiny/gates
(items 3 + 2b) can stop a claim from being promoted to a verdict, so it
cycle-caps and the synthesis falls back to "insufficient" (judged NEI) even when
the evidence could decide — costing CONTRADICT accuracy. Full grounded
issue-source audit: **`docs/epistemic/system_issue_audit.md`**. Cheap next step
(no model runs): inspect which gate blocked promotion on the 4 flipped
false-claim cases (781/793/873/1303) — likely a one-threshold fix.

**Benchmark:** a full-pipeline SciFact harness lives at
`benchmarks/epistemic/pipeline_scifact/run.py` (runs `run_epistemic_graph` verify
mode over labeled SciFact dev claims, records verdict/posterior/terminal_state +
instruments reproducibility loop-back rate). Local models are impractical
(~10+ min/claim; gpt-oss:20b n=1 timed out at 600s in evidence-gathering, before
reaching any of the new machinery).

**N=10 cloud run (`openai:gpt-5.4-nano`, 2026-07-07, ~90 min):** directional
accuracy 0.80 (4/5); terminal states completed 6 / oscillation 1; 2 timeouts + 1
retry-exhaustion (all CONTRADICT claims, degraded by SearxNG-down + OpenAlex
429s inflating the per-item quality-invalidation path — pre-existing, confirmed
by db op-counts, e.g. one claim ran `invalidate_evidence`×47). **All five new
items fired ZERO times across every claim** (`reproducibility_loopbacks=0`;
concordance never engaged; verified via per-claim db op-counts — 0 paraphrase
ops anywhere). Conclusions: (1) **no regression, no over-firing, no crashes** —
the changes are correctly dormant and item 3 made scrutiny cheaper; (2) the one
wrong-direction answer (CONTRADICT→SUPPORT) was environmental — the adversarial
search returned `balance=1.0` (vacuous survival, no counter-evidence retrievable
with providers down), which is the SUSPICIOUS zone the system flags but doesn't
block on; item 1 correctly did not fire (no IBE-vs-legs disagreement to detect).
**Honest limitation:** because the tripwires never fired, this run shows they
don't HURT, not that they HELP — a clean-provider run or targeted
tripwire-triggering scenarios are needed to positively validate items 2a/1's
benefit, and the item-3/2b/4 threshold *values* still want a cleaner run.
