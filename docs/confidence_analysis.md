# Model-reported confidence in the epistemic system — analysis before any code

Status: **analysis only, no implementation.** Worktree `epistemic-confidence`.
Scope: `andamentum.epistemic` only.

This memo does three things, in order, as briefed:
1. Reads the `dirichlet_confidence` experiment honestly (including what it refutes).
2. Maps that result onto the philosophical traditions already wired into `epistemic`.
3. Ranks where model-reported confidence could improve the system by **leverage ÷ complexity**, and says plainly where added code would be vibe-coding.

---

## 1. What the experiment honestly shows

Setup: 150 SciFact claims × 5 calls × 2 perturbation arms (reworded phrasing / resampled temperature) × 3 OpenAI models (gpt-5.4-nano, gpt-5.4-mini, gpt-4o-mini). Each call asks for a 3-way histogram over {SUPPORT, CONTRADICT, NOINFO} summing to 100 — a verbalized distribution. Detectors are graded by error-detection AUROC: can the uncertainty score rank the model's *wrong* answers above its right ones?

**The four findings, in plain terms:**

1. **A single verbalized distribution is a genuinely useful "am I wrong?" signal.** The entropy of one histogram reaches AUROC ≈ 0.64–0.77 on the two mid-size models — a real warning light (0.5 = useless).

2. **The cheap single call matches or beats every expensive method.** Self-consistency, semantic entropy, the mixture-of-Dirichlets "band", and mutual information all tie or lose to one-call entropy / five-call predictive entropy. **The Dirichlet band — the elegant idea the whole experiment was built to validate — was refuted.** Vote-based scores (self-consistency, semantic entropy) were the *weakest of all*: collapsing the histogram to its argmax throws away the useful part.

3. **There is a capability floor.** Every self-knowledge signal degrades on the smallest model, and disagreement-based signals degrade fastest. On nano the signal is near coin-flip. *A confidently-wrong model does not know it is wrong, and does not disagree with itself.*

4. **Multi-call earns its cost in exactly one narrow place: confident paraphrase-flips.** When rewording flips the top answer while each individual call still looks confident, a single call is *structurally blind* (each looks sure) but a 5-call predictive-entropy gate catches it. This does not lower total error — it *reallocates* it, trading silent irreproducible flips for stable reproducible errors. It is a **reproducibility** property, not an accuracy one.

**The honest caveats (the experiment states these loudly itself):**

- **n = 150, ~36–52 errors per cell.** CIs are wide (±0.08–0.11); several headline comparisons are formally "no difference."
- **Closed-set 3-class task.** SciFact is the single regime most favourable to verbalized confidence and most hostile to disagreement methods (no free text to cluster). **None of this generalises to free-text generation without a dedicated experiment.**
- **OpenAI models only, and the one *small* model failed.** The positive result is on cloud models with verbalizable confidence. The epistemic system runs on **small local models** (gemma4, gpt-oss) — precisely the regime where the experiment's own evidence is weakest/absent.
- **It is a warning light, never a certifier.** Low entropy never confirms a right answer; the signal only ever raises suspicion.

**The meta-lesson** is the one most relevant to this session's brief: an elaborate, principled apparatus (Dirichlet mixtures, variance decomposition, the GP idea) was built, and the honest verdict is *"one cheap call does as well; don't build the complex thing."* That is a cautionary tale about adding machinery in hope, and it should discipline what we do next.

---

## 2. How this maps onto the philosophy already in `epistemic`

The module encodes five traditions, each as a concrete mechanism:

| Tradition | Mechanism in code |
|---|---|
| Bayesian / Carnap (degrees of belief) | `compute_posterior`, log-odds, `POSTERIOR_*` thresholds |
| Reichenbach (common cause) | ≥2 independent sources → `convergence_verdict`; independence judgment |
| Popper / Lakatos (falsification) | `adversarial_balance`, REFUTED/SURVIVED thresholds, adversarial confidence cap |
| Lipton (inference to best explanation) | loveliness × likeliness, gap-to-runner-up → confidence, framing-tie cap |
| Peirce (fallibilism, self-correcting inquiry) | cycle caps, no certainty, demand-driven lazy escalation |

**The decisive observation:** the system already trusts an LLM's self-reported confidence in **exactly one place** — IBE's `SelectedExplanation.confidence` → `Claim.integrated_confidence` — and it does **not** take the number at face value. It (a) elicits it as "calibrated to the gap to the runner-up," then (b) applies an adversarial cap, (c) a framing-tie cap, and (d) a K-agreement re-run. The architecture's existing stance is therefore already the right one: *a model's stated confidence is an input to be disciplined by structural checks, never a final authority.* That matches the experiment's "warning light, never a certifier" caveat and Peirce/Popper's "belief must survive challenge."

**Where experiment and architecture agree:** eliciting verbalized confidence is worthwhile. The system does it at the top (integration). The experiment validates the practice for the most basic judgment too.

**Where the gap is — and it is a clean one:** the single most frequent and most load-bearing judgment in the pipeline, **`EvidenceJudgmentOutput` (supports / contradicts / no_bearing)**, is collapsed to a hard categorical with **no distribution and no confidence**. That judgment *is the SciFact task the experiment studied* (claim × evidence → 3-way). The experiment's central finding — "don't collapse it; the entropy of the verbalized distribution is your best wrong-answer detector" — applies directly, and the system currently discards exactly that signal at the point it would matter most.

It compounds downstream: the counting posterior weights evidence by `weight = 1 + log(corroboration_count)` — i.e. by **how many times something was said**, not **how confidently each judgment was made**. That is vote-counting, and the experiment found vote-based scores were the weakest of all. The architecture's instinct ("keep the structural discipline") is right; its *substrate* at the evidence layer ("count corroboration") is exactly the thing the experiment says is the lossy choice.

**The capability floor is not a problem for the architecture — it is the architecture's justification.** Because self-confidence collapses on weak models, you cannot replace Reichenbach independence or Popperian adversarial survival with a self-reported number. Confidence is a cheap *prior*; the structural gates are the falsification *discipline*. They are complementary, and the experiment is the reason to keep both.

**The Dirichlet negative result maps straight onto the "don't add code" instinct:** a multi-call disagreement/ensemble/variance-decomposition layer over every judgment would be the textbook over-build. The experiment already ran that test for us and it lost.

---

## 3. Where to improve — ranked by leverage ÷ complexity

### Tier 0 — almost free, highest leverage, most directly supported
**Make the load-bearing categorical judgment a verbalized distribution.**
Change `EvidenceJudgmentOutput` (and structurally-identical judgments) from a bare `Literal` to a 3-way histogram summing to 100 (or add a scalar confidence). `argmax` preserves today's behaviour exactly; the entropy gives the per-judgment "how sure" signal the experiment validated.
- **Cost:** one schema field + adapter. **Same number of LLM calls** — one call asking for three integers instead of a label. Maximally flat schema (3 ints), which is exactly what small local models fill reliably.
- **Why it's first:** it *is* the experiment's task, applied at the system's highest-frequency judgment. No new machinery, no extra calls.
- **The one caveat that gates it:** capability floor. The positive result is on OpenAI models; the small model failed. **Elicit and store it, but do not let anything depend on it until validated on the canonical local models.**

### Tier 1 — moderate, benchmark-gated
**Let that per-judgment confidence inform aggregation, replacing pure vote-counting.**
Fold judgment confidence (or the distribution itself) into `compute_posterior` instead of `1 + log(corroboration_count)`. The experiment's "keep the soft probabilities, vote-counting is weakest" is precisely this.
- **Risk:** this is a change at a convergence site — exactly the "efficiency/aggregation knob that passes tests but silently regresses TMS/IBE" failure mode. **Must be benchmarked, not just unit-tested, before merge.**

### Tier 2 — real complexity, but the *justified* kind
**A demand-gated paraphrase-flip tripwire on decisive, load-bearing judgments only.**
The only thing multi-call genuinely bought. Re-ask a judgment under a paraphrase; if the top answer flips while each call looks confident, flag a silent-irreproducibility landmine. Make it **selective**: fire only when a verdict is both near-decisive and about to be load-bearing (heading to ROBUST/ACTIONABLE, or posterior near `POSTERIOR_DECISIVE_THRESHOLD`). This fits lazy escalation (P7) perfectly — emit a `Demand("reproducibility check")` rather than re-asking everything.
- **Why complexity is justified here:** it catches an error class invisible to everything else, and it is a *reproducibility* property — exactly the FAIR / observability framing of the broader research. But only narrowly, demand-gated; never blanket 5×.

### Tier 3 — explicitly do **not** build
The Dirichlet band, mixture-of-Dirichlets, variance decomposition, GP layer, or "re-ask every judgment 5 ways." The experiment refuted these. This is where the add-add-add instinct would burn effort for no epistemic payoff.

### Tier −1 — the genuinely lowest-risk first move (do before any of the above)
A **small calibration run** of the verbalized-histogram elicitation on the actual local target models (gemma4, gpt-oss) over a SciFact-style slice, reusing the experiment's harness. This answers the one question that gates everything: *does the signal survive on the models we actually run?* If it does, Tier 0 is a confident green light. If it doesn't, we have saved ourselves from building on sand — which is the whole point of stopping to think.

---

## Bottom line
The experiment's honest message and the system's existing philosophy already agree: elicit verbalized confidence, but discipline it structurally; never certify on it. The cleanest improvement is not new machinery — it is to **stop discarding the verbalized distribution at the evidence-judgment layer** (Tier 0), gated by a **small local-model validation** (Tier −1). Everything heavier is either benchmark-gated (Tier 1), narrowly justified and demand-gated (Tier 2), or already refuted (Tier 3).

---

## Tier −1 validation results (local models, 60 balanced SciFact claims, single call)

Harness: `experiments/dirichlet_confidence/validate_local.py` (zero cloud cost). Detector = entropy of one verbalized histogram; yardstick = error-detection AUROC. Balanced 20/20/20 slice → high accuracy → only ~9–12 errors per model, so AUROC CIs are wide; ECE / degeneracy / mean-entropy resolve cleanly on all 60.

**Baseline elicitation (bare schema, exact mirror of the experiment):**

| Model | Acc | ECE | error-det AUROC | degeneracy | verdict |
|---|---|---|---|---|---|
| `gpt-oss:20b` | 0.800 | 0.118 | 0.850 [0.658, 0.982] | 68% | **usable (CI excludes 0.5)** |
| `gemma4:12b-mxfp8` | 0.850 | 0.121 | 0.702 [0.474, 0.913] | 82% | inconclusive |

Key reading: the signal **replicates on a local model** (`gpt-oss`) but is **capability-dependent**, exactly as the original report warned. Notably `gemma` is *more accurate* yet *worse at flagging its errors* — because it is overconfident (82% one-hot). What matters is not accuracy/size but whether the model emits **graded** confidence. `gpt-oss` is a reasoning model (thinks before answering); `gemma` is not — leading hypothesis: gradedness comes from the reasoning step.

**Engineered elicitation (reasoning-first schema + per-field descriptions + calibration-preserving anti-degeneracy nudge), gemma:**

| Metric | baseline | engineered | Δ |
|---|---|---|---|
| accuracy | 0.850 | 0.850 | **+0.000** |
| ECE | 0.121 | 0.039 | **−0.083** |
| error-det AUROC | 0.702 | 0.825 [0.615, 0.980] | +0.123 (crosses to usable) |
| mean entropy | 0.106 | 0.244 | +0.138 |
| degeneracy | 82% | 57% | **−25 pts** |

**Conclusions from validation:**
1. The verbalized-confidence signal is **viable on local models**, but the naive bare-schema elicitation is **not enough for non-reasoning small models**.
2. **Forcing reasoning before the numbers (reasoning-first field) + field descriptions + a calibration-preserving nudge rescues the weak model** — moves gemma from inconclusive to usable, halves degeneracy, cuts ECE by two-thirds, and **changes zero decisions** (accuracy and error count unchanged → it improved *calibration*, not *answers*).
3. Honest limit: the AUROC *gain* is not individually significant (overlapping CIs, ~9 errors); the result rests on the convergence of ECE + degeneracy + entropy + AUROC all moving together, with ECE/degeneracy being the well-resolved metrics.
4. **Design spec for Tier 0** (evidence-based): the epistemic elicitation should (a) put a `reasoning` field first in the output schema, (b) carry per-field descriptions incl. the sum-to-100 contract, (c) include the calibration nudge, and (d) treat `degeneracy` (one-hot output) as a meta-signal that the entropy is uninformative for that call. This must be a **single uniform recipe** (no per-model branching).

**Full 4-way grid (engineered recipe confirmed on gpt-oss too):**

| Metric | gemma·default | gemma·eng | gpt-oss·default | gpt-oss·eng |
|---|---|---|---|---|
| accuracy | 0.850 | 0.850 | 0.800 | 0.783 |
| ECE | 0.121 | 0.039 | 0.118 | 0.114 |
| error-det AUROC | 0.702 [.47,.91] | 0.825 [.61,.98] | 0.850 [.66,.98] | 0.876 [.75,.97] |
| mean entropy | 0.106 | 0.244 | 0.190 | 0.302 |
| degeneracy | 82% | 57% | 68% | 53% |

The engineered recipe is **neutral-to-positive for the already-working model** (gpt-oss AUROC 0.850→0.876, CI tightened, accuracy −1 item): so **one uniform recipe serves both models** — satisfies "one code path, no model-tier branching". Both models reach a usable signal (CIs exclude 0.5). Validation phase complete; **Tier 0 is green-lit** with this elicitation spec. Residual: degeneracy still ~53–57% (≈half of calls give uninformative entropy → the one-hot meta-signal matters); scope is the 3-way evidence-judgment on closed-set SciFact only.
