# Where the epistemic system's issues come from — a grounded audit

Status: **analysis, 2026-07-08.** Written after an A/B benchmark of the
harness-backstop changes (see `harness_backstops_design.md`) on SciFact dev30.
It is deliberately self-contained and jargon-free — you should be able to read
it without knowing the codebase.

---

## What prompted this

We ran the same 30-claim SciFact benchmark twice on the same cloud model
(`openai:gpt-5.4-nano`), same cases, same evidence providers — the *only*
difference being the five harness-backstop code changes (baseline = changes
stashed; treatment = changes applied). The point was to isolate what the changes
actually do. The headline was a **trade-off**, not a clean win, and digging into
*why* surfaced the system's real issue sources.

### The A/B result (epistemic arm, 20 directional cases: 10 true, 10 false)

| metric | baseline (no changes) | treatment (with changes) | Δ |
|---|---|---|---|
| AUC-ROC (ranking quality) | 0.830 | 0.950 | **+0.12** |
| SUPPORT recall (true claims caught) | 0.20 | 0.30 | **+0.10** |
| Brier | 0.174 | 0.178 | ~flat |
| ECE (calibration error) | 0.086 | 0.196 | **+0.11 worse** |
| macro-F1 | 0.491 | 0.375 | **−0.12 worse** |
| CONTRADICT recall (false claims caught) | 0.40 | 0.10 | **−0.30 worse** |
| NEI recall | 1.00 | 0.90 | −0.10 |

Ranking and true-claim detection improved; false-claim detection and overall
calibration got worse. **Every number here has a huge uncertainty band — the
evaluation is 20 cases (see issue #7), so treat the pattern as a hint, not a
result.**

### The concrete finding that explains the trade-off

On the false-claim cases, the *confidence number* was 0.5 ("unsure") in **both**
runs — yet the graded verdict flipped from CONTRADICT (baseline) to NEI
(treatment). Case 781 (a genuinely false claim about interferon-γ and
myocarditis) shows why:

- **Baseline** synthesis: "Claims Established: **1 of 1**", verdict **"No"**,
  body lays out the refuting studies → grader reads **CONTRADICT** (correct).
- **Treatment** synthesis: "Claims Established: **0 of 1**", verdict
  **"Insufficient evidence to answer"** → the claim was **cycle-capped** before
  it could earn a verdict → grader reads **NEI** (wrong).

The claim didn't get "more honest prose" — it *mechanically failed to climb the
ladder* that earns a verdict, because the stricter scrutiny/gating in the
changes stopped it. That is issue #2 below, and it is the one genuinely
actionable, change-caused problem.

---

## How the system works (needed for the rest)

You hand it one claim. Then:

1. **It searches external databases** (PubMed, OpenAlex, …) and pulls back
   evidence snippets.
2. **It scores and judges each snippet** — quality, and whether it supports /
   contradicts / doesn't bear on the claim.
3. **The claim climbs a ladder of stages** — hypothesis → supported →
   provisional → robust. Each rung has a **gate** (enough good evidence,
   survived scrutiny, …) it must pass.
4. **If it climbs high enough**, a reasoning step stamps a **verdict + a
   confidence number** (the "posterior", 0–1).
5. **It writes a prose report** (the "synthesis").
6. **In testing, a separate AI "judge"** reads *only the prose* and labels it
   SUPPORT / CONTRADICT / NEI. Accuracy is scored on that label; calibration on
   the confidence number.

---

## The issue sources

### 1. Three answers that can disagree
The system emits a **confidence number**, a **prose report**, and (in testing) a
**judge's label** read off that prose — and nothing forces them to agree. We saw
the number say 0.5 in both runs while the graded label flipped. Any measurement
is really measuring one of three loosely-coupled things, and which one you look
at changes the story.

### 2. Claims must "earn" a verdict — a blocked claim falsely says "I don't know" *(actionable; caused by the changes)*
A claim only gets a yes/no if it climbs the stage ladder. If it can't pass a
gate, it retries a fixed number of times, gives up, and the system writes
"insufficient evidence" — **even when the evidence was enough to decide**. Case
781 is the proof: the refuting studies were present (baseline answered "No"), but
the stricter gates stopped the claim from climbing, so it timed out and declared
"insufficient" → a correct "false" became a wrong "don't know". Two of the five
changes — the **deterministic scrutiny weight** and the **"shaky evidence
doesn't count toward promotion" rule** — tightened exactly those gates. This is a
real tuning risk the changes introduced, not noise.

### 3. Only as good as what it retrieved
Every answer rests on what the databases returned for that claim. Rate limits /
outages can make the system score all its evidence as junk and discard it (this
happened earlier — one database's 429 errors collapsed evidence quality and the
system couldn't answer). And plain retrieval luck: if the one refuting paper
isn't surfaced, the system cannot know the claim is false. A chunk of "accuracy"
is really "did retrieval get lucky", which is invisible in the final number.

### 4. Disproving is harder than confirming
Confirming "X is true" can take one supporting paper; establishing "X is false"
requires actively finding refuting papers — a harder search the system commits to
more cautiously. **CONTRADICT is the weakest class in every run** (it was weak in
the old frozen results too). A structural bias, not a bug — but it means false
claims are where the system most often retreats to "I don't know".

### 5. The final grader is itself a fallible AI reading prose
The SUPPORT/CONTRADICT/NEI label comes from a *second* AI reading the report.
Reliable on a clear report ("Verdict: No"); a coin-flip on a hedged one, and
swayable by wording over substance. Some of the CONTRADICT→NEI shift is the
grader reacting to phrasing, not to a worse conclusion. The yardstick has its own
measurement error.

### 6. Many AI steps in a chain — errors multiply
One claim triggers many separate AI judgment calls: judging each snippet,
quality-scoring, scrutiny, best-explanation selection, report-writing, then
grading. Each is fallible; a wrong evidence-judgment early quietly poisons
everything downstream, with no single place to catch it. (One case in this run
just *timed out* mid-pipeline on a transient network hiccup.)

### 7. The evaluation is too small to trust small differences
All of the above is measured on **20 directional cases (10 true, 10 false)**. One
case changing moves a class's score by 10 points; the proper confidence range is
enormous (the old frozen run's headline metric spanned "coin-flip" to "perfect").
No single delta here is real signal yet — only the *pattern* is worth believing.

---

## Bottom line

The one issue that is both **actionable and caused by the changes** is **#2**:
the tightened scrutiny/gates make some claims fail to earn a verdict and fall
back to "insufficient", and on false claims that costs real accuracy. The rest
(#1, #3–#7) are pre-existing structural properties of this kind of system —
important for understanding why the numbers are noisy and why "proving false" is
the soft spot, but not things the five changes broke.

### Cheap next step (no model runs)
Inspect the four false-claim cases that flipped to "insufficient" — **781, 793,
873, 1303** — and check *which* gate blocked promotion on each: the deterministic
scrutiny verdict (item 3) or the "shaky evidence doesn't count" rule (item 2b).
If it's consistently one threshold set too conservatively, it's a one-knob fix,
not a redesign. Results live under
`benchmarks/scifact/results/dev30_baseline/` and `.../dev30_backstops/`
(per-case `epistemic/synthesis.md`, `claims.jsonl`, `operations.jsonl`).
