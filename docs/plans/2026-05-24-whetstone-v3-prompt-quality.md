# PID — whetstone v3 prompt-quality push (release-blocking)

*Status: draft, awaiting approval. Author: Claude. Date: 2026-05-24.*

## 0. Why this exists

The gpt-5.4-mini smoke run exposed a cluster of correctness and quality
problems that the test suite couldn't catch. A focused audit of every
prompt in v3 produced 9 distinct issues. The user has flagged this work
as **release-blocking**: v3 cannot ship until all 9 are addressed
properly, with full confidence in each fix's downstream behaviour.

The cost discipline is inverted compared to a normal feature: spending
2-3 research agents per issue to fully understand the problem and its
ripple effects **before** editing is cheaper than shipping a half-fixed
release. We will dispatch the research first, synthesise it, refine the
implementation specs, and only then write code.

## 1. The 9 issues (recap)

| # | Tier | File | Title |
|---|---|---|---|
| 1 | T1 | `synth.py` | `critique_and_revise` validates against wrong ground truth (author claims, not findings) |
| 2 | T1 | `consolidate.py` | Merges findings without seeing quotes or sections |
| 3 | T1 | `review.py` | `_PROMPT` lacks severity calibration (`minor/moderate/major` = vibe) |
| 4 | T2 | `review.py` | `_PROMPT` doesn't ask for falsifiable / actionable findings |
| 5 | T2 | `extract.py` | `_REQUOTE_PROMPT` doesn't surface common failure causes (whitespace, unicode) |
| 6 | T2 | `extract.py` | `_EXTRACT_PROMPT` doesn't address multi-sentence claims, citation-bearing sentences, tables/equations |
| 7 | T3 | `gaps.py` | `_GAP_PROMPT` has no per-round cap on demands |
| 8 | T3 | `gaps.py` | `_satisfy_reexamine` doesn't see prior findings → may re-flag known issues |
| 9 | T3 | `synth.py` | `synthesise` opinionated about "2-3 sentences synopsis" without document-length awareness |

## 2. How this plan executes

**Phase 1 — research (parallel).** For each of the 9 issues, dispatch
2-3 research agents in parallel. Each agent has a tight, prescribed
question and reports back in ≤200 words. Total: ~22 agents dispatched
in one batch.

**Phase 2 — synthesis.** Read every research report. Produce a
refinement to this PID: per-issue concrete implementation spec, test
plan, and risk notes. Surface anything surprising to the user before
proceeding to code.

**Phase 3 — implementation.** One commit per issue, in dependency
order (prompt-only changes first, schema/input changes after,
behaviour changes last). Canonical-green per commit.

**Phase 4 — verification.** Full whetstone test suite + a smoke
run against `arxiv_1412.6980_v1.md` with both `ollama:gpt-oss:20b`
and `openai:gpt-5.4-mini` to confirm the cascade is healthy and the
report quality has materially improved.

## 3. Research dispatch — per-issue agent briefs

For each issue: 2-3 agents, each with one question, ≤200-word report.
Brief prompts here; the actual dispatched prompts will include the
exact files / line numbers / smoke-run paths each agent should consult.

---

### Issue 1 — `critique_and_revise` ground truth (`synth.py:68-90`)

**Hypothesis:** the critique validates a synthesised review (which
includes weaknesses derived from `findings`) against `model.claims`
(extracted author assertions). Absences flagged as weaknesses cannot
appear in author claims, so the critique may silently delete valid
weaknesses.

- **Agent 1A — empirical:** Trace through the four smoke reports
  (`smoke_gemma26.md`, `smoke_gemma31.md`, `smoke_gptoss.md`,
  `smoke_gpt-5.4-mini.md`). For each, compare the weakness count
  pre- vs post-critique by re-running just the critique step
  against the surviving findings. Report: how often does the
  critique drop weaknesses? Of dropped weaknesses, how many were
  absence-based (would never appear in author claims)?
- **Agent 1B — design:** What does the critique step exist for —
  what failure mode is it catching? Read git blame / commits on
  `synth.py:critique_and_revise`. Should the ground truth be
  findings, claims, both, or section text? Recommend the minimal
  change.
- **Agent 1C — downstream:** What consumes the `StructuredReview`
  that critique produces? Trace through `to_review_result`,
  `_flatten`, renderers. If we change what the critique can
  drop/keep, do any downstream invariants change?

---

### Issue 2 — `consolidate.py` inputs (`consolidate.py:80-95`)

**Hypothesis:** the consolidation agent merges findings using only
`(criterion/severity) issue` as input. Quotes and section_ids — the
data that would tell two near-identical issue strings apart — are
withheld. Risk: false-merge of two distinct findings that happen to
phrase their issue similarly.

- **Agent 2A — empirical:** In each smoke run report, find every
  consolidation event ("N finding(s) → M, merged K group(s)"). For
  the gpt-5.4-mini run specifically: which findings were merged?
  Do they look like legitimate same-point merges or did the agent
  conflate distinct issues? Report findings as a table.
- **Agent 2B — feasibility:** Estimate the prompt token cost of
  adding `quote: "..."` and `section: sX` to each numbered line in
  the consolidate input. With ~10 findings × ~200 chars per quote,
  is the prompt growth acceptable? Are there structural alternatives
  (e.g. group by section_id deterministically *before* asking the
  agent)?
- **Agent 2C — downstream:** When findings merge, the result inherits
  the most-severe member's quote and span (`consolidate.py:111-119`).
  If we feed quotes into the agent and let it pick a "best quote",
  does the merge semantics change? Should it?

---

### Issue 3 — severity calibration (`review.py:_PROMPT`)

**Hypothesis:** `minor/moderate/major` is a free choice — the model has
no rubric, so the distribution drifts.

- **Agent 3A — empirical:** Tally severity distribution across all
  four smoke reports. Is it consistent? Skewed toward "moderate"?
  Random?
- **Agent 3B — prior art:** What rubrics do real peer-review systems
  use (ICML reviews, NeurIPS, journal review forms, IETF errata)?
  Survey 3-5 examples; recommend the simplest 3-tier rubric that
  maps cleanly to our existing `minor/moderate/major` enum.
- **Agent 3C — downstream consumers:** What reads `Finding.severity`
  in v3 and downstream? Renderers, `gate.py`, `consolidate.py`'s
  `_severity_of`, the docx track-changes adapter. If we sharpen
  the rubric, do any consumers rely on a different cutoff?

---

### Issue 4 — falsifiable / actionable findings (`review.py:_PROMPT`)

**Hypothesis:** the current prompt doesn't enforce specificity, so
some findings come out as vague impressions ("the paper's evaluation
is incomplete") that the author can't act on.

- **Agent 4A — empirical:** Read the 8 findings in
  `smoke_gemma31.md` and the 6 in `smoke_gpt-5.4-mini.md`. For each,
  judge: is this actionable (author knows what to change)? Report a
  count.
- **Agent 4B — prompt-engineering:** Look at the
  `andamentum.proofread`, `andamentum.epistemic`, and `whetstone`
  legacy lenses (`src/andamentum/whetstone/lenses/`) for prompt
  language that already enforces actionability. Crib the best.

---

### Issue 5 — requote failure causes (`extract.py:_REQUOTE_PROMPT`)

**Hypothesis:** most requote failures come from whitespace, unicode
differences, or expanded glyphs. The current prompt doesn't say so.

- **Agent 5A — empirical:** From the smoke logs, count how many
  `_REQUOTE` attempts happen (no direct count today — needs grep
  through the v3 log) and what causes them. Sample 5-10 actual
  mismatches: what was the surface-form difference between the
  agent's quote and the source?
- **Agent 5B — locate.py:** Read `locate.py` end-to-end. What does
  it already tolerate (whitespace normalisation, unicode folding,
  etc.) and what does it NOT tolerate? The requote prompt should
  tell the model what's the *actual* mismatch axis.

---

### Issue 6 — multi-sentence claims, tables, figures (`extract.py:_EXTRACT_PROMPT`)

**Hypothesis:** the prompt asks for "verbatim sentences that make a
claim", which forces the model to pick one sentence per claim. Many
contribution statements span 2-3 sentences; the model also has no
guidance on tables, equations, captions.

- **Agent 6A — schema:** Does `Claim.quote` (in `model.py`) allow
  arbitrary length? Does `locate` handle multi-sentence quotes
  correctly? Test with a synthetic example.
- **Agent 6B — empirical:** In the four smoke runs, are any claims
  obviously truncated (one sentence of a clearly multi-sentence
  contribution)? Sample 10 claims from the gpt-5.4-mini run.
- **Agent 6C — corpus:** Look at
  `benchmarks/whetstone/corpus/arxiv_1412.6980_v1.md` — how do
  contributions tend to be expressed (single sentence vs multi)?
  Does the paper have tables/equations the extractor should
  treat specifically?

---

### Issue 7 — gap-loop per-round cap (`gaps.py:_GAP_PROMPT`)

**Hypothesis:** "Be sparing" is too vague. The smoke run produced
exactly 2 demands every round for 3 rounds. A per-round cap with a
shrinking schedule would tighten this.

- **Agent 7A — empirical:** How long do gap-loop rounds take in
  wall-clock per demand? From the gpt-5.4-mini log
  (`smoke_gpt-5.4-mini.md` timing or my conversation history) and
  the gemma logs, compute per-demand latency. Is 6 demands the
  ceiling we want or already too many?
- **Agent 7B — design:** Should the cap be on demands per round, or
  on total demands across the loop, or on wall-clock? Recommend
  one.

---

### Issue 8 — reexamine doesn't see prior findings (`gaps.py:_satisfy_reexamine`)

**Hypothesis:** reexamine could re-flag issues the cascade already
raised. Pre-seeding it with current findings would help it focus.

- **Agent 8A — empirical:** In the gpt-5.4-mini smoke report, did
  the gap loop's reexamine findings overlap with what the cascade
  would have produced if it hadn't been killed by usage limits?
  Hard to know without re-running — instead, compare the
  gpt-5.4-mini report against the gemma31 report (where the
  cascade DID produce findings) and look for overlap in the
  conceptual issues raised.
- **Agent 8B — token cost:** How big is the prior-findings list
  typically? With 6-10 findings × ~150 chars each = ~1.5KB of
  prompt growth per reexamine call. Acceptable?

---

### Issue 9 — synopsis length (`synth.py:synthesise`)

**Hypothesis:** "2-3 sentences" is fine for typical papers but
off for short tech notes or sprawling manuscripts.

- **Agent 9A — empirical:** What's the synopsis length across the
  four smoke reports? Did the model obey the "2-3 sentences"
  instruction? Was it always the right length?
- **Agent 9B — design:** Should length scale with document length,
  with finding count, or just be left to the model with looser
  guidance ("1-5 sentences")?

---

## 4. After research — synthesis and refined specs

Once all ~22 research agents return, this plan grows a **§5.X
"Implementation spec"** subsection per issue, containing:

- The concrete edit (prompt text diff, schema change, etc.)
- Test plan (which existing tests are affected, what new tests to add)
- Risk notes (downstream consequences spotted by the research)

Surface to the user any research finding that:

- changes the hypothesis (e.g. "actually the critique never deletes
  anything — the issue is elsewhere")
- reveals an unexpected dependency (e.g. "consolidate's quote handoff
  is load-bearing for the docx renderer")
- suggests a different fix than the audit proposed

The user signs off on the refined per-issue specs before we cut code.

## 5. Implementation order

After research and sign-off, land in this order — minimises
inter-commit churn:

1. **Prompt-only, no schema/input change** (issues 3, 4, 5, 7, 9).
   These touch only the system-prompt strings — no behavioural
   ripples beyond what the model writes.
2. **Input/schema changes** (issues 2, 8). Consolidate sees more
   data; reexamine sees prior findings. Tests need new fixture data.
3. **Behaviour / contract changes** (issues 1, 6). The critique
   ground-truth change is the largest behavioural delta; the
   multi-sentence claim allowance may affect locate / consolidate
   downstream.

Each commit:
- single issue, single commit
- includes any new/changed tests
- passes `uv run pytest src/andamentum/whetstone` (~580 tests)
- passes `uv run pyright src/andamentum/whetstone/v3/` clean
- passes `uv run ruff check src/andamentum/whetstone/v3/`

## 6. Verification — what "done" looks like

1. **All canonical green.** 23 pyright errors (the documented
   test-only baseline), pytest fully passing, ruff clean.
2. **Smoke re-run.** Against
   `benchmarks/whetstone/corpus/arxiv_1412.6980_v1.md`:
   - **gpt-5.4-mini**: cascade contributes ≥3 of 5 criteria with
     real findings (vs 0/5 today); fewer than 5 silent
     `verify_findings dropped` lines (vs 5 today); consolidation
     visibly preserves distinct findings.
   - **ollama:gpt-oss:20b**: 6-10 final findings; severity
     distribution matches the new rubric (not all-moderate).
3. **Manual quality read.** I open each report, read the findings,
   and judge whether each one would survive a real reviewer's
   "would this be useful to the author?" test. The user does the
   same and signs off.

## 7. Decisions (resolved 2026-05-24)

1. **Commit shape**: one commit per issue, 9 commits total. Each
   commit captures that issue's research findings in the message body.
2. **Evidence base**: existing four smoke reports only
   (`smoke_gemma26.md`, `smoke_gemma31.md`, `smoke_gptoss.md`,
   `smoke_gpt-5.4-mini.md`). No fresh smoke run before dispatch.
3. **Scope drift**: if any research report contradicts the audit
   hypothesis or reveals a larger fix is needed, pause and surface
   to user before writing code. No silent scope expansion.
4. **Branch**: stay on `whetstone-iterative-review`. All 9 commits
   land on the existing iteration branch; the v3 release PR rolls
   them up to `main` at the end.

---

## 8. Phase 2 — research synthesis (2026-05-24)

All 22 agents returned. Per-issue summaries and the refined implementation
specs follow. Surprises and decision-points are flagged ⚑.

### Issue 1 — critique ground truth (`synth.py`)

**Research summary.** Hypothesis confirmed at the empirical layer: of 13
weaknesses across 4 reports, only 4 are "positive overclaims" that
`model.claims` can validate. 5 are absence-based and 4 are
typo/presentation — both classes the current critique would drop as
"unsupported by claims." The gemma26 / gpt-oss runs are particularly
exposed: their entire weakness lists could vanish under a strict critique.

**Original intent** (per docstring + commit `3fb9604`) was a faithfulness
gate on the *narrative* — drop synthesised text the document doesn't
support. Right ground truth = section gists + claims + the findings list
itself (findings are already passed through the hallucination gate in
`gate.py`, so they're trustworthy here).

**Downstream is forgiving.** Renderers handle empty fields. Docx anchors
are tied to `Finding.quote`, independent of the synth draft. Test
`test_to_review_result_maps_quotes_to_section_relative_offsets` already
exercises empty strengths/weaknesses. Safe to ship.

**Implementation spec.** In `critique_and_revise`:
- Add `findings: list[Finding]` parameter (caller `graph.py:149` already has them in `ctx.state`).
- Pass three labelled blocks: `SECTION GISTS`, `AUTHOR CLAIMS`, `SUPPORTED FINDINGS`.
- Update prompt: strengths/synopsis must be supported by gists or author claims; weaknesses must correspond to a listed finding (do NOT drop a weakness just because the author doesn't assert the gap).

**Tests.** Update existing `test_gate_synth.py` to assert weaknesses
survive when supported by findings; add one for absence-based
weakness preservation.

### Issue 2 — consolidate inputs (`consolidate.py`)

**Research summary.** Hypothesis confirmed: gpt-5.4-mini run showed one
likely-good merge AND one suspected bad merge (collapsed two distinct
math problems — "missing sqrt" + "stated bound wrong" — into a single
statement that only mentions the sqrt). Token cost of adding
quote+section to each line is negligible (under 3k tokens at N=20
findings; both gpt-5.4-mini and local Gemma have ≥128k context).

⚑ **CRITICAL — discovered in 2C.** The docx renderer uses
`Finding.quote` as an EXACT-MATCH anchor (`whetstone/docx/anchor.py`
normalises both sides; no fuzzy fallback). If we let the consolidate
agent rewrite quotes (e.g. pick a "best quote" or merge them), comments
silently fail to anchor in the .docx output. Quote ↔ Span invariant
(`verify_findings` already ran) would also break.

**Implementation spec.** Agent receives quote+section as DATA (so it can
distinguish near-duplicates), but does NOT pick/rewrite quotes. Keep
the existing deterministic "most-severe member's quote + span"
tiebreaker. Per-line format:
```
  [i] (Criterion/severity, section=sX) issue text
      quote: "verbatim quote"
```

**Tests.** Update `test_consolidate.py` to assert the merge-anchor still
inherits the most-severe member's quote (no rewrite path).

### Issue 3 — severity calibration (`review.py`)

**Research summary.** Hypothesis confirmed: 58% "moderate", 35% "major",
8% "minor" across 26 findings. Every model uses 2-3 "major" regardless
of total volume — a loose "major ceiling" rather than calibrated
thresholds. gemma31 never picks "minor". gpt-5.4-mini also never picks
"minor" (0/6).

**Downstream consumers**: NO consumer depends on a specific severity
proportion. Gate.py uses severity for ordering + most-severe-wins on
overlap (no thresholds); renderers group by priority bucket without
minimum-count assumptions; tests use hand-constructed fixtures
unrelated to live distribution. Safe to ship a sharper rubric.

**Implementation spec.** Add this rubric block to `review.py:_PROMPT`
(from agent 3B, verbatim — drop-in):

```
Severity rubric — pick the level by what the author would have to do:

- major: the paper's conclusions, validity, or reproducibility are at
  stake. Ignoring this would leave the work wrong, unsupported, or
  unusable. Author must address before the paper is sound.
- moderate: a real weakness that a competent reader will notice and
  that weakens the paper, but conclusions survive if it stays. Author
  should fix to strengthen the work.
- minor: a local improvement — wording, typo, formatting, a single
  sentence that could be sharper. Safe to ignore; nice to fix.

When uncertain between two tiers, pick the lower one.
```

**Tests.** No test changes required (no test asserts a numerical
proportion). Optional: add a smoke-style integration check.

### Issue 4 — actionable findings (`review.py`)

**Research summary.** 13 actionable / 3 vague / 7 borderline of 23
findings — vague clustered in gpt-5.4-mini. Existing andamentum prompt
language to crib from in `whetstone/agents/author_question.py`,
`whetstone/agents/lens_prompts.py`, `epistemic/agents/preplanning.py`.

**Implementation spec.** Append to `review.py:_PROMPT` (from 4B, slight
edit):

> Every finding must be author-actionable: the issue description should
> name what the author would change, add, or verify to resolve it. If
> you cannot say what a fix would look like — even abstractly — the
> finding is too vague to keep; either sharpen it or omit it.

**Tests.** Prompt-only; no test changes required.

### Issue 5 — requote prompt (`extract.py`)

**Research summary.** ⚑ **Diagnosis correction.** The "weird notation"
in smoke quotes (`glyph[lessorapproxeql]`, `β 1`, `θ t -1`) is in the
SOURCE itself — Docling PDF→markdown artefact. Quotes survive locate
because both sides normalise identically. The real risk: an LLM that
"helpfully" rewrites these to clean math symbols would silently fail
locate and bias findings toward sentences with no math.

`locate` tolerates: whitespace folding, case, markdown markers
(`# * _ \` [ ] ~`). Does NOT tolerate: unicode folding, smart vs
straight quotes, dash variants (`-/–/—`), ligatures (`ﬁ` vs `fi`),
ellipsis (`…` vs `...`).

**Implementation spec.** Add to `_REQUOTE_PROMPT` (from 5B):

> Punctuation must match byte-for-byte: copy the exact dash variant
> (- vs – vs —), exact quote marks (straight " ' vs curly " " ' '),
> ellipses (… vs ...), and any accents or special characters as they
> appear in the section. Whitespace, case, and markdown emphasis
> (`**`, `*`, `_`) do not matter — but every other character must be
> identical. Do not rewrite math notation, glyph names, or
> Docling-style spaced symbols (`β 1`, `glyph[circledot]`); keep them
> exactly as the section text shows them.

**Tests.** Prompt-only.

### Issue 6 — multi-sentence claims + special elements (`extract.py`)

**Research summary.** Schema and `locate` already support multi-sentence
quotes (no length cap, `normalize_with_map` collapses sentence-break
whitespace). Renderers preview-truncate at 140-200 chars but store the
full quote. Downstream is safe.

Empirical: 5 self-contained / 4 truncated-looking / 1 compound out of
10 sampled. The 4 truncated cases lose precisely the qualifier/scope
sentence that follows the claim sentence (e.g. `"…regret O(T) for the
online convex function…"` is missing the following `"Our result is
comparable to the best known bound for this general convex online
learning problem."` — exactly the scope detail needed to judge
overclaim).

⚑ **Corpus addition.** The Adam paper has `<!-- formula-not-decoded -->`
placeholders (Docling artefact) where equations should be. Algorithm 1
is rendered as flowing prose. Figure captions exist and occasionally
carry the only mention of a comparison. The extract prompt has no
guidance for these.

**Implementation spec.** Update `_EXTRACT_PROMPT`:
- Allow 2-3 sentence claims when the claim spans a claim+scope-qualifier
  pattern, or when an equation/bound is the load-bearing payload.
- Algorithm pseudocode is EVIDENCE not claim — the claim is in the
  surrounding prose. Don't extract individual pseudocode lines.
- `<!-- formula-not-decoded -->` placeholders mean an equation was
  stripped — extract the surrounding prose sentence that names the
  result, accept the equation itself is unrecoverable.
- Figure captions are valid claim sources but usually redundant with the
  paragraph that references them; prefer the prose unless the caption
  carries a unique claim.

**Tests.** Add `test_extract.py` cases for: 2-sentence claim accepted;
algorithm-pseudocode-skipped; `<!-- formula-not-decoded -->` prose
fallback.

### Issue 7 — gap-loop demand cap (`gaps.py`)

**Research summary.** ⚑ **No wall-time data available** in the smoke .md
files — only finding counts. Cannot empirically measure gap-loop's
fraction of total review time from these inputs. Recommend instrumentation
alongside the cap.

Agent 7B's tradeoff analysis recommends OPTION A (per-round demand cap).
With `_DEFAULT_CAP = 2` rounds × `_DEFAULT_PER_ROUND_DEMANDS = 3` per
round = 6 LLM calls structural ceiling in the gap loop.

**Implementation spec.**
- Add `_DEFAULT_PER_ROUND_DEMANDS = 3` constant in `gaps.py`.
- In `gap_loop`, after `analyze_gaps` returns: `demands = demands[:per_round_demand_cap]`.
- Add `per_round_demand_cap` parameter to `gap_loop()`, default `_DEFAULT_PER_ROUND_DEMANDS`.
- Add `logger.info` timing instrumentation around each round (start time, demands fired, findings added, wall time) so future audits have data.

**Tests.** Add `test_gaps.py` case asserting the cap truncates when
`analyze_gaps` returns more than the cap.

### Issue 8 — pre-seed reexamine with prior findings (`gaps.py`)

**Research summary.** Strong cross-run topical overlap evidence: s7
regret/convergence, s4 efficiency rewrite, s5 effective-stepsize bound
all flagged by multiple models independently. The current dedup is at
the *demand signature* layer, not the *finding* layer.

Token cost: adding ~8-12 prior findings as `[criterion/severity] issue`
adds 500-1050 tokens — well under any context-window risk. Quote field
omitted (per 8B — section text is already in the prompt, model can
re-anchor).

**Implementation spec.** Update `_satisfy_reexamine` to accept a
`prior_findings: list[Finding]` parameter. Prompt grows by a "PRIOR
FINDINGS (do not re-raise these)" block. Caller `gap_loop` passes
`current` (the accumulated findings list).

**Tests.** Add `test_gaps.py` assertion that prior-findings block appears
in the prompt; verify token-cost ceiling.

### Issue 9 — synopsis length (`synth.py`) ⚑

**Research summary** — ⚑ **GENUINE TENSION BETWEEN 9A AND 9B**:

- **9A (empirical)**: all 4 models obeyed "2-3 sentences" (all produced
  exactly 2). For a ~5000-word ML paper the test corpus is, 2 sentences
  is too tight — they cover only "what it is" + one global stance, and
  cannot name where the issues cluster (correctness vs. presentation
  vs. evaluations). Recommends scaling with document length.
- **9B (design)**: recommends staying at "2-3 sentences." Argument:
  synopsis serves a skim role; strengths/weaknesses bullets are right
  below; document-length scaling rewards verbosity; small local models
  obey explicit counts much more reliably than vague guidance.

**Decision required.** Three real options:
- A. Stay at 2-3 sentences (9B's call) — predictable, small-model-friendly,
  may be too tight for long manuscripts.
- B. Scale with document length, three bands: ≤500 words → 1 sentence;
  500-5000 words → 2-4 sentences; >5000 words → 4-8 sentences. Computed
  in the prompt-builder, not the prompt.
- C. Loose guidance: "concise summary, no more than a small paragraph
  (typically 2-5 sentences)" — risks small-model non-compliance.

**Recommendation (mine):** B — scaled bands. The empirical evidence
(9A) is concrete: 2 sentences is genuinely too tight for the corpus
papers v3 will see. 9B's "small models obey counts" argument can be
preserved by giving each band a specific count range; the prompt-builder
picks the band based on `len(model.source)`.

### Other tensions / scope additions

- ⚑ **Issue 6 corpus-specific guidance** (equations, pseudocode, figure
  captions) is an ADDITION beyond the original audit hypothesis. It's
  not a contradiction — extracting against a Docling-rendered paper
  needs this regardless — but it slightly expands Issue 6's scope.
- ⚑ **Issue 7 instrumentation** is a small ADDITION (timing logs) so
  future audits have data. Recommended despite not being in the
  original audit. Cheap to add.

## 9. Refined implementation order

After all decisions, land in the order originally specified — prompt-only
first, then schema/inputs, then behaviour:

1. **Issue 3** — severity rubric (prompt-only)
2. **Issue 4** — actionability nudge (prompt-only)
3. **Issue 5** — requote punctuation guidance (prompt-only)
4. **Issue 7** — gap-loop cap + timing instrumentation
5. **Issue 9** — synopsis length (after user picks A/B/C)
6. **Issue 2** — consolidate inputs (schema growth, no quote rewrite)
7. **Issue 8** — reexamine prior-findings seeding (input growth)
8. **Issue 1** — critique ground truth (behaviour change)
9. **Issue 6** — extract multi-sentence + special-elements guidance

Each commit: canonical-green per-commit, single issue, includes new/
changed tests.

## 10. Decisions (resolved 2026-05-24)

1. **Issue 9 synopsis length** — Option B (scaled 3-band): ≤1000 words
   → 1 sentence; 1000-5000 → 2-4 sentences; >5000 → 4-8 sentences.
   Computed in the prompt-builder.
2. **Issue 6 scope** — wide. Multi-sentence allowance PLUS guidance for
   algorithm pseudocode, formula-not-decoded placeholders, and figure
   captions.
3. **Issue 7 instrumentation** — add timing logs in the same commit as
   the demand cap.

---

Proceeding to Phase 3 implementation.
