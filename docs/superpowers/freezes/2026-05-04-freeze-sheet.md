# Freeze sheet — 2026-05-04

This document is the close-of-session record for 2026-05-04. It stacks
on top of [`2026-05-03-freeze-sheet.md`](./2026-05-03-freeze-sheet.md)
and [`2026-05-03-evening-freeze-sheet.md`](./2026-05-03-evening-freeze-sheet.md);
both remain in force unless explicitly called out.

The methodology is unchanged: declare a *frozen set* (off-limits, even
when tempting) and a *target set* (where the next session works). Bugs
in the frozen set become entry conditions for future sessions, not
in-flight fixes.

---

## Session arc

The session opened against the previous freeze sheet's K8 — the load-
bearing search-quality blocker — and chose entry point (a) per that
sheet: diagnose-then-fix on the provider/query-formulation layer.

The diagnostic phase, working from probe B5's saved DB, established two
distinct bugs in the regular evidence flow:

- **Bug #1** — the round-1 provider ranker (`epistemic_rank_providers`)
  collapsed to picking the same top provider (cochrane) for all 4
  sub-claims, rather than diversifying across them. Deferred per
  user's session sequencing.
- **Bug #2** — the query formulator (`epistemic_formulate_query`) was
  provider-aware in name only. The provider name and description were
  inputs but the prompt gave no syntax-specific guidance, so the LLM
  produced Google-style natural-language queries (with `site:`
  operators sent to PubMed/EuropePMC) regardless of which biomedical
  API would receive them.

Bug #2 was the session's target. The fix was data-layer-only:
each provider self-advertises its native query language and a
catalogue of valid query styles, which the formulator reads.

The session closed having **structurally fixed Bug #2** — verified
on probe B6 (formulator-only, 16 queries, all in native syntax with
genuine per-provider divergence) — but **without movement on the
headline supports rate** (probe B6b, end-to-end on sub-claim A: 0
supports out of 40 items, comparable to B5's 1 / 71). The headline
gap is now bottlenecked downstream (provider deduplication + judge
prompt + seed-claim narrowness), not upstream in query formulation.

The session's load-bearing methodological finding is one the user
named explicitly mid-session: **don't iterate prompt engineering on
N = 1 probes.** I had drafted a Cochrane-specific guidance tweak based
on a single sub-claim's results; the user pushed back, I audited my
reasoning, and we agreed the speculative tweak doesn't ship. Pattern
matches the earlier `feedback_efficiency_changes_can_regress_quality`
memory.

---

## Frozen set — DO NOT edit in the next session

### A. Bug #2 fix — provider `query_guidance` (commit on this branch)

**One feature commit landed:**

| Concept | Where | What changed |
|---|---|---|
| Registry | `epistemic/providers/__init__.py` | New `PROVIDER_QUERY_GUIDANCE: dict[str, str]`. `register_provider()` takes a `query_guidance=` kwarg parallel to `description`. Each of the 10 built-in providers now registers a self-describing query catalogue (5–8 syntactically distinct valid styles), framed as "all of these work" so small LLMs vary their output. |
| Formulator agent | `epistemic/agents/preplanning.py` | `FORMULATE_QUERY_PROMPT` rewritten to read `query_guidance` as a separate input, explicitly license divergent styles, and drop the "5–15 words" length cap (which was biasing toward bag-of-words). |
| Output schema | `epistemic/agents/output_models.py` | `FormulateQueryOutput.query` field description no longer says "5–15 words" (anchored the LLM toward the wrong form). |
| Call site | `epistemic/operations/preplanning.py` | Both formulator call sites (round-1 multi-sub-claim and original per-objective branches) pass `query_guidance=PROVIDER_QUERY_GUIDANCE.get(...)` alongside `provider_description`. |
| Docs | `epistemic/providers/CONTRIBUTING.md` | Example `register_provider` call shows the new kwarg + how to write a guidance block. |

**Test counts:** 1887 passed, 2 skipped, 7 deselected. Pyright: 24 →
24 (no new errors; pre-existing whetstone variance pattern unchanged).
Ruff: clean.

**Unchanged contract:** the `description` field still drives the
ranker (`epistemic_rank_providers`); only the formulator reads
`query_guidance`. Splitting them lets each agent's prompt stay narrow.

### B. Empirically verified

Two probes ran against the new build:

| Probe | Setup | Result |
|---|---|---|
| **B6** (formulator-only) | 4 sub-claims × 4 providers = 16 formulator calls. No provider hits, no judge calls. Compared queries to B5 baseline. | **All 16 queries shifted to native syntax.** Zero `site:` operators (B5 had three). PubMed/EuropePMC use Boolean + field tags + MeSH. Cochrane queries are appropriately shorter. OpenAlex stays plain text. Per-provider divergence confirmed: same sub-claim produces 4 syntactically different queries across providers. |
| **B6b** (one-claim end-to-end) | Sub-claim A → formulate → real provider gather() → judge each item. 40 items total. | **0 supports out of 40 (vs B5's 1 / 71).** The query change reaches better neighborhoods (CD012906 visible in earlier exploratory variant; PubMed/EuropePMC return on-topic metformin RCTs) but the headline rate didn't move. New downstream bugs surfaced — see Section C. |

The probe scripts and full output logs live at `/tmp/probes/B6/`
(`formulator_probe.py`, `end_to_end_probe.py`, `dump_titles.py`,
`inspect_cd012906.py`). They are outside the repo and not retained
across reboots; treat them as one-off observation primitives, not
reusable infrastructure.

---

## Known broken things in the frozen set — do not fix in flight

The Bug #2 fix made several previously-masked downstream bugs visible.
Each is logged with the evidence we have, *and an explicit caveat that
the evidence is N = 1 and a real fix needs multi-claim, multi-question
verification first*.

| ID | Where | What's broken | Evidence | Confidence |
|---|---|---|---|---|
| **K10** | `cochrane` provider's gather → NCBI esearch | NCBI returns each Cochrane review **revision** as a separate record. On probe B6b sub-claim A, 5 of the 10 returned items were CD002967 versions (`CD002967`, `.pub2`, `.pub3`, `.pub4`, plus a duplicate). One off-topic review filled half the top-10. The provider does not dedupe by base DOI before returning. | Direct (the dump titles output is the receipt). Mechanism is structural, not LLM-stochastic. | High. |
| **K11** | `epistemic_judge_evidence` prompt + the seed-claim text | The seed claim says "*versus placebo or control treatments*". The judge interprets "control" as **"no drug"** rather than as **"comparator"**. Active-comparator metformin RCTs (vs sulphonylurea, vs DPP-4, etc.) were judged `no_bearing` because they aren't strictly placebo-controlled. Most diabetes RCTs use active comparators because denying T2DM treatment is unethical, so this rules out the bulk of the actual evidence base. | Probe B6b sub-claim A: ~30 of 40 items judged `no_bearing` cited "active comparator" or "add-on" reasoning. One charitable judgment did fire (PMID 38763510, ATOMIC trial). | Medium. The judgments are internally consistent with a strict reading; whether the right fix is in the claim text, the judge prompt, or both is unclear from one question. |
| **K12** | `epistemic_judge_evidence` schema | The judge can produce `in_scope=True` AND `verdict="no_bearing"`. The judge prompt's own rule says: *"If `in_scope` is True, set `verdict` to "supports" or "contradicts"."* The pydantic schema doesn't enforce the cross-field constraint because `verdict` is a Literal of three values and they're each independently valid. Observed once on CD012906. | Probe inspect_cd012906.py output. One occurrence; mechanism is structural (schema-level), so it can fire any time. | High on existence; low on frequency (1 / 40 in the dump). |
| **K13** (open hypothesis, NOT a confirmed bug) | Cochrane `query_guidance` text in `providers/__init__.py` | Possibly the Cochrane guidance over-emphasises brevity ("3–7 tokens"). On probe B6b sub-claim A, the formulator produced `"Metformin"[MeSH] AND "Diabetes Mellitus, Type 2"[MeSH]` — no outcome term. The PubMed query for the same sub-claim included `(HbA1c[tiab] OR "Glycated Hemoglobin"[MeSH])`. Hypothesis: when a claim names a specific outcome, the Cochrane query should include it. | Single sub-claim, single run. | **Low — explicit "do not change without multi-claim, multi-question evidence."** May also be confounded by K10 (deduplication): even with HbA1c added, CD002967's abstract probably mentions HbA1c, so the duplicates would still dominate. |

K10 (deduplication) is the most mechanically clear and the cheapest
to fix. K11 (judge "control") is the largest semantic fix and
genuinely contested. K12 (schema enforcement) is small, mechanical,
and narrow in scope. K13 stays as a *hypothesis* until tested.

The previous freeze sheets' open items remain unchanged:

- K8 (search quality, marked load-bearing) is **partially addressed**
  by this session's Bug #2 fix on the formulator side. Bug #1
  (provider rank diversity) is still deferred.
- K4 step 2+3 (writer-validator timing characterisation) — still
  deferred. Probe B6b ran but did not reach `Synthesize`, so we still
  haven't observed the 90 s synthesis call in production.
- K9 (claim-level "contested" terminal) — still deferred.

---

## Findings worth remembering

1. **Catalogue-style guidance prevents single-template collapse on
   small LLMs.** Both the user and I were specifically worried that
   adding query examples would cause the formulator to produce one
   canonical form. Probe B6 disproved that on `gpt-5.4-nano`: across
   16 calls (4 sub-claims × 4 providers), each sub-claim received
   4 syntactically distinct query forms — Cochrane stayed short, PubMed
   used MeSH + `[pt]`, EuropePMC used `(TITLE: OR KW:)`, OpenAlex
   stayed plain text. The "all of these work" framing did its job.

2. **Removing length anchors matters.** The original prompt's "5–15
   words" cap and the schema's "(5–15 words)" field description were
   pulling the LLM toward bag-of-words. Removing both was load-bearing
   for the structured queries to emerge.

3. **Mechanical correctness still does not equal answer-quality
   movement.** Same lesson as the previous freeze sheet, now confirmed
   a second time. Bug #2 was structurally fixed, all queries are now
   syntactically valid, the queries reach better neighborhoods — and
   the supports rate did not move (0 / 40 vs 1 / 71). The bottleneck
   moved downstream rather than disappearing.

4. **The methodology held under load when it most needed to.** I had
   drafted a one-line Cochrane guidance tweak (extending K13 above)
   based on a single probe and was about to ship it. The user paused
   and asked: "are we overfitting?" The audit showed: yes — N = 1,
   alternative explanations not ruled out (K10 deduplication), no
   counterfactual test. The tweak was downgraded to "open hypothesis"
   instead of being shipped.

5. **The provider/query-formulator split has a clean architectural
   shape that's worth preserving.** The fix lives entirely in the data
   layer (provider self-description). The formulator, ranker, judge,
   and schedulers are unchanged. New providers self-describe their
   query language and the formulator picks it up automatically.
   `register_provider("name", Cls, description=..., query_guidance=...)`
   is a small surface that captures both axes a provider needs to
   advertise.

6. **CD012906 is not actually the canonical metformin/HbA1c review.**
   The previous freeze sheet hypothesised that CD012906 (Madsen et al.,
   "Metformin monotherapy for adults with type 2 diabetes mellitus")
   was the missing canonical review. Reading its abstract this session
   showed CD012906 actually focuses on **patient-important outcomes**
   (mortality, SAE, HRQoL, CVM, NFMI, NFS, ESRD) — not on HbA1c. The
   review's own conclusion: *"There is no clear evidence whether
   metformin monotherapy compared with no intervention, behaviour
   changing interventions or other glucose-lowering drugs influences
   patient-important outcomes."* When the freeze sheet said "CD012906
   wasn't found," the right reaction was not "fix the query to find
   it" but "fix the question to ask about a different review." This is
   a correction to the previous freeze sheet's diagnosis.

---

## Open question for the next session

**Before any of K10 / K11 / K12 / K13 ship, what does a stable,
multi-question benchmark look like?**

Probe B5 was a single end-to-end run. Probes B6 and B6b are
single-question slices. To make any further prompt or judge change
without overfitting, we need at least 3-4 different question shapes
exercising different evidence patterns:

- A clinical-RCT question with active comparators (metformin/HbA1c).
- A no-treatment-vs-treatment question (aspirin/MI in healthy adults
  — there's a saved DB at `~/.local/share/document-store/probe_b3_aspirin.db`).
- A mechanism question without a clear RCT base (e.g., podocyte
  motility under injury).
- A non-biomedical question (the topic-tools paper would be a fit, or
  any of the historical / physics example questions in
  PROVIDER_EXAMPLES).

The right next session might be "build the benchmark" rather than
"fix K10 / K11 / K12 / K13" individually. Without it, every prompt
change is at risk of repeating today's overfitting trap.

---

## How the next session opens

The first action is reading **all three** freeze sheets:

1. [`2026-05-03-freeze-sheet.md`](./2026-05-03-freeze-sheet.md) (lazy
   escalation + stage runners; K1–K5)
2. [`2026-05-03-evening-freeze-sheet.md`](./2026-05-03-evening-freeze-sheet.md)
   (K1–K7 fixed, K8 surfaced as load-bearing)
3. This sheet (Bug #2 fixed; K10–K13 surfaced; methodology held)

The session declares:

1. **Target** — most likely "build the multi-question benchmark" per
   the open question above. Or one of K10 / K11 if explicitly scoped to
   one and the user agrees the existing single-probe evidence is
   acceptable for that specific change.
2. **Frozen set** — explicitly. The Bug #2 fix is shipped and
   verified; touching it needs explicit reason. The previous freeze
   sheets remain in force.
3. **Observation mechanism** — for the work, what reads state without
   going through the code under test. The probe scripts in
   `/tmp/probes/B6/` are *not* observation primitives in the
   methodology sense — they exercise the changed code. New probes
   should be run and their outputs preserved alongside the saved DBs.

If a finding contradicts what's in any freeze sheet, that's a
separate findings file, not an in-flight edit.

---

## Saved DBs available as fixtures

(Same as previous freeze sheet; nothing new committed this session.)

| DB name | Question | Shape | Use for |
|---|---|---|---|
| `probe_b3_aspirin` | "Does daily low-dose aspirin prevent first myocardial infarction in healthy adults aged 50-70?" | retrieval_failed terminal | Reproducing the retrieval_failed path |
| `probe_b4_metformin` | metformin/HbA1c (pre-K7) | 1 artefact `insufficient`; mostly mislabelled adversarial | Comparing pre-K7 vs post-K7 adversarial |
| `probe_b5_metformin` | metformin/HbA1c (post-K7, pre-Bug-#2-fix) | 1 artefact `insufficient`; 1 / 76 supports on headline claim | The K8 baseline, used by this session for diagnosis. |

---

*Written 2026-05-04 to close the Bug #2 / K8(a) work and freeze
K10–K13 as the new entry conditions.*
