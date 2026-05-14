# Implementation plan — Audit report v2

**Status:** ready to execute
**Author:** Claude (with Timo)
**Date:** 2026-05-14
**Companion docs:**
  - `docs/superpowers/plans/2026-05-14-audit-report-prd.md` — the PRD (the *what* and *why*)
  - `/tmp/audit-report-v2-mockup.html` — the visual target (the *what it looks like*)

**Scope of code change:**
  - `src/andamentum/epistemic/audit_report.py` (renderer — primary)
  - `src/andamentum/epistemic/report_data.py` (data model — new fields)
  - `src/andamentum/epistemic/tests/test_audit_report.py` (tests — primary)
  - Whichever site builds `ReportData` from epistemic state — likely `epistemic/repository.py` or a sibling — for the new fields. Identify the exact site in Phase 0.

**Strict constraints (from Timo):**
  - **Hyper-minimalistic style preserved.** No new colors. No "tone" boxes. Verdict, gate-status, strength flags all rendered as neutral text. The label does the work, not pigment.
  - **No new typeset atoms.** No CSS module changes. Only the 7 atoms already in `andamentum.typeset` (heading, prose, callout, items, aside, card, reference).
  - **No new shared CSS classes.** Anything that would require a new class name is a smell — fold into existing atoms or render as inline italic / `<code>` pills which already have neutral styling.
  - **Hard cut, no backcompat shims.** Per project convention. v13 disappears.

---

## 1. Visual revisions vs the mock (recap)

The mock (`/tmp/audit-report-v2-mockup.html`) carries the right structure but uses some color and custom CSS that violate the minimalism rule. The implementation uses these substitutions, all of which use only existing typeset CSS:

1. **Verdict badge** → existing `.typeset-badge` default style (neutral beige `#f0ede8` / gray `#888`). Use `data-value` strings that don't match the green/red rules already in CSS (`supports`/`contradicts`/`pass`/`fail`/etc.). Use neutral values like `"refuted"`, `"inconclusive"`, `"supported"` (note: the existing CSS *does* map `data-value="supported"` to green — we'll switch to a value that falls through to neutral, e.g. `"completed"` or use no badge and put the verdict in body text).

   **Decision:** badge uses `data-value="verdict"` (falls through to neutral) and the verdict word is the badge text. One closed vocabulary at the renderer boundary: `Supported`, `Supported with refinement`, `Inconclusive`, `Refuted`, `Insufficient evidence`.

2. **Strength flags on evidence** → inline italic parenthetical at the end of each evidence-line judgement.
   `… reports lower mortality with HCQ after adjustment. *(observational, retrospective cohort, confounding-by-indication risk)*`
   No flag chip CSS. The flag taxonomy is a closed set held at the renderer boundary (Phase 3.4); each label is plain italic text.

3. **Gate-trace status** → plain text words in the table cell — `satisfied`, `failed`, `skipped`. No color, no symbol prefix. Reader parses the word.

4. **IBE candidates** → each rendered as its own `card` atom via `r.card(description, badge=status, id=...)`. `badge` carries the role label (`selected`, `runner-up`, `rejected`). Full description text, no truncation. No new CSS.

5. **Reproducibility footer** → existing `sidebar` atom. `.typeset-sidebar` already styles a three-column grid with title / row / label / value. Use it as-is.

6. **TOC jump list** → existing `aside` atom with markdown links inside. No new CSS.

7. **Posterior legend** → folded into the items-panel body of the "How confident are we?" row. Plain prose continuation. No new element.

8. **Headline / claim hierarchy** → h2 for top-level sections only; per-claim sub-sections render as `### …` embedded markdown headings inside a single `prose` atom per claim. The typeset CSS already styles embedded `<h3>` correctly (smaller serif, less prominent). No new heading machinery.

---

## 2. Data plumbing required (Phase 0)

These fields don't exist on the current `ReportData` / `ClaimSummary` types; v2 needs them. Source the values from existing repository state; do not have the renderer compute anything that should be computed upstream.

### 2.1 New fields on `ClaimSummary` (in `report_data.py`)

```python
class GateResult(BaseModel):
    name: str                       # "scrutiny", "convergence", "adversarial_balance", ...
    routing: Literal["PRIMARY", "SECONDARY", "SKIP"]
    required: str                   # human-readable requirement, e.g. ">= 2 independent sources"
    observed: str                   # human-readable observed value, e.g. "14 independent sources"
    status: Literal["satisfied", "failed", "skipped"]
    note: str | None = None         # optional one-sentence elaboration

class ClaimSummary(...):
    # existing fields kept
    gate_trace: list[GateResult] = []     # NEW
    verdict_label: str = ""               # NEW — closed-vocabulary label
                                          #   {Supported, Supported with refinement,
                                          #    Inconclusive, Refuted, Insufficient evidence}
```

Source: gates are decided in `epistemic/gates.py:STAGE_GATES` + `validate_promotion`. Build `GateResult` rows during report-data assembly by reading `claim` state against `STAGE_GATES[question_type]` and the observed values stored on the claim entity (posterior, convergence count, adversarial balance, scrutiny verdict).

`verdict_label` is derived from `claim.integrated_assessment` and the claim's posterior — one mapping function, used everywhere.

### 2.2 New fields on `ReportData`

```python
class ReportData(...):
    # existing fields kept
    snapshot_id: str | None = None        # NEW
    artefact_id: str | None = None        # NEW
    pipeline_version: str = ""            # NEW — andamentum package version
    pipeline_git_ref: str | None = None   # NEW — git short SHA if available
    reproduction_command: str = ""        # NEW — CLI invocation that re-runs this
```

Sources:
- `snapshot_id`, `artefact_id` already exist on the `Objective` entity (`andamentum.epistemic.entities`). Plumb through.
- `pipeline_version` from `importlib.metadata.version("andamentum")`.
- `pipeline_git_ref` — best-effort `subprocess.run(["git", "rev-parse", "--short", "HEAD"])` at report build time, or `None` if not in a git checkout. Cached in the data struct so re-renders are stable.
- `reproduction_command` constructed from the objective's mode (`verify` vs `ask`), the seed text, and the model id. Producer is wherever the CLI lives (`andamentum/cli/epistemic.py` or similar) — it knows the args it was invoked with.

### 2.3 Strength flags on evidence — defer the upstream extraction

The PRD calls for structured `study_design`, `weight`, `weakness` chips on each evidence item. Properly, these belong on the evidence judgement schema upstream. Two options:

- **2.3a (preferred, long term):** extend the evidence-judge agent output to include a structured `flags: list[Literal[...closed set...]]` field. Touches agent prompts and pydantic models.
- **2.3b (v2 ship):** keyword-extract from `judgment_reasoning` text at render time with a small closed dictionary (`{"observational", "RCT", "meta-analysis", "systematic review", "retrospective cohort", "single-arm", "combination intervention", "confounding-by-indication"}` — match case-insensitive substring). Cheap, no upstream change, good enough to demonstrate the feature.

**Decision for v2:** ship 2.3b in `audit_report.py` behind a `_extract_strength_flags(judgement_text) -> list[str]` helper. Mark with `# TODO: replace with upstream structured flags once the judge schema is extended`. The PRD's manuscript framing is the value driver; the extraction quality is a polish item that can iterate.

---

## 3. Phase 1 — Correctness fixes (single PR)

This phase lands the four Class-A correctness bugs from the PRD as one atomic change. Until they're fixed, no other phase makes the report more trustworthy.

### 3.1 Single source of truth for verdict label

**New module-level function in `audit_report.py`:**

```python
def _normalised_verdict(claim: ClaimSummary, posterior: float | None) -> str:
    """Closed-vocabulary verdict for the claim. Maps the raw
    integrated_assessment token + posterior into one of:

      "Supported", "Supported with refinement", "Inconclusive",
      "Refuted", "Insufficient evidence"

    Used everywhere the verdict appears: Q&A panel headline,
    Evidence-at-a-glance table, claim-card badge, gate-trace
    "decisive" row.
    """
```

Mapping logic:
- `claim.integrated_assessment in ("supports", "supports_refined")` and `posterior >= 0.70` → `"Supported"` or `"Supported with refinement"` (latter if `supports_refined`).
- `claim.integrated_assessment == "contradicts"` and `posterior <= 0.30` → `"Refuted"`.
- terminal_state != "completed" → `"Insufficient evidence"`.
- everything else → `"Inconclusive"`.

**Touchpoints to replace:**
- `_render_claim_section` — `card_kw["badge"] = claim.stage` → `card_kw["badge"] = _normalised_verdict(...)`. The `claim.stage`-driven badge is deleted.
- `_verdict_label` (existing helper, currently uses posterior thresholds directly) → call `_normalised_verdict` so the top-level and per-claim verdicts agree.
- Q&A panel "What did we find?" body, when no `data.verdict` is set, falls back to `_normalised_verdict`.

**Test (new):**
```python
def test_verdict_badge_matches_normalised_verdict():
    # Construct a ReportData where claim.integrated_assessment="contradicts"
    # and posterior=0.115 (the HCQ situation). Assert the card badge
    # text is "Refuted", NOT "supported" or "robust" or any stage label.
```

### 3.2 Single source of truth for evidence counts

The mismatch in v13 is that `data.stats.evidence_supports` (set upstream) and the per-claim `_evidence_for_claim` results (computed in the renderer) disagree by 2. v2 picks **the per-claim bucketer** as canonical and asserts agreement.

**Change in `audit_report.py`:**
- Compute counts in *one place* at the top of `build_audit_report`:
  ```python
  per_claim_buckets: dict[str, tuple[list, list, list]] = {
      c.claim_id: _evidence_for_claim(c, data.evidence) for c in data.claims
  }
  total_supports = sum(len(b[0]) for b in per_claim_buckets.values())
  total_contradicts = sum(len(b[1]) for b in per_claim_buckets.values())
  total_no_bearing = sum(len(b[2]) for b in per_claim_buckets.values())
  ```
- All count-rendering sites read from these locals. The `data.stats.evidence_supports` etc. fields are still computed upstream but the renderer **asserts** agreement; mismatch raises `AssertionError` (no silent failures, per project convention).

**Test (new):**
```python
def test_evidence_counts_consistent_across_sections():
    # Build a report with 11 supporting / 31 contradicting / 58 no-bearing.
    # Assert the Summary-of-findings table, the card details, and the
    # Evidence-at-a-glance claim row all report the same numbers.
```

### 3.3 Bucketing rejects "adversarial …" judgements from supporting

The v13 bug: items whose `judgment_reasoning` starts with `"Adversarial (statistical):"` or similar end up in the supporting list because `support_judgment="supports"` was set incorrectly upstream (the adversarial-search items leaked through the judge with the wrong label).

**Defensive change in `_evidence_for_claim`:**

```python
_ADVERSARIAL_PREFIX_RE = re.compile(r"^\s*adversarial\s*\(", re.IGNORECASE)

def _evidence_for_claim(claim, all_evidence):
    ...
    supports = [
        e for e in claim_evidence
        if e.support_judgment == "supports"
        and not _ADVERSARIAL_PREFIX_RE.match(e.judgment_reasoning or "")
    ]
    contradicts = [
        e for e in claim_evidence
        if e.support_judgment == "contradicts"
        or (
            e.support_judgment == "supports"
            and _ADVERSARIAL_PREFIX_RE.match(e.judgment_reasoning or "")
        )
    ]
    ...
```

This is a *renderer-side* defence; the proper fix is upstream so the judge doesn't mislabel them in the first place. Log a warning when the defence trips — that's the upstream signal.

**Test (new):**
```python
def test_adversarial_items_never_appear_in_supporting():
    # Construct an evidence item with support_judgment="supports" but
    # judgment_reasoning starting with "Adversarial (statistical):".
    # Assert it's bucketed as contradicting, not supporting.
```

### 3.4 Summary deduplication

Current `build_audit_report` opens the Summary section with:
```
**Research Question:** *...*
**Evidence Sources:** 33 | **Claims Established:** 1 of 1
<blockquote with the agent's narrative prefix that *also* says
"Research Question:" and "Evidence Sources:" — with a different count>
```

**Change:** drop both the prefix lines and any leading blockquote produced by `data.direct_answer`. The Research Question is already in the H1 + in the Q&A panel. The Evidence Sources count is in the Q&A panel + the Evidence-at-a-glance table. The Summary section becomes narrative-only — the direct answer prose without prefix or blockquote.

The prefix-stripping is a small helper:
```python
def _strip_summary_preamble(direct_answer: str) -> str:
    """Drop leading 'Research Question:' / 'Evidence Sources:' / blockquote
    lines that some agents prepend to their summary prose. The Summary
    section is for narrative answer text only — the metadata is already
    in the Q&A panel and the heading meta-line."""
    lines = direct_answer.split("\n")
    while lines and (
        lines[0].strip().startswith(("**Research Question:**", "**Evidence Sources:**"))
        or lines[0].strip().startswith(">")
        or not lines[0].strip()
    ):
        lines.pop(0)
    return "\n".join(lines).strip()
```

**Test (new):**
```python
def test_summary_section_strips_agent_preamble():
    # direct_answer starts with "**Research Question:** X\n**Evidence Sources:** 10\n> ...\n\nReal answer."
    # Assert the rendered Summary section starts with "Real answer." — and
    # contains no second "Evidence Sources" number anywhere.
```

### 3.5 Caveats and Limitations merged with system-level semantics

Today `Limitations` renders one bullet per non-resolved blocking uncertainty, which gets dumped from per-evidence judge text. Result: 19 bullets that are each a re-rephrasing of a contradicting evidence item.

**Change:** unify the section to a single `Caveats and limitations` h2 with these explicit bullet sources (in this order):

1. **Gate-trace anomalies** — for each gate with `status=failed`, emit a one-line system-level caveat: `"Scrutiny gate failed — the independent verifier disagreed with the per-evidence judge on N items; the claim was resolved by other gates."`
2. **System-acknowledged scope gaps** — currently `data.uncertainties` where `is_resolved=False`. Keep these.
3. **Evidence-cap-reached** — if `data.stats.total_evidence >= cap_threshold` (a known constant from the pipeline; if unavailable, omit).

Per-evidence contradicting prose is **not** included; that content already lives in the Contradicting evidence section.

The current `Caveats` / `Limitations` split (non-blocking vs blocking) is collapsed because the distinction wasn't legible to readers and the routing made the lower section noisier. If we want it back later we can add a third pass.

**Test (new):**
```python
def test_caveats_section_does_not_dump_per_evidence_judgements():
    # Build a report with 5 contradicting evidence items each with a
    # 20-word judgement. Assert the Caveats section contains *none* of
    # those 20-word strings, and instead contains system-level caveats
    # (gate-trace anomalies, scope gaps).
```

### 3.6 Phase 1 PR shape

One PR. Diff:
- `audit_report.py` — six helpers added, several call sites swapped, ~150 net new lines.
- `report_data.py` — new fields on ClaimSummary / ReportData (gate_trace, verdict_label, snapshot/artefact/version/repro-command). The Phase 1 PR can stub these (empty lists / empty strings) — they get populated upstream in Phase 2.
- `test_audit_report.py` — five new tests (3.1 / 3.2 / 3.3 / 3.4 / 3.5).
- Reference-output fixtures (if any are pinned) regenerated.

Acceptance:
- pyright clean (modulo the pre-existing 23-error baseline noted in CLAUDE.md).
- ruff clean.
- All existing tests still pass, no behaviour regression on the v13 reference inputs (verdict labels and counts may *change* — they were wrong — but the diff should be small and explicable).
- HCQ rendering: no green badge, counts agree everywhere, supporting list contains no adversarial items, Caveats section contains 1–4 system-level bullets only.

---

## 4. Phase 2 — Schneider-aligned epistemic transparency

This phase adds the three pieces that turn the report into a credible answer to Schneider (2025): the **gate trace**, the **reproducibility footer**, and the **directional posterior legend**.

### 4.1 Gate trace upstream plumbing

Identify the call site that builds `ReportData`. For each claim, populate `gate_trace: list[GateResult]`:

```python
def _build_gate_trace(claim, question_type) -> list[GateResult]:
    """Read STAGE_GATES[question_type] for which gates are PRIMARY /
    SECONDARY / SKIP for this question type, then evaluate each against
    the claim's recorded values."""
    routing = STAGE_GATES.get(question_type, {})
    trace = []
    for gate_name in ALL_GATES_ORDERED:
        route = routing.get(gate_name, "SKIP")
        if route == "SKIP":
            trace.append(GateResult(
                name=gate_name, routing="SKIP",
                required="n/a — not applicable for this question type",
                observed="—", status="skipped",
            ))
            continue
        required, observed, status, note = _evaluate_gate(claim, gate_name)
        trace.append(GateResult(
            name=gate_name, routing=route,
            required=required, observed=observed, status=status, note=note,
        ))
    return trace
```

`_evaluate_gate` is a switch on gate name that reads the appropriate field from the claim entity. Each gate's required threshold is a constant pulled from `gates.py` so the table mirrors the source of truth.

### 4.2 Gate trace rendering

In `audit_report.py`, inside the per-claim block (after the audit-trail intro paragraph, before IBE candidates):

```python
def _render_gate_trace(trace: list[GateResult]) -> str:
    """Markdown table — Gate | Required | Observed | Status | (Note)."""
    rows = []
    for g in trace:
        gate_label = _humanise_gate_name(g.name)  # "scrutiny" -> "Scrutiny"
        if g.note:
            gate_label = f"{gate_label} *({g.note})*"
        rows.append([gate_label, g.required, g.observed, g.status])
    return _md_table(
        ["Gate", "Required", "Observed", "Status"],
        rows,
    )
```

Status values are plain words: `satisfied`, `failed`, `skipped`. No icons, no color.

Rendered inside the per-claim prose block as a sub-section under `### How the system reasoned about this claim`, between the investigation-rounds list and the IBE candidate cards.

**Tests (new):**
```python
def test_gate_trace_renders_all_gates_with_routing():
    # Construct a claim with a full gate_trace including PRIMARY,
    # SECONDARY, and SKIP routed gates. Assert all appear in the table
    # and skipped ones show "—" observed and "skipped" status.

def test_gate_trace_status_words_only_no_symbols():
    # Render. Assert the gate-status cells contain "satisfied" /
    # "failed" / "skipped" and not "✓" / "✗" or any color span.

def test_gate_trace_for_question_type_routing_respected():
    # For a question_type that routes "deductive_validation" to SKIP,
    # assert that gate's row reads "skipped" with required="n/a".
```

### 4.3 Reproducibility footer

Append to the bottom of `build_audit_report`:

```python
# ── Reproducibility footer (sidebar atom) ───────────────────────────
r.sidebar(groups=[
    {
        "title": "Pipeline",
        "rows": [
            {"label": "version", "value": data.pipeline_version or "unknown"},
            {"label": "git ref", "value": data.pipeline_git_ref or "—"},
            {"label": "cycles used", "value": _cycles_summary(data)},
        ],
    },
    {
        "title": "Model",
        "rows": [
            {"label": "primary", "value": data.model_used},
            {"label": "date", "value": data.investigation_date.isoformat()},
        ],
    },
    {
        "title": "Persistence",
        "rows": [
            {"label": "snapshot", "value": (data.snapshot_id or "—")[:12] + "…"},
            {"label": "artefact", "value": (data.artefact_id or "—")[:12] + "…"},
        ],
    },
])
if data.reproduction_command:
    r.aside(f"```\n{data.reproduction_command}\n```")
```

**Note:** check the actual `Report.sidebar()` signature before writing this. If `sidebar` doesn't accept that shape, the existing `items` atom with `variant=pairs` can substitute — same neutral palette, same fonts.

**Tests (new):**
```python
def test_repro_footer_contains_snapshot_and_reproduction_command():
    # Build a report with snapshot_id="abc-123", reproduction_command="andamentum-epistemic ...".
    # Assert both appear in the rendered atoms list.

def test_repro_footer_handles_missing_git_ref():
    # pipeline_git_ref=None → row reads "—", not "None" or empty string.
```

### 4.4 Posterior with directional legend

In the "How confident are we?" Q&A row body:

```python
def _confidence_body(data: ReportData) -> str:
    cs = data.confidence_scores
    if cs is None or cs.posterior is None:
        return "No posterior computed."

    p = cs.posterior
    verdict = _normalised_verdict(...)  # same function as elsewhere

    lines = [
        f"Probability the claim is true: **{p:.3f}** · verdict {verdict}",
        "",
        "_Decisive bands: posterior ≥ 0.80 → Supported · ≤ 0.20 → Refuted · otherwise Inconclusive._",
    ]
    return "\n".join(lines)
```

Notes:
- The legend renders inline; no new CSS. Italic via `_..._` markdown.
- Bold verdict label uses the closed vocabulary from `_normalised_verdict`.

**Test (new):**
```python
def test_confidence_body_contains_directional_phrasing_and_legend():
    # posterior=0.115. Assert body contains both "Probability the claim is true"
    # and "Decisive bands" and the verdict label "Refuted".
```

---

## 5. Phase 3 — Naming, structure, and the closed verdict vocabulary

### 5.1 Renames (one-liners)

- `Detailed analysis` → `Reasoning trace`
- `Strongest supporting evidence (N total)` → `Supporting evidence (N items)`
- `Adversarial probe (N challenges)` → `Contradicting evidence (N items)`
- `Summary of findings` → `Evidence at a glance`
- Card collapsible label `Details` → remove the collapsible entirely; inline the scope + verdict + counts beneath the claim statement (it's already 3 lines, the disclosure adds friction).

### 5.2 Hierarchy fix — per-claim sub-sections become h3

The flat-h2 problem in v13: per-claim sub-sections compete visually with top-level document sections. Fix:

- The h2 "Reasoning trace" section opens with intro prose only.
- Then for each claim, render **one** big prose atom whose body starts with `### Claim N — {claim statement}`, contains sub-sub-sections as `#### How the system reasoned about this claim`, `#### Supporting evidence (N)`, `#### Contradicting evidence (N)`, and includes the gate trace + IBE cards inline.

Wait — `####` is h4. Let's choose carefully:

```
H1  (heading atom)            research question
H2  Answer in brief           items atom (Q&A panel)
H2  Summary                   prose atom
H2  Evidence at a glance      prose atom
H2  Reasoning trace           prose atom (intro)
    H3  Claim 1 — <text>      embedded in next prose atom
        Card                  card atom (claim statement + verdict badge)
        H4  How the system reasoned    embedded h4 in prose
        H4  Supporting evidence (N)    embedded h4 in prose
        H4  Contradicting evidence (N) embedded h4 in prose
H2  Caveats and limitations   prose atom
H2  Appendix                  card atom with details=
H2  (sidebar)                 reproducibility footer
```

The per-claim breakdown is one prose atom whose body uses `###` and `####` markdown headings. `.typeset-prose h3` and `.typeset-prose h4` both have existing CSS styling. The claim **card** (with the verdict badge) is its own atom, placed *between* the `### Claim N` heading prose atom and the `#### How the system reasoned …` content. This is the cleanest mapping that respects "no typeset module changes".

**Inline anchors** for jump links: at the top of each `###` or `####` heading, insert `<a id="…"></a>` (HTML inline in markdown is supported by the typeset markdown renderer per the existing v13 use of inline `<details>`).

### 5.3 Verdict vocabulary at the renderer boundary

`_normalised_verdict` (added in Phase 1) defines the closed set. All consumer call sites use it. Raw tokens (`supports_refined`, etc.) appear only in the appendix's gate-trace JSON.

### 5.4 Provider human-readable name

Today: `<code>europepmc</code>`. v2: `<code>Europe PMC</code>` — same CSS pill, but the *display* name. Lookup table in `audit_report.py`:

```python
_PROVIDER_DISPLAY_NAMES = {
    "europepmc": "Europe PMC",
    "pubmed": "PubMed",
    "openalex": "OpenAlex",
    "web_search": "Web search",
    "clinicaltrials": "ClinicalTrials.gov",
}

def _provider_label(source_type: str) -> str:
    return _PROVIDER_DISPLAY_NAMES.get(source_type, source_type)
```

Used in `_evidence_line` only. No CSS change.

---

## 6. Phase 4 — Polish (smallest scope)

These are PRD Class-D items. None of them block Phase 1–3. Do them as separate small PRs only if Phase 1–3 land cleanly.

### 6.1 IBE candidates as stacked card atoms

Replace the current single markdown table with one `r.card(...)` call per candidate:

```python
for c in claim.ibe_candidates:
    status = "selected" if c.chosen else ("runner-up" if c.runner_up else "rejected")
    head = (
        f"**Candidate {c.candidate_id}** — verdict {c.verdict} · "
        f"loveliness {c.loveliness:.2f} · likeliness {c.likeliness:.2f}"
    )
    r.card(
        head + "\n\n" + c.description,
        badge=status,
        id=f"ibe-{c.candidate_id}-{_claim_slug(claim.claim_id)}",
    )
```

Full text, no truncation. The `selected` card is distinguished by the badge text only.

### 6.2 TOC

At the top, after the heading:

```python
r.aside(
    "**Jump to:** "
    "[Answer in brief](#answer) · "
    "[Summary](#summary) · "
    "[Evidence at a glance](#evidence-glance) · "
    "[Reasoning trace](#reasoning) · "
    "[Caveats](#caveats) · "
    "[Appendix](#appendix)"
)
```

Single `aside` atom. No new CSS.

### 6.3 Strip judge's internal reasoning from inline support/contra bullets

Currently some `judgment_reasoning` strings include verbose meta-prose ("...the credibility depends on the ability to verify the cited eFigure…"). Helper:

```python
def _one_sentence_judgement(text: str, max_chars: int = 200) -> str:
    """Trim a judgement string to its first complete sentence, capped at
    max_chars. The fuller text remains in the appendix."""
```

Used in `_evidence_line` for the inline support / contradict bullets. The appendix bullets keep the full text via `_sanitize_excerpt(max_chars=600)`.

### 6.4 Strength-flag extraction (the deferred 2.3b)

```python
_FLAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "RCT": ("randomized controlled trial", "RCT ", "randomised trial", "randomized trial"),
    "meta-analysis": ("meta-analysis", "meta analysis"),
    "systematic review": ("systematic review",),
    "observational": ("observational", "retrospective cohort", "case-control"),
    "single-arm": ("single-arm", "single arm", "no control group", "n=1 arm"),
    "combination intervention": ("hcq + azithromycin", "with azithromycin", "combination therapy"),
    "confounding-by-indication": ("confounding by indication", "treatment-selection bias"),
}

def _extract_strength_flags(judgement: str) -> list[str]:
    if not judgement:
        return []
    text = judgement.lower()
    return [flag for flag, kws in _FLAG_KEYWORDS.items() if any(k in text for k in kws)]
```

Inline in `_evidence_line`:

```python
flags = _extract_strength_flags(ev.judgment_reasoning)
flag_str = f" *({', '.join(flags)})*" if flags else ""
return f"{head} — {sentence}{flag_str}"
```

Italic parenthetical. No CSS.

### 6.5 Machine-readable gate-trace JSON appendix block

```python
import json
gate_json = json.dumps({
    "claim_id": claim.claim_id,
    "verdict": _normalised_verdict(...),
    "posterior": claim.posterior,
    "gates": [g.model_dump() for g in claim.gate_trace],
    "evidence_counts": {...},
}, indent=2)

r.card(
    "**Gate-trace JSON (machine-readable)** — for downstream evaluation tooling.",
    badge="machine-readable",
    id=f"gate-json-{_claim_slug(claim.claim_id)}",
    details=f"```json\n{gate_json}\n```",
)
```

No new CSS — uses existing `<pre><code>` rendering for fenced code in card details.

---

## 7. Testing strategy

### 7.1 Unit tests per phase

Each phase ships with the tests listed above. The full v2 test count grows by roughly 12 tests on top of the existing 32.

### 7.2 Reference reports kept under tests

The two reference fixtures (HCQ verify-mode, statins verify-mode) should be regenerated and checked-in as goldens for snapshot diffing. Suggested location: `src/andamentum/epistemic/tests/fixtures/audit_v2_*.html`. Tests assert structural properties (e.g. "the Reasoning trace section contains a `### Claim 1` heading", "no element has `style="color: rgb(231,...)"`") rather than full string equality — that way deterministic prose changes don't break the suite.

A grep test enforces the no-colors rule:
```python
def test_rendered_html_contains_no_red_or_green_inline_styles():
    html = render(build_audit_report(reference_data))
    assert "color: #1a7a3a" not in html  # the green
    assert "color: #b91c1c" not in html  # the red
    assert "background: #eef7f0" not in html
    assert "background: #fef2f2" not in html
```

This is a defence against future drift — if someone reintroduces tone-CSS via a new typeset variant, this test catches it.

### 7.3 Manual visual review at each phase

After each phase, regenerate both reference reports (`/tmp/test2_hcq-audit-v14.html`, `/tmp/test3_statins-audit-v14.html`) and open them. The mock-up file (`/tmp/audit-report-v2-mockup.html`) is the visual target *minus* color/CSS additions; the regenerated v14 should look slightly more austere than the mock but structurally identical.

---

## 8. Implementation order and PR shape

| PR | Phase | Includes | Acceptance gate |
|---|---|---|---|
| 1 | Phase 0 (data) | New fields on ClaimSummary / ReportData, stubbed empty | pyright + tests pass, nothing renders yet |
| 2 | Phase 1 (correctness) | 3.1–3.5 as one atomic change | both reference reports look correct: no lying badge, no count mismatch, no adversarial-in-supports, Caveats is system-level |
| 3 | Phase 2 (Schneider) | Gate trace, repro footer, posterior legend | gate trace visible in HCQ report; clicking the snapshot ID is hashable; posterior is read directionally by an unprimed reader |
| 4 | Phase 3 (naming + hierarchy) | 5.1–5.4 | section names match PRD; per-claim sub-sections nest visibly as h3/h4 |
| 5 | Phase 4 (polish) | IBE cards, TOC, judgement trimming, strength flags, JSON appendix | belt-and-braces — any of these can ship individually |

Total scope: ~5 PRs, two of which (1 and 4) are small. The bulk of work is in PR 2 (correctness) and PR 3 (Schneider answers).

---

## 9. Risks and mitigations

- **R1: Upstream data may not exist for some gate values.** Some claims may not have all gate values recorded (e.g. older entities). Mitigation: `_build_gate_trace` defaults `observed="—"` and `status="skipped"` for any gate where no record exists, with a `note="no value recorded"`.
- **R2: `Report.sidebar()` may not have a `groups=` API.** Verify before Phase 2.3. If not, use `items(entries=...)` three times in a row. Same neutral palette.
- **R3: Strength-flag keyword extraction is brittle.** Phrasing varies across judges. Mitigation: keep the dictionary small (5–8 entries) and ship as visible-progress polish, not a load-bearing trust signal. The flag's *absence* should never be read as "this is strong evidence". Document this in the section header.
- **R4: Closed verdict vocabulary may not cover all states.** If a claim has terminal_state like `oscillation_detected` or `cycle_capped`, `_normalised_verdict` maps it to `"Insufficient evidence"`. The reason for insufficiency is then surfaced in the Caveats section so the reader knows why.
- **R5: The h3/h4 embedded heading approach degrades for very-long reports** (e.g. research mode with 8 sub-claims). Mitigation: in research mode, the per-claim breakdown becomes one prose atom per claim instead of one combined atom, so each claim has its own anchor. TOC lists all claim anchors.

---

## 10. Definition of done

A v2 report is shippable when:

1. `pyright`, `ruff check`, `ruff format`, `pytest` all pass on the canonical green-state baseline.
2. The HCQ reference report:
   - Verdict badge says **Refuted** (not "supported").
   - Supporting / contradicting / no-bearing counts agree across all sections.
   - No adversarial item appears in the supporting list.
   - Gate trace table appears, with `scrutiny: failed` and `posterior: decisive (refute)` visible.
   - Caveats section has 1–4 system-level bullets (not 19 per-evidence dumps).
   - Reproducibility footer at the bottom carries snapshot ID, model, pipeline version, and the literal re-run CLI command.
3. The statins reference report:
   - Verdict badge says **Supported with refinement**.
   - Same structural properties as HCQ (just different counts and different gate-trace values).
4. No new CSS in `typeset/`. No new typeset atoms registered. The implementation is purely additive in `audit_report.py` + minimally additive in `report_data.py`.
5. The "no colors" grep test passes — no green/red CSS values appear in the rendered HTML.
6. The PRD's §8 acceptance criteria are met.

---

## 11. Open questions before starting

1. **Where exactly is `ReportData` constructed?** I need to confirm the call site for plumbing snapshot_id / artefact_id / pipeline_version / reproduction_command. Will identify in Phase 0.
2. **Is there a registered `gates.STAGE_GATES` keyed by question type, with the routing literal `PRIMARY/SECONDARY/SKIP`?** Or is the routing implicit? Will read `epistemic/gates.py` in Phase 2.1 to confirm.
3. **`Report.sidebar()` API shape.** Will read `andamentum/typeset/` in Phase 2.3 to confirm before writing the footer.
4. **Should the Phase 1 PR also drop the "Open questions" section?** v13 had it; the v2 PRD doesn't mention it explicitly. Recommendation: keep it as a separate h2 between Caveats and Appendix, but only when `data.open_questions` is non-empty. No structural change.

These are pre-flight checks, not blockers. Answer is "go read the files" for all four.
