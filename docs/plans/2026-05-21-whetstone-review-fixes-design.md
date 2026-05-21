# Whetstone reviewed-document failures — Design options (Phase 2)

**Status:** Draft. User picks options before Phase 3 (implementation plan).
**Companion to:** `2026-05-21-whetstone-review-fixes-prd.md`
**Branch:** `whetstone-review-fixes`

This document answers the four open questions from PRD §7 with concrete
options, trade-offs, and a recommendation for each. No code yet. The
user's picks here become the contract for the Phase 3 implementation
plan.

---

## Q1 — P1 namespace fix (comments invisible in Word)

**The bug:** `low_level.py:479` creates the `<comments>` root element
without an `nsmap`, so lxml serialises it under `ns0:` and Word ignores
every comment.

### Option 1A — Minimal: bind nsmap on the one root

```python
root = etree.Element(f"{{{NS['w']}}}comments", nsmap=NS)
```

- **Pros:** One line. Fixes the confirmed bug. Lowest risk.
- **Cons:** Leaves the other ~10 root-construction sites unguarded. If
  a future change creates another new XML part the same way, the bug
  silently returns. Doesn't satisfy the user's "perhaps we need a more
  substantive fix" instinct.

### Option 1B — Helper + audit: centralise root creation (recommended)

Introduce one helper in the docx layer:

```python
def w_root(local_name: str) -> etree._Element:
    """Create a WordprocessingML root element with the standard `w:`
    prefix bound. Use for any element that becomes the root of an
    independent XML part (comments.xml, etc.)."""
    return etree.Element(f"{{{NS['w']}}}{local_name}", nsmap=NS)
```

Route `low_level.py:479` (and any other new-part roots the audit finds)
through it. Subelements are unaffected — they inherit the parent's nsmap.

- **Pros:** Fixes the bug AND makes the correct pattern the easy
  default. An audit of the ~10 root sites is bounded work. Self-
  documenting.
- **Cons:** Slightly more surface than 1A. Need to verify which of the
  96 construction sites are genuinely roots vs subelements (most are
  subelements and need no change).

### Option 1C — Render-time validation guard

On top of 1A/1B, add a validation pass that runs after the .docx is
written: re-open the zip, parse each XML part, and raise if any OOXML
element uses a non-`w:` prefix for the wordprocessingml namespace.

- **Pros:** Catches the bug class at runtime, not just in tests. A
  malformed docx never reaches the user — it fails loudly instead.
- **Cons:** Adds a re-parse of the output on every render (small cost
  on large docs). Arguably belongs in tests, not the hot path.

### Recommendation for Q1

**1B + a regression test** (the test belongs to PRD §6.1 anyway).
1B fixes the root cause and makes recurrence hard; the test catches
recurrence definitively. Hold 1C (runtime guard) in reserve — propose
it only if we decide the docx layer is fragile enough to warrant a
belt-and-braces runtime check. My instinct: 1B + test is the
"substantive fix" the user asked for without over-engineering.

---

## Q2 — P2 mechanism (lens findings about absence)

**The problem:** the lens emits findings that are true about the chunk
but false about the document — "this section has only a citation list,
no claims" / "no Discussion section". Two root causes:

- (a) The lens feels pressure to "find issues" even when a section has
  nothing substantive to assess (bookkeeping content like a reference
  list).
- (b) The lens reasons about what's *missing from the document* while
  only seeing one section, so it flags absences that exist elsewhere.

The PRD constrains the fix: domain-agnostic (no section-type labels),
small-LLM-friendly, and the lens must stay *silent* when it has nothing
to say.

### Option 2A — Prompt change only (recommended core)

Two additions to the per-section lens prompt
(`critical_read.py:_run_lens`, the `prompt` string):

1. **Permission to return nothing.** Small models over-produce when
   told to "emit your issues". Add: *"If this section contains no
   substantive content to assess — for example it is a list of
   references, a table of identifiers, or boilerplate — return an empty
   list. Finding nothing is the correct and expected answer for such
   sections. Do not invent issues to fill the list."*

2. **Scope discipline.** Add: *"You are reviewing ONE section of a
   larger document. Comment only on what is present in this section.
   Do NOT flag things as missing from the document as a whole (a
   missing methods section, missing references, no discussion) — you
   cannot see the other sections, and they may well contain what you
   think is absent."*

- **Pros:** Fully domain-agnostic — no labels, no section classifier.
  Matches the "silent when nothing to say" rule directly. Zero new
  tokens beyond the two instructions. Works for any document type.
- **Cons:** Relies on small-LLM compliance. Some models will still
  occasionally emit an absence-finding or a "nothing here" finding.
  Needs a safety net (2B).

### Option 2B — Post-filter on absence-shaped findings (recommended net)

A small, domain-agnostic filter applied to lens output: drop a finding
when the finding's *own text* admits there's nothing to assess. The
agent, when it does misbehave, tends to say so explicitly — "there is
nothing to assess", "this section contains only references", "no
claims are made here". These are self-identifying non-findings.

The filter matches a handful of absence phrases (case-insensitive
substring) in the title/rationale and drops those findings. This is NOT
a section-type classifier — it's a "the model told us this isn't a real
finding" filter.

- **Pros:** Catches the residue that slips past the prompt. Domain-
  agnostic (matches the *model's admission*, not the section type).
  Cheap, deterministic, testable.
- **Cons:** Phrase-matching is inherently approximate; could in theory
  drop a legitimate finding that happens to contain "nothing to
  assess". Mitigated by keeping the phrase list tight and specific.

### Option 2C — Heuristic input skip (NOT recommended)

Skip the lens read on chunks that look like bookkeeping (citation
density, short paragraphs, no verbs) without labelling them.

- **Pros:** Saves the LLM call entirely on skippable sections.
- **Cons:** This is the approach we already rejected once. Even
  "unlabelled" heuristics are tuned from the academic case (citation
  density is an academic signal). Risk of skipping a real content
  section that happens to be citation-dense. Adds input-side complexity
  for a problem better solved at the output. **Recommend against.**

### Option 2D — Document-map preamble (deferred per PRD §5)

Give each lens call a short list of all section titles so it knows what
exists elsewhere. Directly addresses root cause (b).

- **Pros:** The "right" long-term fix for absence-reasoning.
- **Cons:** PRD §5 deferred it; adds ~100-200 tokens per call; more
  invasive. **Recommend keeping deferred** — revisit only if 2A's
  scope-discipline instruction proves insufficient in testing.

### Recommendation for Q2

**2A (prompt change) + 2B (post-filter safety net).** The prompt change
is the primary, principled fix — it's domain-agnostic and encodes
exactly the "silent when nothing to say" behaviour the user wants. The
post-filter is a cheap, deterministic backstop for small-model
non-compliance. Both avoid any section-type label vocabulary. Keep 2C
rejected and 2D deferred.

---

## Q3 — P3 (count vs rendered-comment mismatch)

**The issue:** the header counts 167 findings but only ~137 become
comments (findings without anchors are dropped). After P1 lands the
visible comments jump from 0 to ~137, but the 167 ≠ 137 mismatch
remains.

### Option 3A — Count only renderable findings in the header

Change the header integer to count findings that actually produce a
comment (those with an anchor) plus edits.

- **Pros:** The number matches what the user sees. Simple.
- **Cons:** Hides the existence of un-anchorable findings entirely. A
  finding the agent genuinely made (e.g. "the paper never states its
  sample size") vanishes from the report.

### Option 3B — List un-anchored findings under "Critical Issues"

Instead of "See detailed reviews below", when there are findings
without anchors, list them as text under the Critical Issues heading
(they can't be Word comments — they have no anchor — so they live in
the report header instead).

- **Pros:** No finding is lost. Anchored findings → comments;
  un-anchored findings → listed in the report. The "See detailed
  reviews below" dead-end disappears.
- **Cons:** More renderer work. Needs a sensible format for the listed
  findings.

### Option 3C — Both

Count accurately AND list the un-anchored findings.

### Recommendation for Q3

**3B.** It's the honest fix: every finding the agent made is visible
*somewhere* (comment if anchorable, report-list if not), and the
misleading "See detailed reviews below" with nothing below is replaced
by actual content. This also interacts well with Q2 — once absence-
findings are filtered out (2A/2B), most of the un-anchorable residue
disappears, so the report-list will usually be short. Defer 3A
(recounting) as unnecessary once 3B lands.

**Open sub-question:** is P3 in scope for this branch at all, or a
follow-up? It's lower priority than P1+P2. I lean toward including 3B
because the "See detailed reviews below → nothing" text is part of the
same user-visible confusion, but I'll defer to the user.

---

## Q4 — Patch-application visibility (CLI logging)

Surface `render_docx`'s `PatchApplicationResult` (applied / failed
counts) in the CLI output so the user knows whether comments anchored.

### Recommendation for Q4

**Include, as its own clearly-scoped commit.** This is genuinely useful
(it's how a user would have diagnosed P1 themselves), it's small, and
it's logging-only. But per the user's scope-expansion concern, it ships
as a separate commit with its own line in the implementation plan — not
bundled into P1 or P2. The user can drop it from the plan if they
disagree.

---

## Summary of recommendations

| Question | Recommendation |
|---|---|
| Q1 — namespace fix | **1B** (nsmap helper + audit roots) + regression test |
| Q2 — absence findings | **2A** (prompt: permission-to-be-silent + scope discipline) + **2B** (post-filter safety net). Reject 2C, defer 2D. |
| Q3 — count mismatch | **3B** (list un-anchored findings in report; kill "see below" dead-end). User decides if in scope this branch. |
| Q4 — patch visibility | **Include as a separate commit.** |

## What this produces, end to end

With Q1+Q2 (the must-haves):
- Every anchorable finding renders as a visible Word comment.
- The lens stops emitting absence-findings and "nothing here" noise —
  on References and on every other section, for any document type.
- A regression test prevents the namespace bug from returning.
- No section-type labels anywhere; no env vars; no input-side skipping.

With Q3+Q4 (the should-haves, user's call):
- Un-anchorable findings are visible in the report instead of vanishing.
- The user can see applied/failed comment counts in the CLI.

## Approval gate

Pick one option per question (or override with your own). The chosen
set becomes the Phase 3 implementation plan: sequenced commits, each
with explicit scope and acceptance criteria, for your sign-off before
any code is written.
