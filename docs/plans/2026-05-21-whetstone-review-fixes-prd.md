# Whetstone reviewed-document failures — PRD

**Status:** Draft. User review required before Phase 2 (design options).
**Branch:** `whetstone-review-fixes`
**Phase 0 (diagnosis):** complete.
**Author:** Claude + user discussion 2026-05-21.

---

## 1. Background

The user ran `andamentum-whetstone` on a real academic manuscript with a
cloud LLM and observed two distinct failure modes in the reviewed.docx
output:

1. The report header says `Critical Issues: 167 / See detailed reviews below`,
   but no comments appear anywhere in the document. Word's review pane is
   empty.
2. Some of the LLM-emitted findings, when they do reach the comment layer,
   say things like "this section has only a citation list — no claims or
   evidence" — a true observation about a single References chunk, but a
   false positive at the document level (the rest of the manuscript does
   have claims; the agent just didn't see them).

Failure 1 is the most user-visible: 167 issues reported, zero shown. Failure
2 is a wider architectural problem that surfaces in many flavours
("no Discussion section!" when looking at the Methods chunk, etc.).

## 2. Phase 0 — what is actually broken (diagnosis)

### 2.1 Comments are in the .docx file. Word can't see them.

Opening the reviewed.docx as a zip reveals:

| XML part | Prefix used | Items |
|---|---|---|
| `word/comments.xml` | `ns0:` | 137 `<ns0:comment>` elements (full text present) |
| `word/document.xml` | `w:` | 137 matching `<w:commentRangeStart>` / `<w:commentRangeEnd>` / `<w:commentReference>` anchors |

Both parts reference the same OOXML namespace URI
(`http://schemas.openxmlformats.org/wordprocessingml/2006/main`). Strict
XML treats `ns0:comment` and `w:comment` as equivalent. **Microsoft Word
does not.** Word looks for the `w:` prefix specifically on the comments
part, silently ignoring elements declared under any other prefix.

**Root cause:** `src/andamentum/whetstone/docx/low_level.py:479`:

```python
root = etree.Element(f"{{{NS['w']}}}comments")
tree = etree.ElementTree(root)
```

The element is created with the wordprocessingml namespace URI but
**without an `nsmap=` argument**. When lxml serialises this element, it
must invent a prefix for the URI — it picks `ns0`. Every comment appended
under this root inherits `ns0:` on serialisation.

This bug fires **only when whetstone creates a brand-new `comments.xml`**
— i.e., when adding comments to a .docx that previously had none. If the
source .docx already had a comments part (because Word had been used to
review it before), whetstone reuses the existing tree (which has the
correct `w:` prefix) and the bug doesn't trigger. That likely explains
why the user reports having seen "similar issues with different Word
versions" — older sources with no prior comment activity hit this; ones
already carrying comments do not.

### 2.2 Wider surface area: 96 element constructions, 0 use `nsmap=`

A grep across `src/andamentum/whetstone/docx/` finds 96 `etree.Element` /
`SubElement` / `ElementTree` construction sites and **zero** of them use
the `nsmap=` argument. Most of those are SubElements (which inherit
their parent's nsmap on serialisation, so they're fine in practice). But
several detached Element constructions exist that could become roots:

- `low_level.py:479` — `comments` root (the confirmed bug)
- `low_level.py:395` — `Override` element for `[Content_Types].xml`
  (different namespace; haven't verified whether this hits a similar
  fault, but worth checking)
- `xml_builder.py:47, 78, 165, 173, 174, 193, 214, 227` — `change`
  containers, `r`, comment-range markers, individual comment elements,
  break runs, paragraphs. These are all subelements in practice but
  none of them is *guaranteed* to be attached to a parent before
  serialisation.

So the bug is one identified call site, but the surrounding code has no
guard rails preventing it from recurring elsewhere.

### 2.3 The 167-vs-137 gap

The report's "Critical Issues: 167" counts `result.findings +
result.deterministic_findings`. The actual number of comments in the
file is 137. The 30-comment gap is explained by the docx adapter
`_finding_to_patch` (`renderers/docx.py:181`):

```python
def _finding_to_patch(finding, DocumentPatch):
    if not finding.quotes:
        return None
    ...
```

Findings with no quote anchor (typically LLM-emitted "no methodology
section" / "missing X" style findings, which describe an absence rather
than a specific span) are silently dropped from the comment layer. They
**are** counted in the header (because the header counts findings, not
patches), but never become comments.

This is a smaller bug than 2.1, but it contributes to the user's
"reported 167, see ~0" experience — the gap between count and reality.

### 2.4 The "See detailed reviews below" header text

`docx/finalization.py:227` writes the literal string `"See detailed
reviews below."` whenever `critical_issues=None` is passed to
`_format_critical_issues`. The caller in
`renderers/docx.py:128-140` never passes a `critical_issues=` argument,
so this default fires unconditionally. The intent appears to be: "the
detailed reviews are the comments in the body" — but to a user, "below"
naturally means "lower down in this section of the report." When the
comments don't render (failure 2.1), there is genuinely nothing below.

### 2.5 What is *not* broken (ruling out earlier guesses)

- Chunking is fine; the chunker correctly identified the References
  section (the citation graph has correct `references_section_ids`).
- The proofread integration is producing findings with valid section-
  local anchors. Those would render correctly *if* failure 2.1 were
  fixed.
- The anchor-narrowing logic in the recent proofread commit is working
  as designed.
- The "no claims here" LLM findings (failure 2 in §1) are a real but
  distinct problem from why no comments render at all.

## 3. User-visible problem statement (verified)

Two independent problems, in priority order:

| # | Severity | Visible symptom |
|---|---|---|
| **P1** | Critical | "Critical Issues: N" reported in the docx header, but no comments appear in the body. Word's review pane is empty. |
| **P2** | Important | When comments do appear, some make false claims that depend on the agent not having seen the rest of the document ("no Discussion section", "this section has only a citation list"). |

P1 hides every other improvement we make: any fix to P2 is invisible to
the user until P1 is resolved. **P1 must be fixed first.**

A smaller, related issue worth flagging once but not separately tracking:

| # | Severity | Visible symptom |
|---|---|---|
| **P3** | Nuisance | Report header says "Critical Issues: 167 / See detailed reviews below" but the integer counts findings that will never become comments. The user sees a mismatch between the count and the visible output. |

P3 is mostly resolved once P1 lands (most findings will render) and the
remaining gap can be addressed by either (a) counting only renderable
findings, or (b) listing the un-anchorable findings explicitly under the
"Critical Issues" heading instead of saying "See detailed reviews below"
when there are none.

## 4. Scope and constraints (from user, stated explicitly)

### 4.1 Document-type scope

Whetstone must work on **any kind of written draft**. Academic papers
are the primary near-term use case. The architecture must not assume
academic structure. This rules out:

- Hard-coded section labels (references, funding, ethics, etc.) as a
  primary classification mechanism, because those are academic-specific.
- Pipelines that only make sense for IMRaD documents.

It does not rule out academic-friendly *features* — but they must be
opt-in or detected without a hardcoded list.

### 4.2 Lens "no findings" behaviour

When a lens reviewer looks at a section and concludes there is nothing
to flag (the human-reviewer equivalent of glancing at a bibliography and
moving on), the lens **should emit nothing**. Not a "no findings" note,
not a low-confidence placeholder. Match what a human reviewer would do:
silent.

### 4.3 Small-LLM compatibility

The system targets small local LLMs (Ollama gemma4-style models). Output
schemas must stay flat and simple; we cannot demand the agent classify
sections from a large label vocabulary in a structured response.

### 4.4 No environment-variable configuration

Per the standing project rule (memory `feedback_no_env_vars.md`),
configuration flows via explicit keyword arguments from CLI → surface
API → library. No new env-var reads in this work.

### 4.5 No surprise scope expansion

Each change is scoped, planned, signed off, then implemented. No
bundling of "while I'm here" UX improvements without explicit approval.

## 5. Non-goals (for this work)

The following are explicitly **not** part of this work and will not be
addressed in any commit on this branch:

- **Migrating away from python-docx / lxml.** The bug is fixable in
  place; rebuilding the docx layer is out of scope.
- **Improving the lens prompts themselves** (rigorous / writer / etc.).
  Their behaviour may be revisited later but is not the target here.
- **Adding a "document map preamble" to lens prompts.** Even if it
  would help P2, it's not part of the minimum fix for P1 + P2; we'll
  evaluate it as a Phase 2 design option but it may stay deferred.
- **Building any kind of structured section classifier with a label
  vocabulary.** The previous attempt was rejected for being domain-
  specific and overly complex.
- **Changing the Word-comment rendering machinery (`whetstone/docx/`)
  beyond the namespace fix.** Internal refactors out of scope.

## 6. Constraints on the fix (architectural)

Derived from §4 plus Phase 0 findings:

1. **The P1 namespace fix must include a regression guard.** A unit
   test that constructs a docx and asserts every XML part inside the
   resulting zip uses the standard `w:` prefix for OOXML elements (no
   `ns0:`, `ns1:`, etc.). Otherwise the bug can silently re-emerge in
   a future change to `whetstone/docx/`.

2. **The P2 fix, whatever its mechanism, must not encode a
   document-type-specific label list.** It can use heuristics (citation
   density, paragraph-shape statistics) but must not say "this section
   is the funding statement." Whetstone must remain blind to academic-
   versus-non-academic distinctions at the architectural level.

3. **The fix for P2 must respect the "silent when nothing to say"
   rule.** The cleanest mechanism is likely on the LLM-output side
   (prompt + post-filter), not the document-classification side.

4. **Patch-application visibility.** The user has no way to see
   whether their comments actually anchored to the document body or
   were silently dropped. The CLI should surface the `PatchApplicationResult`
   counts (applied / failed) — but **this is a separate decision** and
   should be a separate commit if approved. (Previously I bundled this
   without asking; the user flagged that as scope expansion.)

## 7. Open questions to be answered in Phase 2 (design options)

The PRD does not yet commit to a mechanism. Phase 2 will lay out
options and trade-offs for these questions:

1. **P1 namespace fix scope.** Minimum: bind `nsmap` on the
   `comments` root in `low_level.py:479`. Larger: introduce a small
   helper that always provides `nsmap` for new roots, then audit the
   ~10 other root-construction sites. Even larger: ship a "validate
   any whetstone-generated .docx has consistent namespaces" check that
   runs as part of every render and raises if a non-`w:` prefix would
   leak.

2. **P2 mechanism.** Options to evaluate:
   - **LLM-side prompt change:** instruct the lens to emit zero
     findings when the section text doesn't have substantive content
     to assess. Domain-agnostic. Cheap. Relies on small-LLM
     compliance.
   - **Framework-side filter on the output:** drop findings whose
     title/rationale matches an "absence-shaped" pattern. Heuristic.
     Brittle.
   - **Heuristic skip on the input:** identify chunks that look like
     bookkeeping content (numbered-citation density, very short
     paragraphs, no verbs) *without* labelling them by type. Skip the
     lens read entirely. Domain-agnostic in name but in practice
     learned from the academic case.
   - **Hybrid:** prompt change + a "no findings" path that's a
     valid Lens output, not coerced into emitting noise.

3. **Should P3 be addressed?** The mismatch between counted findings
   and rendered comments. If yes, options:
   - Count only renderable findings in the header.
   - List unrendered findings under "Critical Issues" instead of
     saying "See detailed reviews below."
   - Both.

4. **Should the patch-application visibility be added?** (§6.4). If
   yes, it's a separate small commit, not part of either P1 or P2.

## 8. Definition of done

When the work on this branch lands on `main`, all of the following
hold:

- Running `andamentum-whetstone` on a fresh source .docx produces an
  output .docx in which **every** finding-with-anchor renders as a
  Word comment visible in Word's review pane.
- The lens reviewers do not emit findings about absence (no "this
  section has no claims" / "no Discussion section" type comments).
- The .docx namespace correctness is enforced by a regression test
  that runs as part of `uv run pytest`.
- No new domain-specific section-label list is shipped.
- No environment-variable reads are added.
- The CLI / pyproject.toml surface unchanged unless explicitly
  approved.
- Canonical green state: pytest passing, pyright 23, ruff clean.

## 9. Approval gate

This PRD requires the user's sign-off before Phase 2 (design options)
begins. Specifically:

1. Does the user-visible problem statement (§3) match what they're
   actually experiencing?
2. Is the document-type scope (§4.1) correctly stated?
3. Are the constraints in §6 the right ones?
4. Are the non-goals in §5 the right ones to defer?
5. Are the open questions in §7 the right things to design against?

A signed-off PRD becomes the contract for what Phase 2 onwards will
produce.
