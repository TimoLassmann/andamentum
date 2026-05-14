"""Audit-style epistemic report — the single HTML renderer.

This module produces the externalised reasoning artefact the system
ships: an audit-first layout optimised for someone who wants to inspect
the system's reasoning trail in addition to reading the answer.

The layout is explicitly designed to answer Schneider (2025)'s
"Chatbot Epistemology" challenge — chatbots fail epistemic-justification
tests under both reliabilism (unreliable: hallucinations, sycophancy)
and internalism (no access to the reasoning — black box). The
externalised reasoning trace below is the positive answer: every
deterministic gate the system applied, every alternative explanation it
considered, every piece of evidence it retrieved, and every threshold
it checked against, all visible without the reader having to trust the
model's testimony.

Structure (top to bottom):

1. **Heading** — research question + meta line (date, model, pipeline
   version, snapshot id).
2. **Jump-to TOC** — single ``aside`` atom with markdown links.
3. **Answer in brief (Q&A)** — single ``items`` panel led by the
   verdict, with the directional posterior + decisive-bands legend
   inline.
4. **Summary** — narrative answer only, deduplicated (no agent-prefix
   preamble, no self-quoting blockquote).
5. **Evidence at a glance** — directional split table + one-line
   per-claim verdict row.
6. **Reasoning trace** — per-claim breakdown:
     - Claim card (statement + verdict badge from closed vocabulary).
     - "How the system reasoned about this claim" — adaptive intro,
       investigation rounds, gate trace table, IBE candidate cards.
     - Supporting evidence (h3 inside body) — one-sentence judgement
       with strength flags as inline italic parentheticals.
     - Contradicting evidence (h3 inside body) — same shape, named
       symmetrically to supporting.
7. **Caveats and limitations** — system-level only (gate-trace
   anomalies, scope gaps, evidence-cap notes). Per-evidence prose
   never reappears here.
8. **Appendix** — full evidence trail, full IBE rationales, gate-trace
   JSON.
9. **Reproducibility footer** — sidebar with pipeline version, model,
   snapshot/artefact IDs, plus the literal CLI re-run command.

Visual constraints (load-bearing):

- **No new CSS** beyond what ``andamentum.typeset`` already styles.
- **No green/red tone callouts.** Verdict badges, gate-status words
  and strength flags are all neutral — the *word* carries the
  meaning, not the colour. Closed verdict vocabulary
  (``Confirmed``/``Confirmed with refinement``/``Inconclusive``/
  ``Refuted``/``Insufficient evidence``) is designed so the lowercased
  ``data-value`` never matches the existing green/red CSS rules in
  ``typeset/atoms.py``.

Run modes:

- **Verify mode** (single claim seeded via ``claim_to_verify``): one
  claim card with full audit trail; the objective-level posterior
  applies directly.
- **Research mode** (decomposition into sub-investigations): one card
  per sub-claim, numbered ``#1, #2, …``. Each sub-claim's gate trace
  uses its own ``integrated_confidence`` for the decisive-posterior
  row.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from andamentum.typeset import Report

from .report_data import (
    ClaimSummary,
    EvidenceSummary,
    GateTraceEntry,
    IBECandidate,
    QUESTION_TYPE_LABELS,
    ReportData,
)
from .thresholds import (
    POSTERIOR_DECISIVE_THRESHOLD,
    POSTERIOR_DIRECTIONAL_BREAKPOINT,
)

# How many top supporting / contradicting items to show inline per claim.
# Items beyond this go into the appendix's full list.
_INLINE_TOP_K = 5


# ──────────────────────────────────────────────────────────────────────────────
# Source-ref → clickable URL
# ──────────────────────────────────────────────────────────────────────────────


_DOI_RE = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_PMID_RE = re.compile(r"^(?:pmid:)?(\d{6,9})$", re.IGNORECASE)
_NCT_RE = re.compile(r"^(NCT\d{8})$", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _source_url(source_ref: str) -> str:
    """Convert a source identifier into a clickable URL.

    - DOI (with or without ``doi:`` prefix) → ``https://doi.org/<doi>``.
    - PMID (with or without ``PMID:`` prefix) → PubMed URL.
    - ClinicalTrials.gov NCT number → trial URL.
    - Already an http(s) URL → unchanged.
    - Anything else → unchanged (the renderer will show it as bare text).
    """
    if not source_ref:
        return source_ref
    s = source_ref.strip()
    if _URL_RE.match(s):
        return s
    m = _DOI_RE.match(s)
    if m:
        return f"https://doi.org/{m.group(1)}"
    m = _PMID_RE.match(s)
    if m:
        return f"https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/"
    m = _NCT_RE.match(s)
    if m:
        return f"https://clinicaltrials.gov/study/{m.group(1)}"
    return s


def _short_source(source_ref: str) -> str:
    """Compact display label for a source identifier."""
    if not source_ref:
        return ""
    s = source_ref.strip()
    if _URL_RE.match(s):
        return re.sub(r"^https?://(www\.)?", "", s)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Closed verdict vocabulary — single source of truth, neutral CSS values
# ──────────────────────────────────────────────────────────────────────────────


# Map raw integrated_assessment tokens + posterior thresholds to one of
# five user-facing labels. The labels are designed so their lowercased
# form does NOT match the existing ``[data-value="supported"]`` /
# ``[data-value="contradicts"]`` CSS rules in typeset — i.e. badges
# render neutral, no green/red tones. The word does the signalling.
_VERDICT_CONFIRMED = "Confirmed"
_VERDICT_CONFIRMED_REFINED = "Confirmed with refinement"
_VERDICT_REFUTED = "Refuted"
_VERDICT_INCONCLUSIVE = "Inconclusive"
_VERDICT_INSUFFICIENT = "Insufficient evidence"

# Posterior bands. Both read from the ``thresholds`` module so the
# renderer and the pipeline cannot disagree.
#
# The pipeline uses TWO different thresholds for the posterior, and the
# legend in the Q&A panel surfaces both:
#
#   - **Directional breakpoint** (``POSTERIOR_DIRECTIONAL_BREAKPOINT``)
#     decides the verdict *label* (Confirmed / Refuted / Inconclusive).
#     Read by ``graph/combination._verdict_label``.
#   - **Decisive threshold** (``POSTERIOR_DECISIVE_THRESHOLD``) decides
#     whether to stop iterating. Read by
#     ``graph/nodes.CheckSynthesisDemand``.
#
# A posterior of 0.70 is "leaning supportive" (above directional) but
# not yet decisive; the system would keep investigating.
_DECISIVE_HI = POSTERIOR_DECISIVE_THRESHOLD             # ≥ this → decisive confirm
_DECISIVE_LO = 1.0 - POSTERIOR_DECISIVE_THRESHOLD       # ≤ this → decisive refute
_DIRECTIONAL_HI = POSTERIOR_DIRECTIONAL_BREAKPOINT      # ≥ this → Confirmed label
_DIRECTIONAL_LO = 1.0 - POSTERIOR_DIRECTIONAL_BREAKPOINT  # ≤ this → Refuted label


def _normalised_verdict(
    data: ReportData,
    claim: Optional[ClaimSummary] = None,
) -> str:
    """Return the closed-vocabulary verdict label.

    Used everywhere the verdict appears: Q&A panel headline, Evidence-
    at-a-glance row, claim-card badge. One mapping function, one source
    of truth. The vocabulary is deliberately limited to five labels so
    a reader (and downstream tooling) can rely on stable strings.

    Selection rules, in order:

    1. ``terminal_state`` not ``completed``         → Insufficient
       evidence (the inquiry suspended judgment; the posterior is
       not a directional answer).
    2. ``posterior`` is None                        → Insufficient
       evidence.
    3. ``posterior ≥ 0.66`` AND
       ``integrated_assessment == "supports_refined"`` → Confirmed
       with refinement.
    4. ``posterior ≥ 0.66``                         → Confirmed.
    5. ``posterior ≤ 0.34``                         → Refuted.
    6. otherwise                                    → Inconclusive.
    """
    cs = data.confidence_scores
    if cs is None or cs.terminal_state != "completed":
        return _VERDICT_INSUFFICIENT
    p = cs.posterior
    if p is None:
        return _VERDICT_INSUFFICIENT
    # Integrated assessment is per-claim; use the claim's if provided,
    # otherwise the first claim's (verify mode) or fall back to the
    # posterior-only mapping for multi-claim cases without a specific
    # claim.
    assessment: Optional[str] = None
    if claim is not None:
        assessment = claim.integrated_assessment
    elif len(data.claims) == 1:
        assessment = data.claims[0].integrated_assessment
    if p >= _DIRECTIONAL_HI:
        if assessment == "supports_refined":
            return _VERDICT_CONFIRMED_REFINED
        return _VERDICT_CONFIRMED
    if p <= _DIRECTIONAL_LO:
        return _VERDICT_REFUTED
    return _VERDICT_INCONCLUSIVE


# ──────────────────────────────────────────────────────────────────────────────
# Posterior body — directional phrasing + decisive-bands legend
# ──────────────────────────────────────────────────────────────────────────────


def _confidence_body(data: ReportData) -> str:
    """Body text for the Q&A panel's "How confident are we?" row.

    Reads the posterior directionally — "Probability the claim is true:
    0.115 → verdict Refuted" — instead of the v1 "Posterior: 11.5%"
    which a non-Bayesian reader misreads as "low confidence". An inline
    legend names the decisive bands so the reader can map any future
    number themselves.
    """
    cs = data.confidence_scores
    if cs is None:
        return "No posterior computed."
    ts = cs.terminal_state
    if ts == "retrieval_failed":
        return "No posterior — retrieval failed before evidence converged."
    if ts == "oscillation_detected":
        return "No posterior — no IBE-certified verdict."
    if ts != "completed":
        return f"No posterior — inquiry terminated: {ts}."
    p = cs.posterior
    if p is None:
        return "No posterior computed."

    verdict = _normalised_verdict(data)
    # Two bands, two different decisions. The legend names both so the
    # reader can map any future number themselves, and the gate-trace
    # row that uses the decisive band stays interpretable.
    parts = [
        f"Probability the claim is true: **{p:.3f}** · verdict **{verdict}**.",
        (
            f"_Verdict band (label-assignment): posterior ≥ "
            f"{_DIRECTIONAL_HI:.2f} → Confirmed · ≤ {_DIRECTIONAL_LO:.2f} → "
            f"Refuted · otherwise Inconclusive._"
        ),
        (
            f"_Decisive band (stop-iteration): posterior ≥ "
            f"{_DECISIVE_HI:.2f} or ≤ {_DECISIVE_LO:.2f} · the pipeline "
            f"would loop back for more evidence outside this band. "
            f"Surfaced in the gate trace below._"
        ),
    ]
    return "\n\n".join(parts)


def _thoroughness_body(data: ReportData) -> str:
    """Body for the Q&A panel's "How thorough?" row — count of evidence
    sources, plus investigation rounds if any fired, plus provider list
    if available from the per-claim trail."""
    parts = [f"{data.stats.total_evidence} evidence sources examined"]
    n_rounds = data.stats.investigation_rounds_total
    if n_rounds:
        parts.append(
            f"across {n_rounds} investigation round{'s' if n_rounds != 1 else ''}"
        )
    base = " ".join(parts) + "."

    # Provider summary from evidence source_types.
    providers = sorted(
        {ev.source_type for ev in data.evidence if ev.source_type},
        key=str.lower,
    )
    if providers:
        named = ", ".join(_provider_label(p) for p in providers[:5])
        if len(providers) > 5:
            named += f" and {len(providers) - 5} more"
        return f"{base} Providers queried: {named}."
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Provider display names — human-readable substitution for source_type
# ──────────────────────────────────────────────────────────────────────────────


_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "europepmc": "Europe PMC",
    "pubmed": "PubMed",
    "openalex": "OpenAlex",
    "web_search": "Web search",
    "clinicaltrials": "ClinicalTrials.gov",
    "crossref": "Crossref",
    "semanticscholar": "Semantic Scholar",
    "arxiv": "arXiv",
}


def _provider_label(source_type: str) -> str:
    """Map internal provider slug → human-readable display name.

    Falls through to the raw slug if not recognised — that way new
    providers do not silently render as empty strings.
    """
    if not source_type:
        return ""
    return _PROVIDER_DISPLAY_NAMES.get(source_type, source_type)


# ──────────────────────────────────────────────────────────────────────────────
# Markdown helpers
# ──────────────────────────────────────────────────────────────────────────────


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table the typeset renderer's TableExtension will
    pick up. Empty rows → empty body (just the header)."""
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_row, sep_row]
    for row in rows:
        cells = [str(c) if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_link(label: str, url: str) -> str:
    safe_label = label.replace("[", "\\[").replace("]", "\\]")
    return f"[{safe_label}]({url})"


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "—"
    return f"{100 * numerator / denominator:.0f}%"


def _sanitize_excerpt(text: str, max_chars: int = 600) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind(". ", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].rstrip() + "…"


def _one_sentence_judgement(text: str, max_chars: int = 200) -> str:
    """Trim a judgement string to its first complete sentence, capped at
    ``max_chars``. The fuller text remains in the appendix. Avoids the
    v1 problem where supporting-evidence bullets dragged in the judge's
    procedural reasoning prose."""
    if not text:
        return ""
    text = text.strip()
    # First period followed by space/end → sentence boundary.
    m = re.search(r"\.(?:\s|$)", text)
    if m and m.end() <= max_chars:
        return text[: m.end()].rstrip()
    return _sanitize_excerpt(text, max_chars=max_chars)


# ──────────────────────────────────────────────────────────────────────────────
# Strength-flag extraction — keyword-driven, no LLM
# ──────────────────────────────────────────────────────────────────────────────


# Closed vocabulary of evidence-strength flags. The presence of a flag
# indicates the system identified the named characteristic in the judge
# prose; the *absence* of flags never means "strong evidence" — it
# means "no characteristic identified by the keyword extractor".
#
# TODO: replace this renderer-side extraction with structured flags
# from the upstream evidence-judge agent schema once the agent prompts
# are extended. The keyword approach is cheap and demonstrative; a
# proper structured field belongs on ``EvidenceSummary``.
_FLAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "RCT": (
        "randomized controlled trial",
        "randomised controlled trial",
        "randomized trial",
        "randomised trial",
        " rct ",
    ),
    "meta-analysis": ("meta-analysis", "meta analysis"),
    "systematic review": ("systematic review",),
    "observational": (
        "observational",
        "retrospective cohort",
        "case-control",
        "case control",
    ),
    "single-arm": (
        "single-arm",
        "single arm",
        "no control group",
    ),
    "combination intervention": (
        "combination therapy",
        "with azithromycin",
        "hcq + azithromycin",
        "+ azithromycin",
    ),
    "confounding-by-indication": (
        "confounding by indication",
        "treatment-selection bias",
        "selection bias",
    ),
    "out-of-scope cohort": (
        "out of scope",
        "out-of-scope",
        "scope-mismatched",
        "scope mismatch",
    ),
}


def _extract_strength_flags(judgement: str) -> list[str]:
    """Extract closed-vocabulary strength flags from a judge's prose.

    Case-insensitive substring match against ``_FLAG_KEYWORDS``. Returns
    flags in declaration order (so a reader sees the strongest signal
    first when multiple match).
    """
    if not judgement:
        return []
    text = " " + judgement.lower() + " "
    return [flag for flag, kws in _FLAG_KEYWORDS.items() if any(k in text for k in kws)]


# ──────────────────────────────────────────────────────────────────────────────
# Evidence-line rendering
# ──────────────────────────────────────────────────────────────────────────────


# Adversarial-search items sometimes leak into the supports bucket
# because the evidence-judge agent flags them with the wrong directional
# label. Defence-in-depth: catch the prefix here at render time so the
# Supporting Evidence section is never contaminated by adversarial
# probe output. The upstream fix is to tighten the judge schema; until
# then this regex is the renderer's guard.
_ADVERSARIAL_PREFIX_RE = re.compile(r"^\s*adversarial\s*\(", re.IGNORECASE)


def _is_adversarial_judgement(judgement: Optional[str]) -> bool:
    if not judgement:
        return False
    return bool(_ADVERSARIAL_PREFIX_RE.match(judgement))


def _evidence_for_claim(
    claim: ClaimSummary, all_evidence: list[EvidenceSummary]
) -> tuple[list[EvidenceSummary], list[EvidenceSummary], list[EvidenceSummary]]:
    """Bucket a claim's evidence into supports / contradicts / no_bearing.

    Defence-in-depth: any item whose ``judgment_reasoning`` begins with
    ``"Adversarial (…":`` is treated as contradicting regardless of the
    ``support_judgment`` field — the prefix is a structural tell of the
    adversarial-search output and should never appear in the supporting
    list. This is the renderer-side guard for the v1 mis-bucketing bug.
    """
    claim_evidence = [
        ev for ev in all_evidence if ev.evidence_id in claim.evidence_ids
    ]
    supports: list[EvidenceSummary] = []
    contradicts: list[EvidenceSummary] = []
    no_bearing: list[EvidenceSummary] = []
    for ev in claim_evidence:
        if _is_adversarial_judgement(ev.judgment_reasoning):
            # Adversarial probe output always goes to contradicting,
            # never supporting, regardless of the support_judgment
            # label the upstream judge attached.
            contradicts.append(ev)
            continue
        if ev.support_judgment == "supports":
            supports.append(ev)
        elif ev.support_judgment == "contradicts":
            contradicts.append(ev)
        else:
            no_bearing.append(ev)
    return supports, contradicts, no_bearing


def _evidence_line(
    ev: EvidenceSummary,
    *,
    short_judgement: bool = True,
    include_flags: bool = True,
) -> str:
    """One markdown line representing an evidence item for inline display.

    Reader-first order: lead with the **reference** (clickable when the
    identifier is a known scheme — DOI/PMID/NCT/URL), then a small
    provider pill (inline-code for monospace badge look) for
    completeness, then the judgement reasoning text. The reference is
    what a reader cares about; the provider is plumbing metadata and is
    rendered subordinate to it.

    Strength flags (RCT / meta-analysis / observational / …) appear as
    an italic parenthetical at the end of the judgement — they share
    the neutral typeset palette and need no new CSS.

    Format::

        [ref](url) · `Provider` — Judgement sentence. *(flag, flag, ...)*
    """
    head_parts: list[str] = []
    if ev.source_ref:
        url = _source_url(ev.source_ref)
        label = _short_source(ev.source_ref)
        if url != ev.source_ref:
            head_parts.append(_md_link(label, url))
        else:
            head_parts.append(label)
    if ev.source_type:
        head_parts.append(f"`{_provider_label(ev.source_type)}`")
    head = " · ".join(head_parts)

    raw = ev.judgment_reasoning or ""
    judgement_text = (
        _one_sentence_judgement(raw, max_chars=200)
        if short_judgement
        else _sanitize_excerpt(raw, max_chars=400)
    )

    flag_str = ""
    if include_flags:
        flags = _extract_strength_flags(raw)
        if flags:
            flag_str = f" *({', '.join(flags)})*"

    if judgement_text:
        return f"{head} — {judgement_text}{flag_str}" if head else f"{judgement_text}{flag_str}"
    return head


# ──────────────────────────────────────────────────────────────────────────────
# Summary preamble stripping
# ──────────────────────────────────────────────────────────────────────────────


def _strip_summary_preamble(direct_answer: str) -> str:
    """Drop leading ``Research Question:`` / ``Evidence Sources:`` /
    blockquote lines that some agents prepend to their summary prose.

    The Summary section is for narrative answer text only — the
    metadata is already in the Q&A panel and the heading meta-line.
    Without this strip, the v1 report rendered the same Research
    Question three times in three lines (each with a different
    Evidence Sources count, no less).
    """
    lines = direct_answer.split("\n") if direct_answer else []
    while lines:
        stripped = lines[0].strip()
        if (
            stripped.startswith("**Research Question:**")
            or stripped.startswith("**Evidence Sources:**")
            or stripped.startswith(">")
            or stripped == ""
        ):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Gate-trace rendering
# ──────────────────────────────────────────────────────────────────────────────


_GATE_DISPLAY_NAMES: dict[str, str] = {
    "scrutiny": "Scrutiny",
    "convergence": "Convergence",
    "adversarial_balance": "Adversarial balance",
    "deductive_validation": "Deductive validation",
    "computational_verification": "Computational verification",
    "posterior_decisive": "Posterior decisive",
}


def _render_gate_trace(trace: list[GateTraceEntry]) -> str:
    """Markdown table rendering the per-claim gate trace.

    Status values are plain words (``satisfied`` / ``failed`` /
    ``skipped``) — no icons, no colour. The reader reads the word. The
    optional ``note`` annotates a row (e.g. ``decisive (refutes)``) and
    renders as italic text after the gate name.
    """
    rows: list[list[str]] = []
    for g in trace:
        display = _GATE_DISPLAY_NAMES.get(g.name, g.name.replace("_", " "))
        if g.note:
            display = f"{display} *({g.note})*"
        rows.append([display, g.routing, g.required, g.observed, g.status])
    return _md_table(
        ["Gate", "Routing", "Required", "Observed", "Status"],
        rows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# IBE candidate cards
# ──────────────────────────────────────────────────────────────────────────────


def _ibe_role(c: IBECandidate) -> str:
    """Closed-vocabulary role label for an IBE candidate.

    Avoids the existing ``[data-value="rejected"]`` CSS rule that
    would tint the badge red — uses ``not selected`` instead so the
    badge falls through to neutral.
    """
    if c.chosen:
        return "selected"
    if c.runner_up:
        return "runner-up"
    return "not selected"


def _render_ibe_card(r: Report, c: IBECandidate, claim_id: str) -> None:
    """Append one ``card`` atom for an IBE candidate.

    Each candidate's description is rendered un-truncated — the value
    proposition is auditability, so truncating mid-clause defeats the
    purpose. The badge carries the role; loveliness and likeliness
    scores are inline at the top of the body.
    """
    lovel = f"{c.loveliness:.2f}" if c.loveliness is not None else "—"
    likl = f"{c.likeliness:.2f}" if c.likeliness is not None else "—"
    head = (
        f"**Candidate {c.candidate_id}** — verdict *{c.verdict}* · "
        f"loveliness {lovel} · likeliness {likl}"
    )
    body = c.description or ""
    r.card(
        f"{head}\n\n{body}".rstrip(),
        badge=_ibe_role(c),
        id=f"ibe-{c.candidate_id}-{_claim_slug(claim_id)}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-claim renderers
# ──────────────────────────────────────────────────────────────────────────────


def _audit_trail_intro(claim: ClaimSummary) -> str:
    """Adaptive overall paragraph that opens the per-claim audit trail.

    Always present when there is any audit-trail content. Tells the
    reader the shape of the investigation before diving into the per-
    step details. Reads differently for claims where the initial gather
    sufficed vs. claims that needed gap-driven follow-up rounds.
    """
    sentences = [
        "The system began with an **initial evidence gather**, retrieving "
        "items from external providers (PubMed, Europe PMC, "
        "ClinicalTrials.gov, and similar)."
    ]
    if claim.investigation_rounds:
        n = len(claim.investigation_rounds)
        sentences.append(
            f"When that initial gather did not fully resolve the claim, "
            f"a gap-analysis step proposed {n} further methodological "
            f"angle{'s' if n != 1 else ''} for follow-up search — each "
            f"is listed below as an *investigation round*."
        )
    else:
        sentences.append(
            "The initial gather was sufficient — the gap-analysis step "
            "did not request any further follow-up rounds for this "
            "claim."
        )
    if claim.ibe_candidates:
        sentences.append(
            "At the integration step the system enumerated **alternative "
            "explanations** of the evidence pattern and selected the "
            "strongest using *inference-to-the-best-explanation* (IBE) — "
            "a standard philosophy-of-science procedure for picking "
            "between rival hypotheses on the same evidence."
        )
    if claim.gate_trace:
        sentences.append(
            "Below: the **deterministic gate trace** — every check the "
            "system applied, with the threshold required and the value "
            "observed. The reader can audit each step without trusting "
            "the model's testimony."
        )
    return " ".join(sentences)


def _render_reasoning_block_md(
    claim: ClaimSummary,
    *,
    n_supports: int = 0,
    n_contradicts: int = 0,
    n_no_bearing: int = 0,
) -> str:
    """Build the markdown body for the per-claim reasoning trace.

    Uses embedded ``####`` headings to nest sub-sections under the
    parent h2 "Reasoning trace". Per the v2 constraint of no typeset-
    module changes, the renderer cannot emit h3 via ``prose(heading=)``
    — that always renders h2. Embedding the heading inside the
    markdown body gives the visual nesting we want using the existing
    ``.typeset-prose h4`` CSS.

    The ``n_supports`` / ``n_contradicts`` / ``n_no_bearing`` counts
    are anchored into the IBE sub-section intro so candidate
    descriptions that reference "the evidence pattern" have a concrete
    referent visible immediately above the candidate cards.
    """
    parts: list[str] = []

    # Adaptive intro — always.
    parts.append(_audit_trail_intro(claim))
    parts.append("")

    # Investigation rounds.
    if claim.investigation_rounds:
        parts.append("#### Investigation rounds")
        parts.append("")
        parts.append(
            "When the initial gather couldn't fully resolve the claim, "
            "the gap-analysis agent proposed new methodological angles. "
            "Each intent below was routed to providers; the yield count "
            "is how many evidence items the routing returned."
        )
        parts.append("")
        for rnd in claim.investigation_rounds:
            plural = "" if rnd.evidence_count == 1 else "s"
            parts.append(
                f"- **Round {rnd.round_index}** "
                f"_(yielded {rnd.evidence_count} item{plural})_ — {rnd.intent}"
            )
        parts.append("")

    # Gate trace.
    if claim.gate_trace:
        parts.append("#### Gate trace")
        parts.append("")
        parts.append(
            "Each gate is a deterministic check the system applies to "
            "the claim. Gates are routed by question type — irrelevant "
            "gates are skipped, not silently passed. Status values are "
            "literal: *satisfied*, *failed*, *skipped*."
        )
        parts.append("")
        parts.append(_render_gate_trace(claim.gate_trace))
        parts.append("")

    # IBE candidates — rendered as separate card atoms appended after
    # this prose, NOT inside this markdown (card atoms cannot be
    # embedded in prose). Surfaced here only as a heading + lead-in,
    # with the actual candidate cards appended by the caller.
    if claim.ibe_candidates:
        parts.append("#### Alternative explanations the system considered")
        parts.append("")
        # Anchor "the evidence pattern" with the concrete counts so the
        # candidate descriptions below have a visible referent. Without
        # this anchor, a candidate that opens with "the evidence pattern
        # is heterogeneous..." reads as a free-floating noun phrase.
        total_judged = n_supports + n_contradicts + n_no_bearing
        if total_judged > 0:
            parts.append(
                f"**The evidence pattern for this claim:** "
                f"{n_supports} supporting · {n_contradicts} "
                f"contradicting · {n_no_bearing} no-bearing "
                f"(total {total_judged} judged items). The individual "
                f"items are listed in full under *Supporting evidence* "
                f"and *Contradicting evidence* below; the directional "
                f"split is in *Evidence at a glance* above. The "
                f"candidates that follow offer rival readings of that "
                f"pattern."
            )
            parts.append("")
        parts.append(
            "Each candidate was scored on **loveliness** (how well the "
            "explanation fits the observed evidence) and **likeliness** "
            "(prior probability that the explanation is true). The "
            "candidate with the strongest combined score was selected as "
            "the integrated verdict; runner-up and not-selected "
            "candidates are retained for auditability."
        )
        parts.append("")
        if claim.integrated_assessment:
            parts.append(
                f"**Integrated assessment:** {claim.integrated_assessment}"
            )
            parts.append("")

    return "\n".join(parts).rstrip()


def _render_supports_block_md(
    supports: list[EvidenceSummary],
) -> str:
    """Build the markdown body for "Supporting evidence (N items)".

    Embedded as a single prose body so the heading nests as h3 under
    the parent h2 "Reasoning trace".
    """
    if not supports:
        return ""
    n = len(supports)
    parts: list[str] = [
        f"### Supporting evidence ({n} item{'s' if n != 1 else ''})"
    ]
    parts.append("")
    top = supports[:_INLINE_TOP_K]
    for ev in top:
        parts.append(f"- {_evidence_line(ev)}")
    if len(supports) > _INLINE_TOP_K:
        extra = len(supports) - _INLINE_TOP_K
        parts.append("")
        parts.append(
            f"_({extra} additional supporting item{'s' if extra != 1 else ''} "
            "in the appendix below.)_"
        )
    return "\n".join(parts)


def _render_contradicting_block_md(
    contradicts: list[EvidenceSummary],
    counterargs: list[Any],
) -> str:
    """Build the markdown body for "Contradicting evidence (N items)".

    Combines judge-bucketed contradicts + adversarial-search items
    surfaced under the symmetric "Contradicting evidence" label
    (previously the asymmetric "Adversarial probe").
    """
    total = len(contradicts) + len(counterargs)
    if total == 0:
        return ""
    parts: list[str] = [
        f"### Contradicting evidence ({total} item{'s' if total != 1 else ''})"
    ]
    parts.append("")
    parts.append(
        "The system explicitly searched for evidence that would "
        "**contradict** this claim: replication failures, null "
        "results, and rival findings. Items below were either judged "
        "as contradicting by the evidence judge or surfaced by the "
        "adversarial-search step."
    )
    parts.append("")
    seen: set[str] = set()
    lines: list[str] = []
    for ev in contradicts[:_INLINE_TOP_K]:
        key = ev.source_ref or f"ev-{ev.evidence_id}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {_evidence_line(ev)}")
    for adv in counterargs[:_INLINE_TOP_K]:
        key = getattr(adv, "source_ref", None) or f"adv-{id(adv)}"
        if key in seen:
            continue
        seen.add(key)
        content = _one_sentence_judgement(
            getattr(adv, "counterargument", "") or "", max_chars=200
        )
        ref = getattr(adv, "source_ref", None)
        if ref:
            url = _source_url(ref)
            label = _short_source(ref)
            if url != ref:
                content = f"{_md_link(label, url)} — {content}"
            else:
                content = f"{label} — {content}"
        lines.append(f"- {content}")
    parts.extend(lines)
    if total > _INLINE_TOP_K:
        extra = max(0, total - _INLINE_TOP_K)
        parts.append("")
        parts.append(
            f"_({extra} additional item{'s' if extra != 1 else ''} "
            "in the appendix below.)_"
        )
    return "\n".join(parts)


def _render_claim_section(
    r: Report,
    data: ReportData,
    claim: ClaimSummary,
    all_evidence: list[EvidenceSummary],
    adv_by_claim: dict[str, list[Any]],
    *,
    show_label_prefix: str = "",
) -> None:
    """Render one claim's section: card + reasoning prose + supports +
    contradicting + IBE candidate cards.

    The h2 "Reasoning trace" header is emitted by the caller. Each
    claim contributes: (1) a card atom with the statement + verdict
    badge, (2) one prose atom for the reasoning trail (intro,
    investigation rounds, gate trace) with embedded h4 headings, (3)
    zero-or-more card atoms — one per IBE candidate, (4) one prose
    atom for supporting evidence (h3 embedded), (5) one prose atom for
    contradicting evidence (h3 embedded).
    """
    supports, contradicts, no_bearing = _evidence_for_claim(claim, all_evidence)
    verdict_label = _normalised_verdict(data, claim=claim)

    # ── Claim card ─────────────────────────────────────────────────────
    statement = f"**Claim:** {claim.statement}"
    if show_label_prefix:
        statement = f"**Claim {show_label_prefix}:** {claim.statement}"

    # Inlined details — replaces the v1 collapsible since the contents
    # are 1–2 lines.
    detail_lines: list[str] = []
    if claim.scope:
        detail_lines.append(f"**Scope:** {claim.scope}")
    detail_lines.append(
        f"**Evidence:** {len(supports)} supporting · "
        f"{len(contradicts)} contradicting · "
        f"{len(no_bearing)} no bearing"
    )
    card_body = statement + "\n\n" + " · ".join(detail_lines)

    r.card(
        card_body,
        badge=verdict_label,
        id=_claim_slug(claim.claim_id),
    )

    # ── Reasoning trail (intro + rounds + gate trace) ───────────────────
    reasoning_md = _render_reasoning_block_md(
        claim,
        n_supports=len(supports),
        n_contradicts=len(contradicts),
        n_no_bearing=len(no_bearing),
    )
    if reasoning_md.strip():
        r.prose(
            "### How the system reasoned about this claim\n\n" + reasoning_md,
            id=f"reasoning-{_claim_slug(claim.claim_id)}",
        )
        # IBE candidate cards appended after the prose (cards cannot be
        # embedded inside a prose atom). Order: selected first, then
        # runner-up, then not-selected — matches reader expectation.
        for c in sorted(
            claim.ibe_candidates,
            key=lambda x: (not x.chosen, not x.runner_up, x.candidate_id),
        ):
            _render_ibe_card(r, c, claim.claim_id)

    # ── Supporting evidence ─────────────────────────────────────────────
    supports_md = _render_supports_block_md(supports)
    if supports_md:
        r.prose(
            supports_md,
            id=f"supports-{_claim_slug(claim.claim_id)}",
        )

    # ── Contradicting evidence ──────────────────────────────────────────
    counterargs = adv_by_claim.get(claim.claim_id, [])
    contra_md = _render_contradicting_block_md(contradicts, counterargs)
    if contra_md:
        r.prose(
            contra_md,
            id=f"contradicting-{_claim_slug(claim.claim_id)}",
        )


def _claim_slug(claim_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", claim_id.lower())


# ──────────────────────────────────────────────────────────────────────────────
# Caveats and Limitations (system-level only)
# ──────────────────────────────────────────────────────────────────────────────


def _build_caveats_and_limitations(data: ReportData) -> list[str]:
    """System-level bullets only — NOT a re-dump of per-evidence
    judgements.

    Sources, in this order:
      1. Gate-trace anomalies (per claim, the first failing gate).
      2. Scope gaps recorded as ``UncertaintySummary`` items.
      3. Evidence-cap notes when the run hit a known retrieval limit
         (not currently surfaced upstream; placeholder for future
         plumbing).
    """
    bullets: list[str] = []

    # 1. Gate-trace anomalies.
    for claim in data.claims:
        failed = [g for g in claim.gate_trace if g.status == "failed"]
        if not failed:
            continue
        # Cite the first failing gate; subsequent failures usually
        # correlate and would be noise here.
        g = failed[0]
        display = _GATE_DISPLAY_NAMES.get(g.name, g.name.replace("_", " "))
        bullets.append(
            f"**{display} gate did not pass** for claim "
            f"*{_sanitize_excerpt(claim.statement, max_chars=120)}* "
            f"(required {g.required}; observed {g.observed}). "
            "The verdict reported above was resolved by the remaining gates."
        )

    # 2. Scope gaps from unresolved uncertainties.
    for unc in data.uncertainties:
        if unc.is_resolved:
            continue
        bullets.append(unc.description)

    return bullets


# ──────────────────────────────────────────────────────────────────────────────
# Appendix block builders
# ──────────────────────────────────────────────────────────────────────────────


def _build_evidence_appendix_md(data: ReportData) -> str:
    """Full evidence trail, grouped by direction. Used as ``details`` on
    the appendix card so the trail is one-click expandable."""
    if not data.evidence:
        return ""
    # Bucket using the same defence as the per-claim split so an
    # adversarial item never appears under "Supporting evidence".
    supporting: list[EvidenceSummary] = []
    contradicting: list[EvidenceSummary] = []
    no_bearing: list[EvidenceSummary] = []
    for ev in data.evidence:
        if _is_adversarial_judgement(ev.judgment_reasoning):
            contradicting.append(ev)
        elif ev.support_judgment == "supports":
            supporting.append(ev)
        elif ev.support_judgment == "contradicts":
            contradicting.append(ev)
        else:
            no_bearing.append(ev)
    groups = [
        ("Supporting evidence", supporting),
        ("Contradicting evidence", contradicting),
        ("Evidence judged as having no bearing on the claim", no_bearing),
    ]
    parts: list[str] = []
    for label, group in groups:
        if not group:
            continue
        parts.append(f"### {label} ({len(group)})")
        parts.append("")
        for ev in group:
            parts.append(
                f"- {_evidence_line(ev, short_judgement=False, include_flags=True)}"
            )
        parts.append("")
    return "\n".join(parts).rstrip()


def _build_gate_trace_json(data: ReportData) -> str:
    """Machine-readable per-claim gate trace for downstream tooling.

    A small but disproportionately useful artefact: the same gate trace
    the human reader sees, dumped as JSON. Evaluation harnesses can
    parse this without having to scrape HTML.
    """
    payload: list[dict[str, Any]] = []
    for claim in data.claims:
        payload.append(
            {
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "verdict": _normalised_verdict(data, claim=claim),
                "posterior": (
                    data.confidence_scores.posterior
                    if data.confidence_scores
                    else None
                ),
                "integrated_assessment": claim.integrated_assessment,
                "gate_trace": [
                    {
                        "name": g.name,
                        "routing": g.routing,
                        "required": g.required,
                        "observed": g.observed,
                        "status": g.status,
                        "note": g.note,
                    }
                    for g in claim.gate_trace
                ],
            }
        )
    return json.dumps(payload, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Terms-of-art glossary — anchors the reader before the Reasoning trace
# ──────────────────────────────────────────────────────────────────────────────


# A short inline glossary placed once at the top of the Reasoning trace,
# so the reader has a definition for every term-of-art that appears in
# the gate trace, adaptive intro, and IBE prose. PRD §8.4: "every
# term-of-art is either defined inline on first use or linked to a
# glossary section." This is the linked-glossary route — single aside
# atom, neutral typography, no new CSS.
_TERMS_GLOSSARY_MD = (
    "**Reading this section.** A few terms appear in the gate trace "
    "and intro prose that come from the system's verification "
    "vocabulary:\n\n"
    "- **Scrutiny** — an independent verifier agent reviews each "
    "piece of evidence's directional judgement and either agrees "
    "(*pass*) or flags a problem (*fail*). A failed scrutiny means "
    "the per-evidence judge and the verifier disagreed.\n"
    "- **Convergence** — Reichenbach's common-cause check: do two or "
    "more *independent* sources point in the same direction? "
    "Independence is judged by source type and corpus origin.\n"
    "- **Adversarial balance** — a score in [0, 1] summarising how the "
    "claim fared under the adversarial probe. Values near 0 mean the "
    "claim was refuted; near 1 mean it survived; the middle band is "
    "contested.\n"
    "- **Gap-analysis** — when initial evidence retrieval doesn't "
    "resolve a claim, this step proposes new methodological angles "
    "for follow-up search (e.g. \"find an RCT with a different "
    "timing of intervention\"). Each angle becomes an *investigation "
    "round*.\n"
    "- **Posterior** — the probability the claim is true, in [0, 1], "
    "computed by the system's confidence-scoring step. A posterior of "
    "0.85 means the system estimates an 85% probability the claim "
    "holds; 0.15 means 15% (i.e. 85% probability against).\n"
    "- **IBE** (*inference-to-the-best-explanation*) — a standard "
    "philosophy-of-science procedure: enumerate rival explanations of "
    "the evidence, score each on **loveliness** (fit) and **likeliness** "
    "(prior plausibility), select the strongest."
)


# ──────────────────────────────────────────────────────────────────────────────
# Count-consistency invariant — PRD R2 "raise loudly on mismatch"
# ──────────────────────────────────────────────────────────────────────────────


def _check_count_invariants(data: ReportData) -> None:
    """Raise ``ValueError`` if ``data.stats`` reports fewer items in a
    direction than ``data.evidence`` actually contains.

    Both come from the same upstream source — ``data.stats`` is tallied
    from the raw ``all_evidence`` list (pre-dedup) and
    ``data.evidence`` is the dedup'd view rendered to the reader.
    Because of dedup, ``stats`` counts must be **≥** ``data.evidence``
    counts; the inverse is impossible without an upstream bug. The
    v1 report had this exact divergence (11 supporting in one
    section, 9 in another) and surfaced no warning — this invariant
    is the renderer's defence against that.

    No silent failures: the project rule is *crashes ≫ silent wrong
    answers*. A failing report makes the upstream inconsistency
    visible immediately rather than letting the reader stare at two
    contradictory numbers.
    """
    s = data.stats
    direction_counts: dict[str, tuple[int, int]] = {}
    for direction, stat_count in (
        ("supports", s.evidence_supports),
        ("contradicts", s.evidence_contradicts),
        ("no_bearing", s.evidence_no_bearing),
    ):
        ev_count = sum(
            1 for ev in data.evidence if ev.support_judgment == direction
        )
        direction_counts[direction] = (stat_count, ev_count)
        if ev_count > stat_count:
            raise ValueError(
                f"Evidence count invariant violated for direction "
                f"{direction!r}: data.stats reports {stat_count} but "
                f"data.evidence contains {ev_count} items with this "
                f"judgement. stats counts must be ≥ data.evidence counts "
                f"(stats is tallied from the pre-dedup raw entity list; "
                f"data.evidence is its dedup'd view). This inversion "
                f"suggests upstream ReportData extraction is "
                f"inconsistent — fix the producer rather than render a "
                f"misleading report. All directions (stat_count, "
                f"data.evidence count): {direction_counts}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Top-level builder
# ──────────────────────────────────────────────────────────────────────────────


def build_audit_report(data: ReportData) -> list[dict[str, Any]]:
    """Build the audit report's atom list.

    Returns a list of typeset atom dicts ready to be passed to the
    typeset renderer. Caller produces HTML via ``typeset.render``.

    Invariants enforced at build time:
      - Evidence counts agree across all sections (one source of truth
        from ``_evidence_for_claim``; if data.stats disagrees, log a
        warning — the per-claim split wins).
      - No adversarial item is rendered under "Supporting evidence" (see
        ``_is_adversarial_judgement`` defence).
      - Verdict badge uses the closed vocabulary defined in this module
        — labels whose lowercased ``data-value`` does NOT match the
        existing green/red CSS rules in ``typeset/atoms.py``.
      - ``data.stats`` direction counts must be ≥ the corresponding
        counts in ``data.evidence`` — see ``_check_count_invariants``.
        Failure raises ``ValueError`` rather than silently rendering
        contradictory counts.
    """
    _check_count_invariants(data)
    r = Report(style="article")

    # ── Heading ─────────────────────────────────────────────────────────
    meta_parts: list[str] = [
        data.investigation_date.strftime("%Y-%m-%d"),
        data.model_used,
    ]
    if data.pipeline_version:
        if data.pipeline_git_ref:
            meta_parts.append(
                f"andamentum v{data.pipeline_version} ({data.pipeline_git_ref})"
            )
        else:
            meta_parts.append(f"andamentum v{data.pipeline_version}")
    if data.snapshot_id:
        meta_parts.append(f"snapshot {data.snapshot_id[:12]}…")
    r.heading(data.research_question, meta=" · ".join(meta_parts))

    multi_claim = len(data.claims) > 1

    # ── Jump-to TOC ─────────────────────────────────────────────────────
    r.aside(
        content=(
            "**Jump to:** "
            "[Answer](#answer) · "
            "[Summary](#summary) · "
            "[Evidence at a glance](#evidence-glance) · "
            "[Reasoning trace](#reasoning) · "
            "[Caveats](#caveats) · "
            "[Appendix](#appendix)"
        )
    )

    # ── Answer in brief (Q&A panel) ─────────────────────────────────────
    qt_body: str | None = None
    if data.question_type:
        qt_label = QUESTION_TYPE_LABELS.get(data.question_type, data.question_type)
        qt_body = f"This is a {qt_label}."

    verdict_body = data.verdict
    if not verdict_body:
        verdict_body = _normalised_verdict(data) + "."
    if multi_claim and not data.verdict:
        verdict_body = (
            f"{_normalised_verdict(data)} — see per-sub-claim verdicts below."
        )

    qa_entries: list[dict[str, str]] = []
    qa_entries.append({"label": "What did we find?", "body": verdict_body})
    qa_entries.append(
        {
            "label": "What was studied?",
            "body": data.clarified_question or data.research_question,
        }
    )
    if qt_body:
        qa_entries.append({"label": "What type of question?", "body": qt_body})
    qa_entries.append(
        {"label": "How confident are we?", "body": _confidence_body(data)}
    )
    qa_entries.append(
        {
            "label": "How thorough was the investigation?",
            "body": _thoroughness_body(data),
        }
    )
    if data.reproduction_command:
        qa_entries.append(
            {
                "label": "Reproduction",
                "body": (
                    f"`{data.reproduction_command}`\n\n"
                    "Full reproducibility metadata in the footer below."
                ),
            }
        )
    r.items(entries=qa_entries, id="answer")

    # ── Summary (narrative only — preamble stripped) ─────────────────────
    narrative = _strip_summary_preamble(data.direct_answer or "")
    if narrative:
        r.prose(narrative, heading="Summary", id="summary")

    # ── Evidence at a glance ─────────────────────────────────────────────
    s = data.stats

    # Single source of truth for per-direction counts: the per-claim
    # bucketer. If data.stats disagrees, the per-claim split wins
    # (data.stats is computed upstream and can be stale by 1-2 items).
    total_supports = 0
    total_contradicts = 0
    total_no_bearing = 0
    for c in data.claims:
        sup, con, nb = _evidence_for_claim(c, data.evidence)
        total_supports += len(sup)
        total_contradicts += len(con)
        total_no_bearing += len(nb)
    judged_total = total_supports + total_contradicts + total_no_bearing
    # Fallback: if no claims yet, use data.stats so single-claim cards
    # with no evidence_ids still render a sensible row.
    if judged_total == 0:
        total_supports = s.evidence_supports
        total_contradicts = s.evidence_contradicts
        total_no_bearing = s.evidence_no_bearing
        judged_total = total_supports + total_contradicts + total_no_bearing

    sof_rows = [
        ["Supporting", str(total_supports), _pct(total_supports, judged_total)],
        [
            "Contradicting",
            str(total_contradicts),
            _pct(total_contradicts, judged_total),
        ],
        [
            "No bearing _(retained for audit, not weighted into verdict)_",
            str(total_no_bearing),
            _pct(total_no_bearing, judged_total),
        ],
    ]
    sof_lines: list[str] = [
        f"The system retrieved **{s.total_evidence} evidence items** "
        f"and judged each piece against the claim. The directional split:",
        "",
        _md_table(["Direction", "Items", "Share"], sof_rows),
    ]
    if data.claims:
        sof_lines.append("")
        sof_lines.append(
            "Per-claim verdict"
            f"{'s' if multi_claim else ''}:"
        )
        sof_lines.append("")
        claim_rows: list[list[str]] = []
        for i, claim in enumerate(data.claims, start=1):
            label = (
                f"Claim #{i}: {_sanitize_excerpt(claim.statement, max_chars=120)}"
                if multi_claim
                else _sanitize_excerpt(claim.statement, max_chars=140)
            )
            claim_rows.append([label, _normalised_verdict(data, claim=claim)])
        sof_lines.append(_md_table(["Claim", "Verdict"], claim_rows))
    if s.investigation_rounds_total:
        sof_lines.append("")
        sof_lines.append(
            f"_{s.investigation_rounds_total} investigation round"
            f"{'s' if s.investigation_rounds_total != 1 else ''} across "
            f"{len([c for c in data.claims if c.investigation_rounds])} "
            "claim(s) with audit trail visible below._"
        )
    r.prose(
        "\n".join(sof_lines),
        heading="Evidence at a glance",
        id="evidence-glance",
    )

    # ── Reasoning trace (per-claim breakdown) ───────────────────────────
    adv_by_claim: dict[str, list[Any]] = {}
    for adv in data.adversarial:
        adv_by_claim.setdefault(adv.claim_id, []).append(adv)

    if multi_claim:
        intro = (
            f"The question was decomposed into {len(data.claims)} sub-claims, "
            "each investigated separately. For each sub-claim below: the "
            "**claim under investigation**, how the system reasoned about it, "
            "the strongest supporting evidence, and the strongest "
            "counter-evidence the adversarial probe surfaced. The system's "
            "overall verdict is in the Q&A panel above."
        )
    else:
        intro = (
            "Below: the **claim under investigation**, how the system "
            "reasoned about it, the strongest supporting evidence, and the "
            "strongest counter-evidence the adversarial probe surfaced. "
            "The system's verdict is in the Q&A panel above."
        )
    r.prose(intro, heading="Reasoning trace", id="reasoning")

    # Terms-of-art glossary — placed once before the per-claim
    # breakdowns so the reader sees the definitions for scrutiny,
    # convergence, adversarial balance, gap-analysis, posterior, and
    # IBE *before* encountering them in the gate trace and intro prose
    # below. Uses the existing ``aside`` atom — small Inter sans-serif
    # on the neutral beige tone the typeset CSS already provides; no
    # new styling.
    r.aside(content=_TERMS_GLOSSARY_MD)

    if multi_claim:
        for i, claim in enumerate(data.claims, start=1):
            _render_claim_section(
                r,
                data,
                claim,
                data.evidence,
                adv_by_claim,
                show_label_prefix=f"#{i}",
            )
    else:
        for claim in data.claims:
            _render_claim_section(r, data, claim, data.evidence, adv_by_claim)

    # ── Caveats and limitations (system-level only) ─────────────────────
    bullets = _build_caveats_and_limitations(data)
    if bullets:
        body = "\n".join(f"- {b}" for b in bullets)
        r.prose(body, heading="Caveats and limitations", id="caveats")

    if data.open_questions:
        r.prose(
            "\n".join(f"1. {q}" for q in data.open_questions),
            heading="Open questions",
            id="open-questions",
        )

    # ── Appendix ────────────────────────────────────────────────────────
    appendix_md = _build_evidence_appendix_md(data)
    if appendix_md:
        r.card(
            "**Full evidence trail** — every retrieved item with its "
            "one-sentence judgement and clickable source.",
            badge="appendix",
            id="appendix",
            details=appendix_md,
        )

    # Machine-readable gate trace — small, cheap, hugely useful for
    # downstream evaluation harnesses that want to parse the report
    # without scraping HTML.
    if any(c.gate_trace for c in data.claims):
        r.card(
            "**Gate-trace JSON** — machine-readable copy of the per-claim "
            "deterministic checks. Useful for evaluation harnesses.",
            badge="machine-readable",
            id="appendix-gate-json",
            details="```json\n" + _build_gate_trace_json(data) + "\n```",
        )

    # ── Reproducibility footer (sidebar atom) ───────────────────────────
    repro_groups: dict[str, dict[str, str]] = {
        "Pipeline": {
            "version": data.pipeline_version or "unknown",
            "git ref": data.pipeline_git_ref or "—",
        },
        "Model": {
            "primary": data.model_used,
            "date": data.investigation_date.isoformat(),
        },
        "Persistence": {
            "snapshot": (
                (data.snapshot_id[:12] + "…") if data.snapshot_id else "—"
            ),
            "artefact": (
                (data.artefact_id[:12] + "…") if data.artefact_id else "—"
            ),
            "database": data.database_name or "—",
        },
    }
    r.aside(groups=repro_groups)
    # The reproduction command is already surfaced in the Q&A panel's
    # "Reproduction" row; the sidebar above carries the supporting
    # metadata (snapshot id, version). A second prose block here would
    # render the command a third time, so we drop it.

    return r.atoms


__all__ = [
    "build_audit_report",
]
