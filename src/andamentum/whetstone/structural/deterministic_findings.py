"""Synthesise deterministic Findings from StructuralFacts.

Every check here is pure data manipulation — no LLM, no embeddings, no
network. Each finding has confidence ``high`` because the underlying
checks are mechanically true (a citation key either appears in the
references list or it doesn't).

Severity is calibrated by the issue's likely real-world impact:
  • ``major``    — would invalidate or seriously mislead the reader
  • ``moderate`` — author should fix; readers can usually work around
  • ``minor``    — stylistic; flagged for completeness
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..schemas import Finding, Quote
from .checklist import run_checklist_checks
from .crossrefs import (
    find_equation_anchors,
    find_figure_anchors,
    find_section_anchors,
    find_table_anchors,
)
from .stat_consistency import check_stat_consistency
from .types import (
    CitationOccurrence,
    CrossReference,
    NumericClaim,
    SectionRef,
    StructuralFacts,
    TermDefinition,
    TermUsage,
)


def synthesize_deterministic_findings(
    *,
    sections: list[SectionRef],
    facts: StructuralFacts,
    markdown: str = "",
) -> list[Finding]:
    """Run every deterministic check against the StructuralFacts.

    Returns Findings in stable order (per check type, then per occurrence)
    so output is reproducible across runs.

    ``markdown`` is the harvested full-document text. When provided,
    pre-submission checklist checks (required statements, keywords,
    title, abstract) run alongside the structural-facts checks. When
    empty, those checks are skipped — useful for unit tests that supply
    only StructuralFacts.
    """
    findings: list[Finding] = []
    findings.extend(_check_missing_references(facts))
    findings.extend(_check_unused_references(facts))
    findings.extend(_check_redefined_acronyms(facts.term_glossary.definitions))
    findings.extend(
        _check_undefined_acronym_usage(
            definitions=facts.term_glossary.definitions,
            usages=facts.term_glossary.usages,
        )
    )
    findings.extend(_check_inconsistent_sample_sizes(facts.numeric_claims))
    findings.extend(_check_broken_cross_references(facts.cross_references, sections))
    findings.extend(check_stat_consistency(sections))
    if markdown:
        findings.extend(run_checklist_checks(markdown=markdown, sections=sections))
    return findings


# ── Citations ──────────────────────────────────────────────────────────


def _check_missing_references(facts) -> list[Finding]:
    """Citations to keys that have no entry in the references list."""
    out: list[Finding] = []
    if not facts.citation_graph.references_section_ids:
        # No references section was identified — emit one finding rather
        # than a flood of "missing reference" findings.
        if facts.citation_graph.occurrences:
            out.append(
                Finding(
                    title="No references section identified",
                    severity="major",
                    confidence="high",
                    rationale=(
                        "The document contains in-text citations but no section "
                        "titled 'References', 'Bibliography', or similar was found. "
                        "Without a reference list, citations cannot be resolved."
                    ),
                    quotes=_quotes_from_occurrences(
                        facts.citation_graph.occurrences[:3]
                    ),
                    sections_involved=_unique(
                        o.section_id for o in facts.citation_graph.occurrences[:3]
                    ),
                )
            )
        return out

    defined = set(facts.citation_graph.references_defined.keys())
    by_key: dict[str, list[CitationOccurrence]] = defaultdict(list)
    for occ in facts.citation_graph.occurrences:
        by_key[occ.key.key].append(occ)

    for key in sorted(by_key):
        if key in defined:
            continue
        occs = by_key[key]
        out.append(
            Finding(
                title=f"Citation [{key}] used but not in references",
                severity="major",
                confidence="high",
                rationale=(
                    f"The citation key '{key}' is cited "
                    f"{len(occs)} time(s) in the document but has no matching "
                    f"entry in the references section."
                ),
                quotes=_quotes_from_occurrences(occs[:5]),
                sections_involved=_unique(o.section_id for o in occs),
            )
        )
    return out


def _check_unused_references(facts) -> list[Finding]:
    """Reference entries that are never cited from the body text."""
    if not facts.citation_graph.references_section_ids:
        return []
    cited = {occ.key.key for occ in facts.citation_graph.occurrences}
    unused = sorted(set(facts.citation_graph.references_defined) - cited)
    if not unused:
        return []
    return [
        Finding(
            title=f"{len(unused)} reference entr{'y' if len(unused) == 1 else 'ies'} never cited",
            severity="moderate",
            confidence="high",
            rationale=(
                f"The references section defines {len(unused)} entr{'y' if len(unused) == 1 else 'ies'} "
                f"that the body text never cites: {', '.join(unused[:10])}"
                + ("…" if len(unused) > 10 else "")
            ),
            sections_involved=list(facts.citation_graph.references_section_ids),
        )
    ]


# ── Acronyms ───────────────────────────────────────────────────────────


def _check_redefined_acronyms(definitions: list[TermDefinition]) -> list[Finding]:
    """An acronym defined more than once with DIFFERENT expansions."""
    by_term: dict[str, list[TermDefinition]] = defaultdict(list)
    for d in definitions:
        by_term[d.term].append(d)
    out: list[Finding] = []
    for term, defs in sorted(by_term.items()):
        unique_expansions = {_norm(d.expansion) for d in defs}
        if len(unique_expansions) <= 1:
            continue
        out.append(
            Finding(
                title=f"Acronym {term!r} defined inconsistently",
                severity="moderate",
                confidence="high",
                rationale=(
                    f"The acronym '{term}' is defined with "
                    f"{len(unique_expansions)} different expansions: "
                    + "; ".join(sorted(f'"{e}"' for e in unique_expansions))
                ),
                quotes=[
                    Quote(
                        section_id=d.section_id,
                        char_start=d.char_start,
                        char_end=d.char_end,
                        text=f"{d.expansion} ({d.term})",
                    )
                    for d in defs
                ],
                sections_involved=_unique(d.section_id for d in defs),
            )
        )
    return out


def _check_undefined_acronym_usage(
    *,
    definitions: list[TermDefinition],
    usages: list[TermUsage],
) -> list[Finding]:
    """An acronym used before its first definition (or never defined)."""
    # Collate first-definition position per term, by section id.
    # Section_id ordering ("sec_001", "sec_002", ...) is monotonic, so we
    # can compare as strings.
    first_def: dict[str, str] = {}
    for d in sorted(definitions, key=lambda x: (x.section_id, x.char_start)):
        first_def.setdefault(d.term, d.section_id)

    out: list[Finding] = []
    by_term_used_before: dict[str, list[TermUsage]] = defaultdict(list)
    for u in usages:
        if u.term not in first_def:
            # Never defined — caught below.
            continue
        if u.section_id < first_def[u.term]:
            by_term_used_before[u.term].append(u)

    for term in sorted(by_term_used_before):
        first_uses = by_term_used_before[term]
        out.append(
            Finding(
                title=f"Acronym {term!r} used before its definition",
                severity="minor",
                confidence="high",
                rationale=(
                    f"The acronym '{term}' is used in section "
                    f"{first_uses[0].section_id} but only defined later in "
                    f"section {first_def[term]}."
                ),
                quotes=[
                    Quote(
                        section_id=u.section_id,
                        char_start=u.char_start,
                        char_end=u.char_end,
                        text=u.term,
                    )
                    for u in first_uses[:3]
                ],
                sections_involved=_unique(u.section_id for u in first_uses),
            )
        )
    return out


# ── Numeric claims ─────────────────────────────────────────────────────


def _check_inconsistent_sample_sizes(claims: list[NumericClaim]) -> list[Finding]:
    """Different N values reported in different sections."""
    by_prefix: dict[str, set[str]] = defaultdict(set)  # "N" or "n" → set of values
    occurrences: dict[str, list[NumericClaim]] = defaultdict(list)
    for c in claims:
        if c.kind != "sample_size":
            continue
        # value is "N=50" or "n=250"; split it
        prefix, _, value = c.value.partition("=")
        by_prefix[prefix].add(value)
        occurrences[prefix].append(c)

    out: list[Finding] = []
    for prefix, values in sorted(by_prefix.items()):
        if len(values) <= 1:
            continue
        out.append(
            Finding(
                title=f"Inconsistent {prefix} values across sections",
                severity="major",
                confidence="high",
                rationale=(
                    f"The sample size '{prefix}' is reported with "
                    f"{len(values)} different values: "
                    + ", ".join(sorted(values, key=lambda v: int(v) if v.isdigit() else 0))
                ),
                quotes=[
                    Quote(
                        section_id=c.section_id,
                        char_start=c.char_start,
                        char_end=c.char_end,
                        text=c.raw,
                    )
                    for c in occurrences[prefix][:6]
                ],
                sections_involved=_unique(c.section_id for c in occurrences[prefix]),
            )
        )
    return out


# ── Cross-references ───────────────────────────────────────────────────


def _check_broken_cross_references(
    refs: list[CrossReference],
    sections: list[SectionRef],
) -> list[Finding]:
    """References to figures/tables/sections/equations with no matching anchor."""
    anchors = {
        "figure": find_figure_anchors(sections),
        "table": find_table_anchors(sections),
        "section": find_section_anchors(sections),
        "equation": find_equation_anchors(sections),
    }
    by_kind_and_target: dict[tuple[str, str], list[CrossReference]] = defaultdict(list)
    for r in refs:
        by_kind_and_target[(r.kind, r.target)].append(r)

    out: list[Finding] = []
    for (kind, target), occs in sorted(by_kind_and_target.items()):
        if target in anchors[kind]:
            continue
        # Equation anchors are the least reliably detected — downgrade severity.
        severity = "minor" if kind == "equation" else "moderate"
        out.append(
            Finding(
                title=f"Reference to {kind.capitalize()} {target} has no anchor",
                severity=severity,  # type: ignore[arg-type]
                confidence="high" if kind != "equation" else "medium",
                rationale=(
                    f"The text refers to {kind.capitalize()} {target} but no "
                    f"matching {kind} anchor was found in the document."
                ),
                quotes=[
                    Quote(
                        section_id=r.section_id,
                        char_start=r.char_start,
                        char_end=r.char_end,
                        text=r.raw,
                    )
                    for r in occs[:3]
                ],
                sections_involved=_unique(r.section_id for r in occs),
            )
        )
    return out


# ── Helpers ────────────────────────────────────────────────────────────


def _quotes_from_occurrences(occs: list[CitationOccurrence]) -> list[Quote]:
    return [
        Quote(
            section_id=o.section_id,
            char_start=o.char_start,
            char_end=o.char_end,
            text=o.key.raw,
        )
        for o in occs
    ]


def _unique(items: Iterable[str]) -> list[str]:
    """Order-preserving deduplicate."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _norm(s: str) -> str:
    """Normalise whitespace + lowercase for comparison."""
    return " ".join(s.lower().split())
