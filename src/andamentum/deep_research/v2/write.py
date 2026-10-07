"""WRITE: the report, from the findings table and nothing else.

One language-model call per sub-question turns that sub-question's findings
into sentences. The model returns each sentence with the ids of the findings
that support it; the code appends the ids, so *every sentence of the report ends
with ``[F12]``-style ids* by construction, and a sentence naming a finding that
does not exist is sent back for repair. Everything else in the report is built
in code from the run's records:

- a title and a section per sub-question, in order;
- for a sub-question without findings, a fixed line saying so; for one that is
  not covered, a fixed line stating how many independent sources there are;
- a "Disagreements" section stating each conflict from the two findings'
  claims, each with its id;
- a "Sources" section listing the works behind the cited findings, with the
  link or DOI that ``verify`` later checks.

Findings of a retracted work are never given to the model. The sentences the
model returns are checked, afterwards, by ``verify.py`` against the quotes.

When the writer cannot run for a section (the language-model budget is spent,
or the model fails), that section lists its findings' own claims, each with its
id, and ``Report.degraded`` says so. The claims were extracted with quotes that
code verified, so the fallback still cites and still passes through VERIFY; it
is plainer, never invented.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from andamentum.references import Work

from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .coverage import origin_key
from .types import Conflict, Coverage, Criterion, Finding, SubQuestion

NO_EVIDENCE_LINE = "No supporting evidence was found for this question."
LIMITED_EVIDENCE_PREFIX = "Evidence is limited:"
DISAGREEMENTS_HEADING = "## Disagreements"
SOURCES_HEADING = "## Sources"
# Lines written by code that carry no finding id; verification does not expect one.
CODE_LINE_PREFIXES = (NO_EVIDENCE_LINE, LIMITED_EVIDENCE_PREFIX)

# At most this many findings are put to the model for one sub-question: the
# first of each origin in turn, so no single source fills the prompt.
MAX_FINDINGS_PER_SECTION = 30

_CITATION = re.compile(r"\[\s*F\d+(?:\s*[,;]\s*F\d+)*\s*\]")


class DraftSentence(BaseModel):
    text: str
    finding_ids: list[str] = Field(min_length=1)


class SectionDraft(BaseModel):
    sentences: list[DraftSentence]


_WRITE_INSTRUCTIONS = (
    "You write one section of a research report, only from the numbered findings "
    "you are given. When the findings answer the question, the first sentence "
    "gives that answer directly. Write short, plain sentences. Each sentence states one thing "
    "the findings say, and lists in finding_ids the ids of every finding that says "
    "it (for example F3 or F3, F7). Never state anything the findings do not say: "
    "no numbers, names or causes of your own. When findings agree, say so in one "
    "sentence citing all of them. When they differ in scope or conditions, say "
    "how. Do not describe how the findings were found. Do not write ids or "
    "brackets inside the sentence text. Follow the criteria listed, if any, as "
    "the facets to cover, and skip a facet the findings do not touch. " + UNTRUSTED
)


class Report(BaseModel):
    """The report and what went into it."""

    markdown: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    excluded_finding_ids: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)


def _usable(finding: Finding, works: Mapping[str, Work]) -> bool:
    work = works.get(finding.work_id)
    return work is None or work.retracted != "retracted"


def _spread(findings: Sequence[Finding], works: Mapping[str, Work]) -> list[Finding]:
    """At most ``MAX_FINDINGS_PER_SECTION`` findings, one origin at a time."""
    by_origin: dict[str, list[Finding]] = {}
    for finding in findings:
        work = works.get(finding.work_id)
        key = origin_key(work) if work is not None else f"work:{finding.work_id}"
        by_origin.setdefault(key, []).append(finding)
    chosen: list[Finding] = []
    depth = 0
    while len(chosen) < min(len(findings), MAX_FINDINGS_PER_SECTION):
        added = False
        for group in by_origin.values():
            if depth < len(group) and len(chosen) < MAX_FINDINGS_PER_SECTION:
                chosen.append(group[depth])
                added = True
        if not added:
            break
        depth += 1
    return sorted(chosen, key=lambda f: int(f.finding_id[1:]))


def citation(finding_ids: Sequence[str]) -> str:
    """``[F3, F7]``: the citation group the report puts at the end of a sentence."""
    return f"[{', '.join(finding_ids)}]"


def _sentence(text: str, finding_ids: Sequence[str]) -> str:
    clean = _CITATION.sub("", text).strip()
    clean = re.sub(r"\s+", " ", clean).rstrip(" .;,")
    return f"{clean} {citation(finding_ids)}."


async def _draft_section(
    subquestion: SubQuestion,
    findings: Sequence[Finding],
    *,
    criteria: Sequence[Criterion],
    works: Mapping[str, Work],
    asker: Asker,
) -> list[tuple[str, list[str]]]:
    known = {f.finding_id for f in findings}
    lines = []
    for finding in findings:
        work = works.get(finding.work_id)
        source = f"source: {work.title}, {work.year}\n" if work and work.title else ""
        facets = (
            f" [criteria: {', '.join(finding.criterion_ids)}]"
            if finding.criterion_ids
            else ""
        )
        lines.append(
            f"{finding.finding_id}{facets}\n"
            + fence(f'{source}claim: {finding.claim}\nquote: "{finding.quote}"')
        )
    wanted = "\n".join(f"- {c.id}: {c.text}" for c in criteria if c.kind != "exclude")

    def known_ids(draft: SectionDraft) -> SectionDraft:
        unknown = sorted(
            {i for s in draft.sentences for i in s.finding_ids if i not in known}
        )
        if unknown:
            raise ModelRetry(
                f"these ids are not findings of this section: {unknown}; "
                f"use only {sorted(known, key=lambda i: int(i[1:]))}"
            )
        return draft

    draft = await asker.llm(
        instructions=_WRITE_INSTRUCTIONS,
        output_type=SectionDraft,
        user_message=(
            f"QUESTION: {subquestion.text}\n"
            + (f"CRITERIA:\n{wanted}\n" if wanted else "")
            + "\nFINDINGS:\n"
            + "\n".join(lines)
        ),
        validators=[known_ids],
    )
    sentences: list[tuple[str, list[str]]] = []
    for s in draft.sentences:
        ids = list(dict.fromkeys(s.finding_ids))
        if _CITATION.sub("", s.text).strip():
            sentences.append((_sentence(s.text, ids), ids))
    return sentences


def _claims(findings: Sequence[Finding]) -> list[tuple[str, list[str]]]:
    """Each finding's own claim as a sentence citing it: the writer's fallback."""
    return [(_sentence(f.claim, [f.finding_id]), [f.finding_id]) for f in findings]


def _link(work: Work) -> str:
    if work.doi:
        return f"https://doi.org/{work.doi}"
    if work.arxiv_id:
        return f"https://arxiv.org/abs/{work.arxiv_id}"
    if work.pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{work.pmid}/"
    return work.url or ""


def _source_line(work: Work, finding_ids: Sequence[str]) -> str:
    authors = ", ".join(work.authors[:3]) + (" et al." if len(work.authors) > 3 else "")
    credit = " ".join(p for p in (authors, f"({work.year})" if work.year else "") if p)
    parts = [
        f"{credit}." if credit and not credit.endswith(".") else credit,
        (work.title or work.work_id).rstrip(".") + ".",
        f"{work.venue}." if work.venue else "",
        _link(work),
    ]
    return f"- {citation(finding_ids)} " + " ".join(p for p in parts if p)


def _disagreement(conflict: Conflict, by_id: Mapping[str, Finding]) -> str | None:
    first, second = (by_id.get(i) for i in conflict.finding_ids)
    if first is None or second is None:
        return None
    one, two = (
        _CITATION.sub("", f.claim).strip().rstrip(" .") for f in (first, second)
    )
    return (
        f"- Sources disagree. One reports: {one} [{first.finding_id}]; "
        f"another reports: {two} [{second.finding_id}]."
    )


async def write_report(
    question: str,
    *,
    subquestions: Sequence[SubQuestion],
    criteria: Mapping[str, Sequence[Criterion]],
    findings: Sequence[Finding],
    conflicts: Sequence[Conflict],
    coverage: Sequence[Coverage],
    works: Mapping[str, Work],
    asker: Asker,
) -> Report:
    """The report for *question*, written only from *findings*.

    *works* maps work id to work (for titles, retraction state and the
    Sources section). A section the writer cannot write (spent budget, failed
    model) lists its findings' claims instead, and ``degraded`` names it.
    """
    usable = [f for f in findings if _usable(f, works)]
    excluded = [f.finding_id for f in findings if not _usable(f, works)]
    by_id = {f.finding_id: f for f in usable}
    covered = {c.subquestion_id: c for c in coverage}

    out = [f"# {question.strip()}", ""]
    cited: list[str] = []
    degraded: list[str] = []
    for subquestion in subquestions:
        out += [f"## {subquestion.text.strip()}", ""]
        own = [f for f in usable if f.subquestion_id == subquestion.id]
        if not own:
            out += [NO_EVIDENCE_LINE, ""]
            continue
        chosen = _spread(own, works)
        try:
            sentences = await _draft_section(
                subquestion,
                chosen,
                criteria=criteria.get(subquestion.id, ()),
                works=works,
                asker=asker,
            )
        except BudgetExhausted as spent:
            sentences = _claims(chosen)
            degraded.append(
                f"section {subquestion.id} lists its findings' claims: {spent}"
            )
        except MODEL_ERRORS as error:
            sentences = _claims(chosen)
            degraded.append(
                f"section {subquestion.id} lists its findings' claims: the writer "
                f"failed ({type(error).__name__}: {error})"
            )
        out += [
            " ".join(t for t, _ in sentences) if sentences else NO_EVIDENCE_LINE,
            "",
        ]
        for _, ids in sentences:
            cited += ids
        status = covered.get(subquestion.id)
        if status is not None and not status.covered:
            out += [
                f"{LIMITED_EVIDENCE_PREFIX} {status.independent_sources} "
                f"independent source{'' if status.independent_sources == 1 else 's'} "
                "found for this question.",
                "",
            ]

    lines = [d for c in conflicts if (d := _disagreement(c, by_id)) is not None]
    if lines:
        out += [DISAGREEMENTS_HEADING, "", *lines, ""]
        for conflict in conflicts:
            if all(i in by_id for i in conflict.finding_ids):
                cited += conflict.finding_ids

    cited = list(dict.fromkeys(cited))
    by_work: dict[str, list[str]] = {}
    for finding_id in cited:
        by_work.setdefault(by_id[finding_id].work_id, []).append(finding_id)
    if by_work:
        out += [SOURCES_HEADING, ""]
        for work_id, ids in by_work.items():
            work = works.get(work_id)
            out.append(
                _source_line(work, ids)
                if work is not None
                else f"- {citation(ids)} {work_id}"
            )
        out.append("")
    return Report(
        markdown="\n".join(out).rstrip() + "\n",
        cited_finding_ids=cited,
        excluded_finding_ids=excluded,
        degraded=degraded,
    )
