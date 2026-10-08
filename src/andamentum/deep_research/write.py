"""WRITE: the report, from the findings table and nothing else.

ONE model call over all sub-questions that have usable findings. The prompt
carries the weighing rules (best-supported answer first, source quality,
agreement of independent origins, the question's own time scope, real
disagreement only, an unverified quote is weaker); the model returns sentences,
each with the ids of the findings that support it. The code appends the
``[F12]``-style ids, so every sentence of the report ends with them by
construction, and a sentence naming a finding that does not exist is sent back
for repair. Everything else is built in code from the run's records:

- a title and a section per sub-question, in order;
- for a sub-question without findings, a fixed line saying so (no model call
  for it; when no sub-question has findings there is no model call at all);
- a "Sources" section listing the works behind the cited findings, with the
  link or DOI that ``verify`` later checks.

Findings of a retracted work are never offered to the model.

There is no plainer fallback: the report is model-written or the run fails. A
section must hold at least one sentence that is not only citations; the output
validator rejects an empty section by name and the model repairs it. A section
still empty after the repairs is written by its own model call, for that
sub-question only. If that fails too (or the budget is spent), ``write_report``
raises ``ReportWriterError`` naming the sub-question and the reason. The engine
reserves ``1 + number of sub-questions`` language-model calls for this step.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from andamentum.references import Work

from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .origin import independent_sources, origin_key, origin_label
from .types import Finding, SubQuestion, SubQuestionResult

NO_EVIDENCE_LINE = "No evidence could be read for this question."
SOURCES_HEADING = "## Sources"
# Lines written by code that carry no finding id; verification does not expect one.
CODE_LINE_PREFIXES = (NO_EVIDENCE_LINE,)
# Findings put to the model for one sub-question: one origin at a time.
MAX_FINDINGS_PER_SECTION = 30

_CITATION = re.compile(r"\[\s*F\d+(?:\s*[,;]\s*F\d+)*\s*\]")


class ReportWriterError(RuntimeError):
    """The report could not be written: a section has no model-written sentence
    after the repairs and its own call. Names the sub-question and the reason."""

    def __init__(self, subquestion: SubQuestion, reason: str) -> None:
        super().__init__(
            f"the report writer could not write the section for sub-question "
            f"{subquestion.id} ({subquestion.text.strip()!r}): {reason}"
        )
        self.subquestion_id = subquestion.id
        self.reason = reason


class Report(BaseModel):
    """The report and what went into it."""

    markdown: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    excluded_finding_ids: list[str] = Field(default_factory=list)


class DraftSentence(BaseModel):
    text: str
    finding_ids: list[str] = Field(default_factory=list)


class SectionDraft(BaseModel):
    subquestion: str
    # Required and non-empty: with a default of [] the model's first (flattened)
    # answer was "repaired" by adding an empty list, so no repair ever wrote prose.
    sentences: list[DraftSentence] = Field(min_length=1)


class ReportDraft(BaseModel):
    sections: list[SectionDraft] = Field(min_length=1)


def write_reserve(subquestions: int) -> int:
    """Language-model calls kept for the report: the joint call, and one call
    per sub-question for a section that is still empty after the repairs."""
    return 1 + subquestions


WRITE_INSTRUCTIONS = (
    "You write a research report from numbered findings, and only from them. "
    "The findings are grouped under sub-questions (Q1, Q2, ...). Return one "
    "section per sub-question that has findings, with subquestion set to its id. "
    "Each section is a list of short, plain sentences. Each sentence states one "
    "thing the findings say and lists in finding_ids the ids of every finding "
    "that says it (for example F3 or F3, F7). Do not write ids or brackets "
    "inside the sentence text. Every section holds at least one sentence, and "
    "a section is a list of sentences, never a single text. The answer has "
    "this shape: "
    '{"sections": [{"subquestion": "Q1", "sentences": [{"text": "First '
    'sentence.", "finding_ids": ["F1", "F2"]}, {"text": "Second sentence.", '
    '"finding_ids": ["F3"]}]}]}\n'
    "How to weigh the findings:\n"
    "1. Give the best-supported answer first, in the first sentence of the "
    "section. Commit to one answer and say how sure the evidence makes you in "
    "words (for example 'likely', 'reported by one source', 'several independent "
    "sources agree'). A draft answer from the research is given for each "
    "question: keep it only where findings support it, and drop or correct it "
    "where they do not.\n"
    "2. Weigh sources by quality: primary, official and encyclopedic sources "
    "(the organisation, person, publisher, database or reference work the answer "
    "comes from) over aggregators, listings, forums, content farms and pages "
    "written to rank in search.\n"
    "3. Findings from independent origins that agree make an answer stronger; "
    "several pages of one origin count as one. The origin of each finding is "
    "shown.\n"
    "4. Respect the time scope of the question. A figure for another year, "
    "edition, version or date than the one asked about is not a conflict; use "
    "the finding that matches the scope, and mention the other only if it helps.\n"
    "5. Report a disagreement only when it is real and the weighing above does "
    "not settle it. Then give the answer first, and afterwards say that sources "
    "disagree and how.\n"
    "6. A finding marked 'quote not verified' is weaker: its quote could not be "
    "found in the page. Prefer a verified finding, and say so when the answer "
    "rests on an unverified one alone.\n"
    "7. Never state anything the findings do not say: no numbers, names, dates "
    "or causes of your own. Do not describe how the findings were found. If the "
    "findings do not answer a question, say what they do show and leave the "
    "rest out.\n" + UNTRUSTED
)


def _usable(finding: Finding, works: Mapping[str, Work]) -> bool:
    work = works.get(finding.work_id)
    return work is None or work.retracted != "retracted"


def _number(finding: Finding) -> int:
    return int(finding.finding_id[1:])


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
    return sorted(chosen, key=_number)


def citation(finding_ids: Sequence[str]) -> str:
    """``[F3, F7]``: the citation group the report puts at the end of a sentence."""
    return f"[{', '.join(finding_ids)}]"


def _sentence(text: str, finding_ids: Sequence[str]) -> str:
    clean = _CITATION.sub("", text).strip()
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"\s+([,;:.])", r"\1", clean).rstrip(" .;,")
    return f"{clean} {citation(finding_ids)}."


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


def _finding_block(finding: Finding, works: Mapping[str, Work]) -> str:
    work = works.get(finding.work_id)
    details = []
    if work is not None:
        if work.title:
            details.append(f"source: {work.title}")
        details.append(f"origin: {origin_label(work)}")
        if work.year:
            details.append(f"year: {work.year}")
        if work.venue:
            details.append(f"venue: {work.venue}")
    head = f"{finding.finding_id}" + (
        "" if finding.quote_verified else " (quote not verified)"
    )
    body = "\n".join(
        [*(["; ".join(details)] if details else []), f"claim: {finding.claim}"]
        + [f'quote: "{finding.quote}"']
    )
    return f"{head}\n{fence(body)}"


def _section_prompt(
    subquestion: SubQuestion,
    findings: Sequence[Finding],
    *,
    answer: str,
    works: Mapping[str, Work],
) -> str:
    known = {f.work_id: works[f.work_id] for f in findings if f.work_id in works}
    origins = independent_sources(findings, known)
    lines = [
        f"SUB-QUESTION {subquestion.id}: {subquestion.text}",
        f"Independent origins among the findings below: {origins}",
    ]
    if answer.strip():
        lines += [
            "Draft answer from the research (unverified; keep it only where the "
            "findings support it):",
            fence(answer.strip()),
        ]
    lines += ["FINDINGS:", *(_finding_block(f, works) for f in findings)]
    return "\n".join(lines)


def _section_for(label: str, ids: Sequence[str]) -> str | None:
    """The sub-question id a section label names: the label itself, or an id it
    starts with ("Q2: the text")."""
    label = label.strip()
    if label in ids:
        return label
    for sub_id in sorted(ids, key=len, reverse=True):
        if (
            label.startswith(sub_id)
            and not label[len(sub_id) : len(sub_id) + 1].isalnum()
        ):
            return sub_id
    return None


async def _draft(
    question: str,
    sections: Sequence[tuple[SubQuestion, list[Finding]]],
    *,
    answers: Mapping[str, SubQuestionResult],
    works: Mapping[str, Work],
    asker: Asker,
) -> dict[str, list[tuple[str, list[str]]]]:
    """Sentences by sub-question id, from one model call."""
    ids = [sq.id for sq, _ in sections]
    offered = {f.finding_id for _, fs in sections for f in fs}

    def check(draft: ReportDraft) -> ReportDraft:
        problems: list[str] = []
        seen: set[str] = set()
        for section in draft.sections:
            sub_id = _section_for(section.subquestion, ids)
            if sub_id is None:
                problems.append(f"section {section.subquestion!r} is not one of {ids}")
                continue
            usable_sentences = 0
            for sentence in section.sentences:
                if _CITATION.sub("", sentence.text).strip():
                    usable_sentences += 1
                else:
                    problems.append(
                        f"a sentence of {sub_id} has no text besides citations"
                    )
                if not sentence.finding_ids:
                    problems.append(
                        f"a sentence of {sub_id} lists no finding ids: "
                        f"{sentence.text[:60]!r}"
                    )
                unknown = sorted(set(sentence.finding_ids) - offered)
                if unknown:
                    problems.append(
                        f"these ids are not findings you were given: {unknown}"
                    )
            if usable_sentences:
                seen.add(sub_id)
            else:
                problems.append(
                    f"section {sub_id} has no sentence; write at least one "
                    f"sentence for it from its findings"
                )
        missing = [i for i in ids if i not in seen]
        if missing:
            problems.append(f"no section for {missing}; write one for each")
        if problems:
            raise ModelRetry("; ".join(problems))
        return draft

    user_message = f"QUESTION: {question}\n\n" + "\n\n".join(
        _section_prompt(
            sq,
            fs,
            answer=answers[sq.id].answer if sq.id in answers else "",
            works=works,
        )
        for sq, fs in sections
    )
    draft = await asker.llm(
        step="Write",
        instructions=WRITE_INSTRUCTIONS,
        output_type=ReportDraft,
        user_message=user_message,
        validators=[check],
    )
    by_section: dict[str, list[tuple[str, list[str]]]] = {}
    for section in draft.sections:
        sub_id = _section_for(section.subquestion, ids)
        if sub_id is None:
            continue
        for s in section.sentences:
            sentence_ids = list(dict.fromkeys(s.finding_ids))
            if _CITATION.sub("", s.text).strip():
                by_section.setdefault(sub_id, []).append(
                    (_sentence(s.text, sentence_ids), sentence_ids)
                )
    return by_section


async def _write_alone(
    question: str,
    subquestion: SubQuestion,
    findings: list[Finding],
    *,
    answers: Mapping[str, SubQuestionResult],
    works: Mapping[str, Work],
    asker: Asker,
) -> list[tuple[str, list[str]]]:
    """The sentences of one section, from a model call for that sub-question
    only. Raises ``ReportWriterError`` when it cannot get them."""
    try:
        got = await _draft(
            question,
            [(subquestion, findings)],
            answers=answers,
            works=works,
            asker=asker,
        )
    except BudgetExhausted as spent:
        raise ReportWriterError(
            subquestion, f"the language-model budget is spent ({spent})"
        ) from spent
    except MODEL_ERRORS as error:
        raise ReportWriterError(
            subquestion, f"the model failed ({type(error).__name__}: {error})"
        ) from error
    sentences = got.get(subquestion.id)
    if not sentences:
        raise ReportWriterError(subquestion, "the model returned no usable sentence")
    return sentences


async def write_report(
    question: str,
    *,
    subquestions: Sequence[SubQuestion],
    answers: Mapping[str, SubQuestionResult],
    findings: Sequence[Finding],
    works: Mapping[str, Work],
    asker: Asker,
) -> Report:
    """The report for *question*, written only from *findings*, in one model call.

    *answers* maps a sub-question id to the research's last judgement (the draft
    answer the writer may keep where findings support it). *works* maps work id
    to work (titles, sites, retraction state, the Sources section). Every section
    with findings is model-written; one the model cannot write raises
    ``ReportWriterError``. Findings of retracted works are never offered to the
    model."""
    usable = [f for f in findings if _usable(f, works)]
    excluded = [f.finding_id for f in findings if not _usable(f, works)]
    by_id = {f.finding_id: f for f in usable}

    sections: list[tuple[SubQuestion, list[Finding]]] = []
    for subquestion in subquestions:
        own = [f for f in usable if f.subquestion_id == subquestion.id]
        if own:
            sections.append((subquestion, _spread(own, works)))

    written: dict[str, list[tuple[str, list[str]]]] = {}
    if sections:
        try:
            written = await _draft(
                question, sections, answers=answers, works=works, asker=asker
            )
        except (BudgetExhausted, *MODEL_ERRORS):
            # The joint call failed (its error is in the run log); each section
            # is then written by its own call below.
            written = {}
    for subquestion, own in sections:
        if written.get(subquestion.id):
            continue
        written[subquestion.id] = await _write_alone(
            question,
            subquestion,
            own,
            answers=answers,
            works=works,
            asker=asker,
        )

    out = [f"# {question.strip()}", ""]
    cited: list[str] = []
    chosen_by_id = {sq.id: fs for sq, fs in sections}
    for subquestion in subquestions:
        out += [f"## {subquestion.text.strip()}", ""]
        if subquestion.id not in chosen_by_id:
            out += [NO_EVIDENCE_LINE, ""]
            continue
        sentences = written[subquestion.id]
        out += [" ".join(t for t, _ in sentences), ""]
        for _, ids in sentences:
            cited += ids

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
    )
