"""Shared builders for the ``write`` tests: synthetic project files and a scripted model."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from andamentum.compose.brief import BriefDraft
from andamentum.compose.evidence import RawFact, RawFacts
from andamentum.compose.harness.checks.numbers import NumericClaims
from andamentum.compose.harness.draft import CitableWork
from andamentum.compose.harness.draft.cite import CiteVerdict
from andamentum.compose.harness.draft.claims import ClaimLabel, ClaimLabels
from andamentum.compose.schemas import Outline, ReaderQuestion, Roles, SectionPlan
from andamentum.compose.write import SectionBody

USAGE = RunUsage(input_tokens=11, output_tokens=7)
ROLES = Roles(
    drafter="stub:drafter",
    critic="stub:critic",
    judge="stub:judge",
    reader="stub:reader",
)

NOTES = (
    "# SaltLog pilot notes\n\n"
    "The pilot enrolled 48 children in three schools. Teachers logged every "
    "session in the SaltLog app.\n"
)
SCORES = "group,baseline,followup\ntreatment,41.2,53.8\ncontrol,40.9,43.1\n"
WORK_TITLE = "Structured session logs and the fidelity of school interventions"
WORK_ABSTRACT = (
    "Structured logging of reading sessions improves fidelity of delivery in "
    "school interventions."
)
WORK_CSL: dict[str, object] = {
    "id": "saltlog2020",
    "type": "article-journal",
    "title": WORK_TITLE,
    "author": [{"family": "Okafor", "given": "Ada"}],
    "issued": {"date-parts": [[2020]]},
    "container-title": "Journal of Invented Studies",
    "DOI": "10.5555/saltlog.2020.1",
    "abstract": WORK_ABSTRACT,
}
# The invented paper the cite loop finds, reads and cites for the Introduction's
# background sentence (the stand-in tools below).
WORK_PAPER = (
    f"# {WORK_TITLE}\n\n{WORK_ABSTRACT} In the invented trial, teachers in "
    "twelve schools logged every session for a school year.\n"
)

SECTIONS = ("Abstract", "Introduction", "Methods", "Results", "Discussion")
BODIES = {
    "Abstract": "We piloted structured session logging. Reading scores rose in the treatment group [[f2]].",
    "Introduction": (
        "Structured logging of reading sessions improves delivery fidelity.\n\n"
        "Whether this carries over to reading outcomes is not yet known."
    ),
    "Methods": (
        "The pilot enrolled 48 children in three schools [[f1]].\n\n"
        "- Teachers logged every session.\n- Scores were collected at baseline and follow-up."
    ),
    "Results": (
        "The treatment group's mean score rose from 41.2 to 53.8 [[f2]]. "
        "[TO FILL: control group sample size]"
    ),
    "Discussion": "The gain suggests that logging helps teachers deliver sessions consistently.",
}


def make_project(root: Path) -> tuple[Path, Path]:
    """``(notes.md, results folder)`` of the synthetic SaltLog project."""
    notes = root / "notes.md"
    notes.write_text(NOTES, encoding="utf-8")
    results = root / "results"
    results.mkdir()
    (results / "scores.csv").write_text(SCORES, encoding="utf-8")
    return notes, results


def _section_of(message: str) -> str:
    found = re.search(r"^SECTION: (.+)$", message, re.MULTILINE)
    assert found, message[:200]
    return found.group(1)


def _facts(message: str) -> RawFacts:
    if "SaltLog pilot notes" in message:
        return RawFacts(
            facts=[
                RawFact(
                    statement="The pilot enrolled 48 children in three schools.",
                    quote="The pilot enrolled 48 children in three schools.",
                )
            ]
        )
    if "scores.csv" in message:
        return RawFacts(
            facts=[
                RawFact(
                    statement="The treatment group's mean score rose from 41.2 to 53.8.",
                    quote="treatment,41.2,53.8",
                )
            ]
        )
    return RawFacts(facts=[RawFact(statement=WORK_ABSTRACT, quote=WORK_ABSTRACT)])


def _labels(message: str) -> ClaimLabels:
    labels = []
    for line in message.splitlines():
        number, _, text = line.partition(": ")
        kind = (
            "interpretation"
            if "suggests" in text or "carries over" in text
            else "background"
            if "improves" in text
            else "method"
        )
        labels.append(ClaimLabel(number=int(number), kind=kind))
    return ClaimLabels(labels=labels)


def default_script() -> dict[type, Callable[[str], BaseModel]]:
    def outline(_: str) -> Outline:
        return Outline(
            sections=[
                SectionPlan(
                    title=title,
                    purpose=f"{title} of the paper",
                    key_points=[],
                    fact_ids=ids,
                    target_words=words,
                )
                for title, ids, words in (
                    ("Abstract", ["f2"], 200),
                    ("Introduction", ["f3"], 700),
                    ("Methods", ["f1"], 700),
                    ("Results", ["f2"], 700),
                    ("Discussion", [], 700),
                )
            ]
        )

    return {
        BriefDraft: lambda _: BriefDraft(
            genre="article",
            title="SaltLog pilot",
            audience="reading researchers",
            purpose="report the SaltLog pilot",
            key_messages=[
                "Structured logging improves fidelity",
                "Reading scores rose",
            ],
            must_include=[],
            constraints=[],
            reader_questions=[
                ReaderQuestion(question="What was piloted?", expected_answer="SaltLog"),
                ReaderQuestion(question="How many children?", expected_answer="48"),
            ],
        ),
        RawFacts: _facts,
        Outline: outline,
        SectionBody: lambda m: SectionBody(body=BODIES[_section_of(m)]),
        ClaimLabels: _labels,
        NumericClaims: lambda _: NumericClaims(claims=[]),
        CiteVerdict: _cite_verdict,
    }


def _cite_verdict(message: str) -> CiteVerdict:
    """The SaltLog paper supports the sentences that say logging improves fidelity."""
    sentence = next(
        (line for line in message.splitlines() if line.startswith("SENTENCE: ")), ""
    )
    if WORK_TITLE in message and "improves" in sentence:
        return CiteVerdict(supports="yes", quote=WORK_ABSTRACT)
    return CiteVerdict(supports="no")


class Script:
    """A model stub that answers by output type; records every call."""

    def __init__(self, handlers: dict[type, Callable[[str], BaseModel]]) -> None:
        self.handlers = handlers
        self.calls: list[tuple[type, str]] = []

    async def __call__(
        self, model: str, **kwargs: object
    ) -> tuple[BaseModel, RunUsage]:
        output_type = kwargs["output_type"]
        message = str(kwargs["user_message"])
        assert isinstance(output_type, type)
        self.calls.append((output_type, message))
        return self.handlers[output_type](message), USAGE

    def count(self, output_type: type) -> int:
        return sum(1 for t, _ in self.calls if t is output_type)


def install_script(
    monkeypatch: pytest.MonkeyPatch,
    handlers: dict[type, Callable[[str], BaseModel]] | None = None,
) -> Script:
    script = Script(handlers or default_script())
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", script)
    monkeypatch.setattr(
        "andamentum.compose.harness.checks.numbers.run_agent_with_usage", script
    )
    return script


def no_cite_tools(store_dir: Path):  # noqa: ANN201
    """An ``open_cite`` for ``run_draft`` whose finder finds nothing: the cite
    loop runs over a real document store in *store_dir* and cites nothing."""
    from contextlib import asynccontextmanager

    from andamentum.compose.harness.draft.cite import CiteTools
    from andamentum.document_store import Store

    async def nothing(sentence: str, *, paragraph: str) -> list:  # noqa: ARG001
        return []

    async def unreadable(work, *, query: str):  # noqa: ANN001, ANN202, ARG001
        raise AssertionError("nothing is found, so nothing is read")

    async def unverifiable(work) -> str:  # noqa: ANN001
        raise AssertionError("nothing is found, so nothing is verified")

    @asynccontextmanager
    async def open_cite(**_kwargs: object):  # noqa: ANN202
        path = store_dir / "papers.db"
        async with await Store.open(path) as papers:
            yield CiteTools(
                library=None,
                finder=nothing,
                read=unreadable,
                verify=unverifiable,
                papers=papers,
                store_path=path,
            )

    return open_cite


def saltlog_cite_tools(
    store_dir: Path,
    *,
    csl: dict[str, object] = WORK_CSL,
    paper: str = WORK_PAPER,
    asked: list[str] | None = None,
):  # noqa: ANN201
    """An async context manager giving cite tools that find, read and verify one
    invented paper (*csl*, text *paper*) over a real document store in
    *store_dir*; every sentence the finder is asked about goes into *asked*."""
    from contextlib import asynccontextmanager

    from andamentum.compose.harness.draft.cite import CiteTools
    from andamentum.deep_research import WorkText
    from andamentum.document_store import Store
    from andamentum.references import Work

    paper_work = Work(
        doi=str(csl["DOI"]),
        title=str(csl["title"]),
        year=2020,
        authors=["Ada Okafor"],
    )

    async def finder(sentence: str, *, paragraph: str) -> list[Work]:  # noqa: ARG001
        if asked is not None:
            asked.append(sentence)
        return [paper_work]

    async def read(work: Work, *, query: str) -> WorkText:  # noqa: ARG001
        return WorkText(work=work, text=paper, window=paper, via="fetched")

    async def verify(work: Work) -> CitableWork:  # noqa: ARG001
        return CitableWork(csl=csl, abstract="")

    @asynccontextmanager
    async def opened():  # noqa: ANN202
        path = store_dir / "papers.db"
        async with await Store.open(path) as papers:
            yield CiteTools(
                library=None,
                finder=finder,
                read=read,
                verify=verify,
                papers=papers,
                store_path=path,
            )

    return opened()
