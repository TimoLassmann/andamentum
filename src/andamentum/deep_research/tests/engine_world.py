"""A small invented research world for the engine's end-to-end tests.

Everything is synthetic: two topics (the Apollo cache and SaltLog logging),
six scholarly works served in OpenAlex's shape, one of them retracted in
Crossref, one found only as a citation, and four web pages behind a SearxNG
endpoint. ``World.handler`` answers every HTTP request the engine makes;
``world_llm`` answers every language-model call of the new loop (plan, queries, choose, notes, reflect,
write) by reading the prompt the way a model would; ``world_decisions`` answers
the decision-model questions of the novelty check. No network, no real model.
The stand-ins at the end replace the reading, writing, verifying and ranking
modules for the tests of the loop alone.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import httpx
import pytest
from pydantic import BaseModel

from andamentum.core.decisions import DecisionModel
from andamentum.core.decisions.testing import (
    StubDecisions,
    choice_answer,
    noul_answer,
)

from andamentum.references import ReferenceHttp, Work

from .. import fusion
from ..ask import MODEL_ERRORS, Asker, BudgetExhausted, fence
from ..read import ReadDeps
from ..sources.base import SourceClient
from ..store import ResearchStore
from ..types import (
    Candidate,
    Finding,
    PageReading,
    Passage,
    RawFinding,
    SubQuestion,
    SubQuestionResult,
    VerifyFlag,
)
from ..verify import VerifyOutcome
from ..write import NO_EVIDENCE_LINE, SOURCES_HEADING, Report
from .llm_stub import ScriptedLLM

QUESTION = "How do the Apollo cache and SaltLog logging affect request latency?"
SUBQ_APOLLO = "How much does the Apollo cache reduce median request latency?"
SUBQ_SALTLOG = "How much overhead does SaltLog logging add per request?"
SEARXNG = "http://searx.test"
TOPICS = ("apollo", "saltlog")


@dataclass(frozen=True)
class Paper:
    openalex_id: str
    doi: str
    title: str
    author: str
    abstract: str
    cites: tuple[str, ...] = ()  # OpenAlex ids of works this one cites


PAPERS = (
    Paper(
        "W1001",
        "10.5555/apollo-a",
        "Apollo cache halves median latency in read-heavy services",
        "Jane Doe",
        "We studied read-heavy services. Enabling the Apollo cache reduced median "
        "request latency from 120 ms to 45 ms in every tested workload.",
    ),
    Paper(
        "W1002",
        "10.5555/apollo-b",
        "Latency effects of the Apollo caching layer",
        "Ravi Rowe",
        "In a replication across three services, the Apollo caching layer lowered "
        "median latency by about half.",
    ),
    Paper(
        "W1003",
        "10.5555/apollo-retracted",
        "Apollo cache removes all request latency",
        "Max Mustermann",
        "In one test the Apollo cache removed all measurable request latency from "
        "the service.",
    ),
    Paper(
        "W1004",
        "10.5555/saltlog-d",
        "SaltLog logging overhead per request",
        "Ada Byte",
        "Verbose SaltLog logging added roughly 3 ms of overhead to each request in "
        "the benchmark.",
    ),
    Paper(
        "W1005",
        "10.5555/saltlog-e",
        "Measuring SaltLog overhead in production",
        "Lin Quay",
        "In production traces, SaltLog added between 2 and 4 ms per request when "
        "verbose logging was on.",
    ),
)
# Found only by following citations of W1001.
FOLLOW_UP = Paper(
    "W1006",
    "10.5555/apollo-f",
    "A follow-up on Apollo cache latency",
    "Kim Lee",
    "A follow-up study confirmed that the Apollo cache cut median request latency "
    "roughly in half.",
    cites=("W1001",),
)
RETRACTED_DOI = "10.5555/apollo-retracted"

PAGES = {
    "https://apollo-docs.example/latency": (
        "Apollo cache latency guide",
        "This guide covers the Apollo cache.\n\nTurning on the Apollo cache brought "
        "median latency down from 120 ms to 50 ms on our staging cluster. The gain "
        "held for every read-heavy endpoint we tried, and the cache needed no "
        "tuning beyond the defaults.",
    ),
    "https://ops-notes.example/apollo": (
        "Ops notes: Apollo in production",
        "Notes from the operations team.\n\nOur measurements show the Apollo cache "
        "cut median latency by roughly 60 percent for read-heavy endpoints. Write "
        "paths saw no change, which matches what the cache is designed to do for "
        "reads only.",
    ),
    "https://saltlog-blog.example/overhead": (
        "What SaltLog costs",
        "A short post on logging costs.\n\nVerbose SaltLog logging costs about 3 ms "
        "per request on our hosts. With verbose mode off the cost fell below the "
        "noise of our measurements, so we keep it off in production.",
    ),
    "https://saltlog-faq.example/faq": (
        "SaltLog FAQ",
        "Frequently asked questions about SaltLog.\n\nSaltLog adds 2 to 4 "
        "milliseconds per request when verbose mode is enabled. The overhead comes "
        "from formatting each record, and batching the writes reduces it further.",
    ),
}


def _inverted(text: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


def _openalex(paper: Paper) -> dict[str, object]:
    return {
        "id": f"https://openalex.org/{paper.openalex_id}",
        "doi": f"https://doi.org/{paper.doi}",
        "title": paper.title,
        "publication_year": 2024,
        "primary_location": {"source": {"display_name": "Journal of Invented Systems"}},
        "open_access": {"oa_url": None},
        "type": "article",
        "ids": {},
        "is_retracted": False,
        "authorships": [{"author": {"display_name": paper.author}}],
        "abstract_inverted_index": _inverted(paper.abstract),
    }


def _topic(text: str) -> str:
    lowered = text.lower()
    return next((t for t in TOPICS if t in lowered), "")


@dataclass
class World:
    """The HTTP side of the world. ``searxng_down`` makes SearxNG answer 500."""

    searxng_down: bool = False
    requests: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = request.url
        host = url.host
        if url.path == "/robots.txt":
            return httpx.Response(404)
        if host == "api.openalex.org":
            return self._openalex(request)
        if host == "api.crossref.org":
            doi = url.path.removeprefix("/works/")
            message: dict[str, object] = {"DOI": doi}
            if doi == RETRACTED_DOI:
                message["updated-by"] = [
                    {
                        "type": "retraction",
                        "DOI": "10.5555/notice",
                        "source": "retraction-watch",
                    }
                ]
            return httpx.Response(200, json={"message": message})
        if host == "doi.org":
            return httpx.Response(200)
        if host == "searx.test":
            if self.searxng_down:
                return httpx.Response(500, text="searxng is down")
            if url.params.get("pageno") != "1":
                return httpx.Response(200, json={"results": []})
            results = [
                {"url": u, "title": title, "content": text[:160]}
                for u, (title, text) in PAGES.items()
            ]
            return httpx.Response(200, json={"results": results})
        page = PAGES.get(str(url))
        if page is not None:
            return httpx.Response(
                200, text=page[1], headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, text=f"no route for {url}")

    def _openalex(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        everything = (*PAPERS, FOLLOW_UP)
        if request.url.path.startswith("/works/W"):
            return httpx.Response(200, json={"referenced_works": []})
        flt = params.get("filter", "")
        if flt.startswith("cites:"):
            cited = flt.removeprefix("cites:")
            found = [p for p in everything if cited in p.cites]
        elif flt.startswith("openalex:"):
            wanted = set(flt.removeprefix("openalex:").split("|"))
            found = [p for p in everything if p.openalex_id in wanted]
        else:
            topic = _topic(params.get("search.title_abstract_keywords", ""))
            found = [p for p in PAPERS if topic and topic in p.title.lower()]
        return httpx.Response(
            200,
            json={
                "meta": {"next_cursor": None},
                "results": [_openalex(p) for p in found],
            },
        )

    def client(self, store: ResearchStore, *, offline: bool = False) -> SourceClient:
        """A source client over this world: no waiting, plain-text extraction.
        ``offline`` makes the client a frozen run uses: it answers from the
        store's cache only and never reaches the world."""

        async def no_sleep(_seconds: float) -> None:
            return None

        async def plain(body: bytes, _content_type: str, _url: str) -> str:
            return body.decode("utf-8")

        return SourceClient(
            store,
            client=None
            if offline
            else httpx.AsyncClient(
                transport=httpx.MockTransport(self.handler), follow_redirects=False
            ),
            offline=offline,
            intervals={
                name: 0.0
                for name in (
                    "openalex",
                    "crossref",
                    "europepmc",
                    "arxiv",
                    "semantic_scholar",
                    "web",
                )
            },
            sleep=no_sleep,
            extractor=plain,
        )


# -- the language model ------------------------------------------------------------


def holds(question: str, text: str) -> bool:
    """Whether a yes/no question (a decision-model request) holds of *text*."""
    lowered = question.lower()
    item = text.lower()
    if lowered.startswith("do the quotes state"):
        return True
    if lowered.startswith("do the two findings contradict"):
        return False
    if "advertisement" in lowered:
        return False
    if "measurement" in lowered:
        return bool(re.search(r"\d", item)) or "half" in item
    topic = _topic(lowered)
    return bool(topic) and topic in item


_FENCED = re.compile(r"(<{3,})\n.*?\n(>{3,})", re.DOTALL)
_QUERY_WORDS = ("latency", "benchmark", "measurement", "study")


def _fenced(prompt: str) -> str:
    found = re.search(r"(<{3,})\n(.*?)\n(>{3,})", prompt, re.DOTALL)
    return found.group(2) if found else prompt


def _outside_fences(prompt: str) -> str:
    return _FENCED.sub("", prompt)


def _line(prompt: str, label: str) -> str:
    found = re.search(rf"^{label}: (.+)$", prompt, re.MULTILINE)
    return found.group(1) if found else ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def _choose(instructions: str, prompt: str) -> dict[str, object]:
    topic = _topic(_line(prompt, "SUB-QUESTION"))
    limit = int(re.search(r"the (\d+) results most likely", instructions).group(1))  # type: ignore[union-attr]
    entries = re.findall(
        r"^\[(\d+)\] (.+?)(?=^\[\d+\] |\Z)", _fenced(prompt), re.MULTILINE | re.DOTALL
    )
    relevant = [int(n) for n, text in entries if topic and topic in text.lower()]
    return {"picks": relevant[:limit], "reserves": relevant[limit : 2 * limit]}


def _notes(prompt: str) -> dict[str, object]:
    topic = _topic(_outside_fences(prompt))
    findings = []
    for number, text in re.findall(
        r"^\[(\d+)\]\n<{3,}\n(.*?)\n>{3,}(?=\n\n\[\d+\]\n|\Z)",
        prompt,
        re.MULTILINE | re.DOTALL,
    ):
        for sentence in _sentences(text):
            if topic in sentence.lower() and re.search(r"\d|half", sentence):
                findings.append(
                    {
                        "passage": int(number),
                        "claim": sentence,
                        "quote": sentence.rstrip("."),
                    }
                )
                break
    return {"findings": findings}


def _reflect(prompt: str) -> dict[str, object]:
    topic = _topic(_line(prompt, "QUESTION"))
    claims = re.findall(r"^\[\d+\] claim: (.+)$", prompt, re.MULTILINE)
    sources = set(re.findall(r"^\s+source: (.+)$", prompt, re.MULTILINE))
    # Apollo needs three independent sources (the follow-up found only through
    # citations makes the third); SaltLog is settled by two.
    needed = 3 if topic == "apollo" else 2
    answered = len(sources) >= needed
    return {
        "answered": answered,
        "answer": f"Likely: {claims[0]}" if claims else "",
        "confidence": "high" if answered else "low",
        "gaps": [] if answered else [f"a further independent source on {topic}"],
        "next_queries": [] if answered else [f"{topic} replication study"],
    }


def _queries(prompt: str) -> dict[str, object]:
    topic = _topic(_line(prompt, "SUB-QUESTION"))
    done = len(re.findall(r"^- ", prompt.split("ALREADY RUN:")[-1], re.MULTILINE))
    done = done if "ALREADY RUN:" in prompt else 0
    return {
        "queries": [
            f"{topic} {_QUERY_WORDS[done % 4]}",
            f"{topic} {_QUERY_WORDS[(done + 1) % 4]}",
        ]
    }


def _write(prompt: str) -> dict[str, object]:
    """Group the findings under the sub-question they follow: a finding is an
    ``F<n>`` line and its ``claim:`` line (the writer's own layout), or one line
    ``F<n> claim: ...``."""
    sections: dict[str, list[dict[str, object]]] = {}
    label = ""
    current = ""
    for line in prompt.splitlines():
        head = re.match(r"^(?:SUB-QUESTION |#+\s*)?(Q\d+)\b", line)
        if head:
            label = head.group(1)
            sections.setdefault(label, [])
            continue
        finding = re.match(r"^(F\d+)\b(?:.*?claim: (.+))?$", line)
        if finding:
            current = finding.group(1)
            if finding.group(2):
                sections.setdefault(label or "Q1", []).append(
                    {"text": finding.group(2), "finding_ids": [current]}
                )
            continue
        claim = re.match(r"^claim: (.+)$", line)
        if claim and current:
            sections.setdefault(label or "Q1", []).append(
                {"text": claim.group(1), "finding_ids": [current]}
            )
            current = ""
    return {
        "sections": [
            {"subquestion": name, "sentences": sentences}
            for name, sentences in sections.items()
            if sentences
        ]
    }


def respond(instructions: str, prompt: str) -> dict[str, object]:
    """Answer one language-model call of the engine, from the prompt alone."""
    if instructions.startswith("You turn a question into the sub-questions"):
        return {
            "subquestions": [
                {"text": SUBQ_APOLLO, "kind": "topical"},
                {"text": SUBQ_SALTLOG, "kind": "topical"},
            ]
        }
    if instructions.startswith("You are a search query specialist"):
        return _queries(prompt)
    if instructions.startswith("You choose which search results"):
        return _choose(instructions, prompt)
    if instructions.startswith("You read passages of ONE document"):
        return _notes(prompt)
    if instructions.startswith("You judge the progress of research"):
        return _reflect(prompt)
    if "report" in instructions.lower() or "section" in instructions.lower():
        return _write(prompt)
    raise AssertionError(f"the world has no answer for: {instructions[:80]!r}")


def world_llm() -> ScriptedLLM:
    return ScriptedLLM(respond)


# -- the decision model ------------------------------------------------------------


def _state_text(state: object) -> str:
    """What a question is asked about: the state without the sub-question,
    which names the topic and would make every item look relevant."""
    if isinstance(state, dict):
        return json.dumps({k: v for k, v in state.items() if k != "question"})
    return str(state)


def world_decisions(
    *, judge: Callable[[str, str], bool] = holds
) -> tuple[DecisionModel, StubDecisions]:
    """A decision model that answers by *judge(question, state_text)*:
    p 0.9 for yes and 0.05 for no; choices pick the first sensible option."""

    def answer(state: object, questions: dict[str, dict[str, object]]):
        text = _state_text(state)
        answers = {}
        for key, question in questions.items():
            if question["type"] == "choice":
                described = question["criteria"]
                assert isinstance(described, dict)
                options = [str(name) for name in described]
                pick = "primary_study" if "primary_study" in options else "topical"
                answers[key] = choice_answer(options, pick, 0.8)
            else:
                yes = judge(str(question["instructions"]), text)
                answers[key] = noul_answer(0.9 if yes else 0.05)
        return answers

    stub = StubDecisions(answer)
    return stub.model(), stub


# -- stand-ins for the reading, writing, verifying and ranking modules -----------------
#
# The loop is tested apart from the modules it calls. These stand in for
# ``read.read_page``, ``write.write_report``, ``verify.verify_report``,
# ``read.unreadable_degraded`` and ``fusion.rank_works`` / ``fused_scores``: they
# honour the documented signatures and the ``PageReading`` / ``RawFinding`` /
# ``Report`` shapes, answer through the engine's ``Asker`` (so the calls are
# counted, budget-checked and logged) and emit the events those modules emit.


class NoteDraft(BaseModel):
    passage: int
    claim: str
    quote: str


class NotesDraft(BaseModel):
    findings: list[NoteDraft]


class WrittenSentence(BaseModel):
    text: str
    finding_ids: list[str]


class WrittenSection(BaseModel):
    subquestion: str
    sentences: list[WrittenSentence]


class WriteDraft(BaseModel):
    sections: list[WrittenSection]


STAND_IN_NOTES = (
    "You read passages of ONE document to find evidence for a question. Extract "
    "the findings: each a claim and the exact words that state it. Quote "
    "character for character."
)
STAND_IN_WRITE = (
    "You write the report for a question from its findings. One section per "
    "sub-question; every sentence cites the findings it rests on."
)


@dataclass
class StandInReader:
    """A ``read_page`` over the world: pages from ``PAGES``, abstracts for
    papers. ``unreadable`` lists substrings of a url or title that cannot be
    opened; ``snippet_only`` lists those whose page cannot be opened but whose
    search snippet is read; ``texts`` overrides a url's text; ``skip_notes`` lists substrings
    of a url or title whose page is read but gets no notes call; ``read`` lists
    what was read, in order, and ``peak`` the most pages read at once."""

    unreadable: tuple[str, ...] = ()
    snippet_only: tuple[str, ...] = ()
    skip_notes: tuple[str, ...] = ()
    texts: dict[str, str] = field(default_factory=dict)
    read: list[str] = field(default_factory=list)
    in_flight: int = 0
    peak: int = 0

    async def __call__(
        self,
        candidate: Candidate,
        *,
        subquestion: SubQuestion,
        queries: Sequence[str],
        deps: ReadDeps,
        round_number: int = 1,
    ) -> PageReading:
        self.read.append(candidate.work.work_id)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0)
            return await self._read(candidate, subquestion, deps)
        finally:
            self.in_flight -= 1

    async def _read(
        self,
        candidate: Candidate,
        subquestion: SubQuestion,
        deps: ReadDeps,
    ) -> PageReading:
        asker = deps.asker
        log = asker.log
        work = candidate.work
        if deps.fetched_bytes is not None and (
            deps.fetched_bytes() >= asker.budget.max_fetch_bytes
        ):
            raise BudgetExhausted("max_fetch_bytes")
        if work.doi == RETRACTED_DOI:
            if log is not None:
                log.event(
                    "excluded",
                    f"{work.title!r} excluded: retracted",
                    step="Read",
                    work_id=work.work_id,
                    reason="retracted",
                )
            return PageReading(
                work=work.model_copy(update={"retracted": "retracted"}),
                subquestion_id=subquestion.id,
                excluded="retracted",
            )
        label = work.url or work.title
        url = work.url or ""
        snippet = any(mark in label for mark in self.snippet_only)
        text = (
            ""
            if snippet
            else self.texts.get(url) or (PAGES[url][1] if url in PAGES else "")
        )
        via = "fetched" if text else "snippet" if snippet else "abstract"
        text = text or (
            candidate.snippet or work.abstract
            if snippet
            else work.abstract or candidate.snippet
        )
        if any(mark in label for mark in self.unreadable) or not text:
            if log is not None:
                log.event(
                    "unreadable",
                    f"{work.title!r}: nothing read",
                    step="Read",
                    work_id=work.work_id,
                    url=url,
                    title=work.title,
                    reasons=["HTTP 403"],
                )
            return PageReading(
                work=work.model_copy(update={"access": "unreadable"}),
                subquestion_id=subquestion.id,
                via="none",
                status=403,
                error="HTTP 403",
                failed_notes=[f"{label}: HTTP 403"],
            )
        passage = Passage.make(work_id=work.work_id, start=0, text=text)
        asker.store.add_passages([passage])
        if log is not None:
            log.event(
                "fetched",
                f"{work.title!r}: {len(text)} characters",
                step="Read",
                work_id=work.work_id,
                url=url,
                title=work.title,
                via=via,
                ok=True,
                chars=len(text),
                status=200,
                error="",
                truncated=False,
                text=text,
            )
            log.event(
                "notes_input",
                f"{work.title!r}: {len(text)} characters given to the notes",
                step="Notes",
                work_id=work.work_id,
                url=url,
                title=work.title,
                subquestion=subquestion.id,
                source_chars=len(text),
                window_chars=len(text),
                window_spans=[[0, len(text)]],
                text=text,
            )
        reading = PageReading(
            work=work.model_copy(
                update={
                    "access": "full_text"
                    if via == "fetched"
                    else "metadata"
                    if via == "snippet"
                    else "abstract"
                }
            ),
            subquestion_id=subquestion.id,
            via=via,  # type: ignore[arg-type]
            passages=[passage],
            status=200,
        )
        if any(mark in label for mark in self.skip_notes):
            return reading
        try:
            drafted = await asker.llm(
                step="Notes",
                instructions=STAND_IN_NOTES,
                output_type=NotesDraft,
                user_message=(f"QUESTION: {subquestion.text}\n\n[1]\n{fence(text)}"),
            )
        except MODEL_ERRORS as error:
            reading.degraded.append(
                f"notes for {work.title!r} failed ({type(error).__name__}: {error})"
            )
            return reading
        unverified = 0
        for note in drafted.findings:
            at = text.find(note.quote)
            verified = at >= 0
            unverified += not verified
            reading.findings.append(
                RawFinding(
                    subquestion_id=subquestion.id,
                    claim=note.claim,
                    quote=text[at : at + len(note.quote)] if verified else note.quote,
                    quote_verified=verified,
                    passage_id=passage.passage_id,
                    work_id=work.work_id,
                )
            )
            if log is not None:
                log.event(
                    "finding_noted",
                    f"{work.title!r}: {note.claim}",
                    step="Notes",
                    subquestion=subquestion.id,
                    work_id=work.work_id,
                    title=work.title,
                    claim=note.claim,
                    quote=note.quote,
                    verified=verified,
                )
        if log is not None:
            log.event(
                "notes",
                f"{work.title!r}: {len(reading.findings)} findings",
                step="Notes",
                work_id=work.work_id,
                url=url,
                subquestion=subquestion.id,
                findings=len(reading.findings),
                unverified=unverified,
            )
        return reading


async def stand_in_write(
    question: str,
    *,
    subquestions: Sequence[SubQuestion],
    answers: Mapping[str, SubQuestionResult],
    findings: Sequence[Finding],
    works: Mapping[str, Work],
    asker: Asker,
) -> Report:
    """One model call over every sub-question that has findings; a failed or
    spent model lists the findings' own claims."""
    lines: list[str] = []
    for subquestion in subquestions:
        own = [f for f in findings if f.subquestion_id == subquestion.id]
        if not own:
            continue
        lines += [f"{subquestion.id}: {subquestion.text}"]
        draft = answers[subquestion.id].answer
        if draft:
            lines.append(f"  draft answer: {draft}")
        lines += [f"{f.finding_id} claim: {f.claim}" for f in own]
    degraded: list[str] = []
    sentences: dict[str, list[tuple[str, list[str]]]] = {}
    known = {f.finding_id for f in findings}
    try:
        if lines:
            written = await asker.llm(
                step="Write",
                instructions=STAND_IN_WRITE,
                output_type=WriteDraft,
                user_message="\n".join(lines),
            )
            for section in written.sections:
                sentences[section.subquestion] = [
                    (s.text, [i for i in s.finding_ids if i in known])
                    for s in section.sentences
                ]
    except (BudgetExhausted, *MODEL_ERRORS) as error:
        degraded.append(f"the report could not be written ({error}); claims listed")
    body = [f"# Report: {question}"]
    for subquestion in subquestions:
        own = [f for f in findings if f.subquestion_id == subquestion.id]
        body += ["", f"## {subquestion.id}: {subquestion.text}", ""]
        if not own:
            body.append(NO_EVIDENCE_LINE)
            continue
        written_here = sentences.get(subquestion.id) or [
            (f.claim, [f.finding_id]) for f in own
        ]
        for text, ids in written_here:
            body.append(f"{text} [{', '.join(ids)}]")
    body += ["", SOURCES_HEADING]
    return Report(markdown="\n".join(body) + "\n", degraded=degraded)


async def stand_in_verify(
    report_markdown: str,
    *,
    findings: Sequence[Finding],
    passages: Mapping[str, Passage],
    works: Mapping[str, Work],
    http: ReferenceHttp,
) -> VerifyOutcome:
    known = {f.finding_id for f in findings}
    flags = [
        VerifyFlag(kind="unknown_finding", ref=ref)
        for ref in dict.fromkeys(re.findall(r"\bF\d+\b", report_markdown))
        if ref not in known
    ]
    return VerifyOutcome(report_markdown=report_markdown, flags=flags)


def stand_in_fused_scores(candidates: Mapping[str, Candidate]) -> dict[str, float]:
    return {
        work_id: sum(1.0 / (60 + rank) for rank in c.source_ranks.values())
        for work_id, c in candidates.items()
    }


def stand_in_rank_works(
    candidates: Mapping[str, Candidate],
    *,
    cited: Sequence[str],
    chosen: Sequence[str],
    excluded: Collection[str] = (),
) -> list[str]:
    scores = stand_in_fused_scores(candidates)

    def by_score(ids: Iterable[str]) -> list[str]:
        # ties keep the candidates' order: a work id is random per store
        return sorted(ids, key=lambda w: -scores.get(w, 0.0))

    ordered = [w for w in dict.fromkeys(cited) if w in candidates]
    ordered += by_score(
        w for w in dict.fromkeys(chosen) if w in candidates and w not in ordered
    )
    ordered += by_score(w for w in candidates if w not in ordered)
    return [w for w in ordered if w not in excluded]


def stand_in_unreadable_degraded(*, attempted: int, unreadable: int) -> list[str]:
    if attempted and unreadable / attempted > 0.5:
        return [f"{unreadable} of {attempted} pages could not be read"]
    return []


def install_stand_ins(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reader: StandInReader | None = None,
    writer: Callable[..., Awaitable[Report]] | None = None,
) -> StandInReader:
    """Replace the reading, writing, verifying and ranking functions the engine
    calls; returns the reader so a test can look at what was read."""
    from .. import choose as choose_module
    from .. import engine

    reader = reader or StandInReader()
    monkeypatch.setattr(engine, "read_page", reader)
    monkeypatch.setattr(engine, "write_report", writer or stand_in_write)
    monkeypatch.setattr(engine, "verify_report", stand_in_verify)
    monkeypatch.setattr(engine, "unreadable_degraded", stand_in_unreadable_degraded)
    monkeypatch.setattr(fusion, "fused_scores", stand_in_fused_scores)
    monkeypatch.setattr(fusion, "rank_works", stand_in_rank_works)
    assert choose_module.fusion is fusion
    return reader
