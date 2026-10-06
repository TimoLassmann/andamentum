"""A small invented research world for the engine's end-to-end tests.

Everything is synthetic: two topics (the Apollo cache and SaltLog logging),
six scholarly works served in OpenAlex's shape, one of them retracted in
Crossref, one found only as a citation, and four web pages behind a SearxNG
endpoint. ``World.handler`` answers every HTTP request the engine makes;
``world_llm`` answers every language-model call by reading the prompt the way
a model would; ``world_decisions`` answers the decision-model questions. No
network, no real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from andamentum.core.decisions import DecisionModel
from andamentum.core.decisions.testing import (
    StubDecisions,
    choice_answer,
    noul_answer,
)

from ..sources.base import SourceClient
from ..store import ResearchStore
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

_CRITERIA = {
    "apollo": [
        ("Is the result about the Apollo cache?", "must", 3),
        ("Does the result report a latency measurement?", "should", 2),
        ("Is the result an advertisement?", "exclude", 1),
    ],
    "saltlog": [
        ("Is the result about SaltLog logging?", "must", 3),
        ("Does the result report an overhead measurement?", "should", 2),
        ("Is the result an advertisement?", "exclude", 1),
    ],
}


def holds(question: str, text: str) -> bool:
    """Whether a criterion (or any yes/no question the engine asks) holds of *text*."""
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


def _block(prompt: str) -> str:
    found = re.search(r"<<<\n(.*?)\n>>>", prompt, re.DOTALL)
    return found.group(1) if found else prompt


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def respond(instructions: str, prompt: str) -> dict[str, object]:
    """Answer one language-model call of the engine, from the prompt alone."""
    if instructions.startswith("You split a research question"):
        return {
            "subquestions": [
                {"text": SUBQ_APOLLO, "kind": "topical"},
                {"text": SUBQ_SALTLOG, "kind": "topical"},
            ]
        }
    if instructions.startswith("You write the screening rubric"):
        topic = _topic(prompt.split("Sub-question", 1)[-1])
        return {
            "criteria": [
                {
                    "text": text,
                    "kind": kind,
                    "weight": weight,
                    "applies_to": "candidate",
                }
                for text, kind, weight in _CRITERIA[topic]
            ]
        }
    if instructions.startswith("You write search queries"):
        done = prompt.count("\n- ")
        words = ("latency", "benchmark", "measurement", "study")
        labels = re.findall(r"^(Q\d+): (.+)$", prompt, re.MULTILINE)
        return {
            "queries": [
                {"subquestion": label, "query": f"{_topic(text)} {words[done % 4]}"}
                for label, text in labels
            ]
        }
    if instructions.startswith("You screen one search result"):
        item = _block(prompt)
        answers = [
            {"label": label, "answer": "yes" if holds(text, item) else "no"}
            for label, text in re.findall(r"^(C\d+): (.+)$", prompt, re.MULTILINE)
        ]
        result: dict[str, object] = {"answers": answers}
        if "Source types:" in prompt:
            result["source_type"] = "primary_study"
        return result
    if instructions.startswith("You check one decision"):
        return {"keep": "Decision: kept" in prompt}
    if instructions.startswith("You choose which passages"):
        count = len(re.findall(r"^\[\d+\]$", prompt, re.MULTILINE))
        return {"passages": list(range(1, min(count, 3) + 1))}
    if instructions.startswith("You extract findings"):
        topic = _topic(prompt.split("\n", 1)[0])
        findings = []
        for number, text in re.findall(
            r"^\[(\d+)\]\n<<<\n(.*?)\n>>>(?=\n\n\[\d+\]\n|\Z)",
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
                            "criteria": [],
                        }
                    )
                    break
        return {"findings": findings}
    if instructions.startswith("You write one section"):
        found = re.findall(
            r"^(F\d+).*\n<<<\n(?:source: .*\n)?claim: (.+)$", prompt, re.MULTILINE
        )
        return {
            "sentences": [
                {"text": claim, "finding_ids": [finding_id]}
                for finding_id, claim in found
            ]
        }
    if instructions.startswith("You answer one yes/no question"):
        question = prompt.split("\n", 1)[0].removeprefix("QUESTION: ")
        items = re.findall(r"^\[(\d+)\]$", prompt, re.MULTILINE)
        return {
            "verdicts": [
                {"item": int(n), "answer": "yes" if holds(question, "") else "no"}
                for n in items
            ]
        }
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
