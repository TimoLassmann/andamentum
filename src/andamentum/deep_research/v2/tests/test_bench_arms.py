"""The two answer arms: how each result is reduced to what the metrics read.

The legacy pipeline and v2's ``research()`` are replaced by functions that
return invented results; nothing here reaches a model or the network.
"""

from __future__ import annotations

from typing import Any

import pytest

from andamentum.deep_research.models import (
    EvidenceItem,
    EvidenceReport,
    FetchedPage,
    ResearchErrors,
)
from andamentum.deep_research.models import ResearchResult as LegacyResult
from andamentum.deep_research.v2.bench import arms
from andamentum.deep_research.v2.bench.arms import (
    LegacyArm,
    V2Arm,
    map_legacy,
    map_v2,
    report_claims,
)
from andamentum.deep_research.v2.sources.base import SourceConfig
from andamentum.deep_research.v2.types import (
    Budget,
    Finding,
    Passage,
    ResearchResult,
    RunStats,
)
from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.store import ResearchStore
from andamentum.references import Work

from .engine_world import QUESTION, World, world_decisions, world_llm

PAGE_A = (
    "Filler about kilns. " * 200
    + "SaltLog cut brine loss by forty percent in the Apollo kiln trial. "
    + "More filler about pottery. " * 200
)
PAGE_B = "A page about something else entirely, with nothing on salt. " * 30


def legacy_result(*, evidence_items: list[EvidenceItem]) -> LegacyResult:
    return LegacyResult(
        output=EvidenceReport(
            evidence_summary="SaltLog works. See https://kilns.example.org/trial.",
            key_findings=["SaltLog cut brine loss by forty percent."],
            sources=["https://kilns.example.org/trial", "https://other.example.org/x"],
            evidence_items=evidence_items,
            total_searches_performed=2,
            total_pages_fetched=2,
            iterations_required=1,
            degraded=True,
            degraded_reason="3 of 8 fetches failed",
        ),
        fetched_pages=[
            FetchedPage(
                url="https://kilns.example.org/trial/",
                title="Kiln trial",
                content=PAGE_A,
                word_count=10,
                relevance_score=0.9,
                is_relevant=True,
            ),
            FetchedPage(
                url="https://pots.example.org/other",
                title="Other",
                content=PAGE_B,
                word_count=10,
                relevance_score=0.1,
                is_relevant=False,
            ),
        ],
        iterations=1,
        searches=2,
        pages_fetched=2,
        verification={
            "total_cited": 2,
            "verified_count": 1,
            "verified": [],
            "unverified": [],
            "accessed_not_cited": [],
            "verification_rate": 0.5,
        },
        errors=ResearchErrors(search_errors=0, fetch_errors=3),
    )


def test_legacy_claims_are_checked_against_the_window_of_their_source_page() -> None:
    result = legacy_result(
        evidence_items=[
            EvidenceItem(
                finding="SaltLog cut brine loss by forty percent",
                source_url="https://kilns.example.org/trial",
                source_title="Kiln trial",
            ),
            EvidenceItem(
                finding="A claim citing a page that was never fetched",
                source_url="https://never.example.org/fetched",
                source_title="Ghost",
            ),
        ]
    )
    out = map_legacy(result, wall_s=4.5)
    [claim] = out.claims
    assert "cut brine loss by forty percent" in claim.evidence
    assert len(claim.evidence) <= 2400  # a window, not the whole page
    assert out.n_unsourced == 1
    assert out.cost.wall_s == 4.5 and out.cost.llm_calls is None
    assert out.degraded == ["3 of 8 fetches failed"]
    assert out.stop_reason == "iterations:1"


def test_legacy_key_findings_are_the_claims_when_there_are_no_evidence_items() -> None:
    out = map_legacy(legacy_result(evidence_items=[]), wall_s=1.0)
    [claim] = out.claims
    assert "forty percent" in claim.evidence
    assert out.n_unsourced == 0


def test_legacy_urls_and_works_are_the_cited_sources_then_the_other_pages() -> None:
    out = map_legacy(legacy_result(evidence_items=[]), wall_s=1.0)
    assert out.urls == [
        "https://kilns.example.org/trial",
        "https://other.example.org/x",
    ]
    assert [w.url for w in out.works] == [
        "https://kilns.example.org/trial",
        "https://other.example.org/x",
        "https://pots.example.org/other",
    ]
    assert out.works[0].title == "Kiln trial"  # the title of the page that was fetched


def finding(n: int, quote: str, work_id: str) -> Finding:
    return Finding(
        finding_id=f"F{n}",
        subquestion_id="S1",
        claim=f"claim {n}",
        quote=quote,
        passage_id=Passage.make(work_id=work_id, start=0, text=quote).passage_id,
        work_id=work_id,
    )


def v2_result(report: str) -> ResearchResult:
    return ResearchResult(
        run_id="r1",
        question="Does SaltLog work?",
        mode="scholarly",
        works=[
            Work(work_id="w1", doi="10.5555/trial.1", title="Trial one of SaltLog"),
            Work(work_id="w2", url="https://kilns.example.org/two", title="Two"),
            Work(work_id="w3", title="Kept but never cited"),
        ],
        findings=[
            finding(1, "SaltLog cut brine loss by forty percent.", "w1"),
            finding(2, "No effect was seen in the small kiln.", "w2"),
        ],
        report_markdown=report,
        stats=RunStats(
            llm_calls=7,
            input_tokens=100,
            output_tokens=50,
            decision_calls=40,
            wall_s=12.5,
        ),
        stop_reason="covered",
        degraded=["openalex: 429"],
    )


REPORT = """# Brine loss

SaltLog cut brine loss in the Apollo trial [F1]. The small kiln saw no effect [F2].
A sentence that cites nothing at all appears here.
Two findings back this sentence [F1, F2].
This cites a finding that does not exist [F9].
- A list item stating a cited result with enough words [F1]

| table | row |
Short one.

More at https://kilns.example.org/two and 10.5555/trial.1.
"""


def test_report_sentences_are_paired_with_the_quotes_they_cite() -> None:
    claims, unsourced = report_claims(v2_result(REPORT))
    assert [c.claim for c in claims] == [
        "SaltLog cut brine loss in the Apollo trial.",
        "The small kiln saw no effect.",
        "Two findings back this sentence.",
        "A list item stating a cited result with enough words",
    ]
    assert claims[0].evidence == "SaltLog cut brine loss by forty percent."
    assert claims[2].evidence == (
        "SaltLog cut brine loss by forty percent.\nNo effect was seen in the small kiln."
    )
    # the uncited sentence, the unknown finding and the link line have nothing to
    # check against; headings, table rows and fragments are not claims at all
    assert unsourced == 3


def test_v2_result_maps_to_works_cost_and_links() -> None:
    out = map_v2(v2_result(REPORT))
    assert [w.work_id for w in out.works] == ["w1", "w2", "w3"]
    assert out.cost.llm_calls == 7 and out.cost.decision_calls == 40
    assert out.cost.input_tokens == 100 and out.cost.wall_s == 12.5
    assert out.degraded == ["openalex: 429"] and out.stop_reason == "covered"
    assert "https://kilns.example.org/two" in out.urls
    assert "https://doi.org/10.5555/trial.1" in out.urls
    assert len(out.urls) == len(set(out.urls))


async def test_the_v2_arm_calls_research_with_the_spec_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def research(question: str, **kwargs: Any) -> ResearchResult:
        seen.update(question=question, **kwargs)
        return v2_result(REPORT)

    monkeypatch.setattr(arms, "research", research)
    sources = SourceConfig(searxng_url="http://127.0.0.1:4070")
    arm = V2Arm(
        mode="both",
        model="ollama:stub",
        decision_model="nimble:9b",
        sources=sources,
        budget=Budget(max_rounds=1),
    )
    out = await arm.run("Does SaltLog work?")
    assert arm.name == "v2-both" and not arm.returns_pages
    assert seen == {
        "question": "Does SaltLog work?",
        "mode": "both",
        "model": "ollama:stub",
        "decision_model": "nimble:9b",
        "store": None,  # nothing persists
        "budget": Budget(max_rounds=1),
        "sources": sources,
        "source_client": None,
    }
    assert out.cost.llm_calls == 7


async def test_the_v2_arm_runs_the_real_engine_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()
    store = ResearchStore.in_memory()
    arm = V2Arm(
        mode="scholarly",
        model=world_llm().model,
        decision_model=world_decisions()[0],
        sources=SourceConfig(scholarly=("openalex",)),
        store=store,
        source_client=World().client(store),
    )
    out = await arm.run(QUESTION)
    assert out.stop_reason == "covered" and out.degraded == []
    assert out.claims and all(c.evidence for c in out.claims)
    assert out.works and out.works[0].doi
    assert any(u.startswith("https://doi.org/") for u in out.urls)
    assert out.cost.llm_calls and out.cost.decision_calls


async def test_the_legacy_arm_calls_run_research_and_times_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def run_research(question: str, **kwargs: Any) -> LegacyResult:
        seen.update(question=question, **kwargs)
        return legacy_result(evidence_items=[])

    monkeypatch.setattr(arms, "run_research", run_research)
    out = await LegacyArm(model="ollama:stub", searxng_url="http://127.0.0.1:4070").run(
        "Does SaltLog work?"
    )
    assert seen["model"] == "ollama:stub"
    assert seen["searxng_url"] == "http://127.0.0.1:4070"
    assert seen["decision_model"] is None
    assert out.cost.wall_s >= 0 and out.cost.llm_calls is None
