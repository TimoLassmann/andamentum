"""The answer arm: how a result is reduced to what the metrics read.

``research()`` is replaced by a function that returns invented results; nothing here reaches a model or the network.
"""

from __future__ import annotations

from typing import Any

import pytest

from andamentum.deep_research.v2.bench import arms
from andamentum.deep_research.v2.bench.arms import (
    V2Arm,
    map_v2,
    report_claims,
)
from andamentum.deep_research.v2.sources.base import SourceConfig
from andamentum.deep_research.v2.types import (
    Budget,
    Finding,
    Passage,
    ResearchResult,
    RunLog,
    RunStats,
)
from andamentum.references import Work


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
    assert out.cost.llm_calls == 7 and out.cost.decision_calls == 0
    assert out.cost.input_tokens == 100 and out.cost.wall_s == 12.5
    assert out.degraded == ["openalex: 429"] and out.stop_reason == "covered"
    assert "https://kilns.example.org/two" in out.urls
    assert "https://doi.org/10.5555/trial.1" in out.urls
    assert len(out.urls) == len(set(out.urls))


async def test_the_v2_arm_calls_research_with_the_new_signature(
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
        sources=sources,
        budget=Budget(max_rounds=1),
    )
    out = await arm.run("Does SaltLog work?")
    assert arm.name == "v2-both"
    assert seen == {
        "question": "Does SaltLog work?",
        "mode": "both",
        "model": "ollama:stub",
        "store": None,  # nothing persists
        "budget": Budget(max_rounds=1),
        "sources": sources,
        "http_cache": None,
    }
    assert out.cost.llm_calls == 7 and out.cost.decision_calls == 0
    assert out.result is not None and out.result.passages == []  # slimmed
    assert out.log is not None  # the run log travels to the cell log files


async def test_the_v2_arm_takes_a_name_and_a_shared_http_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research.v2.store import ResearchStore

    seen: dict[str, Any] = {}

    async def research(question: str, **kwargs: Any) -> ResearchResult:
        seen.update(kwargs)
        return v2_result(REPORT)

    monkeypatch.setattr(arms, "research", research)
    cache = ResearchStore.in_memory()
    arm = V2Arm(
        mode="web",
        model="ollama:stub",
        sources=SourceConfig(),
        http_cache=cache,
        name="new-engine",
    )
    await arm.run("Does SaltLog work?")
    assert arm.name == "new-engine"
    assert seen["http_cache"] is cache and seen["store"] is None


def test_the_run_log_is_kept_out_of_the_registry_dump() -> None:
    result = v2_result(REPORT).model_copy(update={"log": RunLog(run_id="r1")})
    out = map_v2(result)
    assert out.log is not None
    assert "log" not in out.model_dump()
    assert out.result is not None and out.result.log.events == []
