"""The v2 engine end to end, offline: web, scholarly and both, over an invented world.

Real adapters, store, screener, reader, writer and verifier; only the network
(a mock transport), the language model (``ScriptedLLM``) and the decision model
(``StubDecisions``) are stand-ins. See ``engine_world.py``.
"""

from __future__ import annotations

import pytest
from pydantic_graph.id_types import NodeID
from pydantic_graph.paths import DestinationMarker

from andamentum.core import fetch_gate, url_safety

from .. import research
from ..config import PRESETS, ResearchConfig
from ..graph import graph
from ..sources import SourceConfig
from ..store import ResearchStore
from ..types import Budget, Mode, ResearchResult
from ..verify import report_units
from ..write import CODE_LINE_PREFIXES, SOURCES_HEADING
from .engine_world import (
    QUESTION,
    RETRACTED_DOI,
    SEARXNG,
    World,
    respond,
    world_decisions,
    world_llm,
)
from .llm_stub import ScriptedLLM

SCHOLARLY_ONLY = ("openalex",)


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The world's hosts are invented; treat every name as a public address."""
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


async def _run(
    mode: Mode,
    *,
    world: World | None = None,
    decision: bool = False,
    budget: Budget | None = None,
    store: ResearchStore | None = None,
    config: ResearchConfig = PRESETS["baseline"],
) -> tuple[ResearchResult, ResearchStore, World]:
    world = world or World()
    store = store or ResearchStore.in_memory()
    decision_model = world_decisions()[0] if decision else None
    result = await research(
        QUESTION,
        mode=mode,
        model=world_llm().model,
        decision_model=decision_model,
        store=store,
        budget=budget or Budget(),
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=SCHOLARLY_ONLY),
        config=config,
        source_client=world.client(store),
    )
    return result, store, world


def _assert_evidence_is_sound(result: ResearchResult) -> None:
    """Every finding's quote is a verbatim slice of a stored passage, and every
    report sentence ends with finding ids that exist."""
    passages = {p.passage_id: p for p in result.passages}
    assert result.findings
    for finding in result.findings:
        assert finding.quote in passages[finding.passage_id].text
        assert passages[finding.passage_id].work_id == finding.work_id
    known = {f.finding_id for f in result.findings}
    body = result.report_markdown.split(SOURCES_HEADING)[0]
    sentences = [
        u for u in report_units(body) if not u.text.startswith(CODE_LINE_PREFIXES)
    ]
    assert sentences
    for unit in sentences:
        assert unit.finding_ids, unit.text
        assert set(unit.finding_ids) <= known
    assert not [f for f in result.verification if f.kind == "unsupported"]


@pytest.mark.parametrize("preset", ["baseline", "criteria"])
@pytest.mark.parametrize("decision", [False, True], ids=["llm", "decision-model"])
async def test_scholarly_mode_end_to_end(decision: bool, preset: str) -> None:
    config = PRESETS[preset]
    result, store, _ = await _run("scholarly", decision=decision, config=config)
    assert result.config_name == preset
    assert result.config_hash == config.config_hash()

    assert result.stop_reason == "covered"
    assert result.degraded == []
    assert len(result.subquestions) == 2
    assert all(c.covered for c in result.coverage)
    _assert_evidence_is_sound(result)

    # the retracted work was kept by screening, then excluded and never read
    retracted = [w for w in store.iter_works() if w.doi == RETRACTED_DOI]
    assert len(retracted) == 1 and retracted[0].retracted == "retracted"
    assert retracted[0].work_id in result.excluded
    assert retracted[0].work_id not in {w.work_id for w in result.works}
    assert retracted[0].work_id not in {f.work_id for f in result.findings}

    # citation expansion found the follow-up study, which then gave a finding
    follow_up = [w for w in result.works if w.doi == "10.5555/apollo-f"]
    assert follow_up and follow_up[0].work_id in {f.work_id for f in result.findings}

    # the run is in the store, finished, with its rows
    record = store.get_run(result.run_id)
    assert record is not None and record.finished_at is not None
    assert record.config["config_hash"] == config.config_hash()
    kinds = {i.kind for i in store.run_items(result.run_id)}
    assert {"subquestion", "candidate", "finding", "coverage"} <= kinds
    # only the criteria preset compiles criteria (and scores passages by them)
    assert ("criterion" in kinds) == (preset == "criteria")
    assert bool(result.criteria) == (preset == "criteria")
    if decision:
        steps = {d.step for d in store.decisions(result.run_id)}
        assert {"plan_kind", "screen", "claim_support"} <= steps
        assert ("passage" in steps) == (preset == "criteria")
    assert result.stats.llm_calls > 0
    assert result.stats.per_source.get("openalex", 0) > 0
    assert result.to_csl()


async def test_web_mode_end_to_end() -> None:
    result, _, world = await _run("web")

    assert result.stop_reason == "covered"
    assert result.degraded == []
    _assert_evidence_is_sound(result)
    assert all(w.url and "example" in w.url for w in result.works)
    # web mode follows no citations
    assert not [r for r in world.requests if "cites:" in str(r.url)]


async def test_both_mode_uses_web_and_scholarly_sources() -> None:
    result, _, _ = await _run("both", decision=True)

    assert result.stop_reason == "covered"
    _assert_evidence_is_sound(result)
    cited = {w.work_id for w in result.works if w.work_id in result.cited_work_ids()}
    kinds = {
        "web" if (w.url and not w.doi) else "scholarly"
        for w in result.works
        if w.work_id in cited
    }
    assert kinds == {"web", "scholarly"}


async def test_a_dead_source_degrades_the_run_but_does_not_empty_it() -> None:
    result, _, _ = await _run("both", world=World(searxng_down=True))

    assert any("web search failed" in d for d in result.degraded)
    assert result.findings  # the scholarly sources still answered
    _assert_evidence_is_sound(result)
    record_degraded = result.degraded
    assert record_degraded == sorted(set(record_degraded), key=record_degraded.index)


async def test_a_spent_language_model_budget_stops_the_run_with_its_reason() -> None:
    # plan (1) and two compiles (2) leave 3 calls: the write reserve (4) is reached
    result, store, _ = await _run("scholarly", budget=Budget(max_llm_calls=6))

    assert result.stop_reason == "budget:max_llm_calls"
    assert result.stats.llm_calls <= 6
    assert result.findings == []
    assert result.report_markdown.startswith("# ")
    record = store.get_run(result.run_id)
    assert record is not None and record.finished_at is not None


async def test_a_spent_decision_budget_stops_screening_with_its_reason() -> None:
    result, _, _ = await _run(
        "scholarly", decision=True, budget=Budget(max_decision_calls=4)
    )
    assert result.stop_reason == "budget:max_decision_calls"
    assert result.stats.decision_calls <= 4


async def test_the_round_cap_stops_an_uncoverable_search() -> None:
    # reading one work a round cannot cover a sub-question, and only one
    # round is allowed, so the cap decides
    result, _, _ = await _run("web", budget=Budget(max_rounds=1, read_per_round=1))
    assert result.stop_reason == "budget:max_rounds"
    assert not all(c.covered for c in result.coverage)


async def test_a_second_run_reads_the_first_runs_cache() -> None:
    store = ResearchStore.in_memory()
    first, _, world = await _run("scholarly", store=store)
    sent = len(world.requests)
    second, _, _ = await _run("scholarly", store=store, world=world)
    assert second.stop_reason == first.stop_reason
    assert second.stats.cache_hits > 0
    # search and metadata answers came from the store: few new requests
    assert len(world.requests) - sent < sent


async def test_a_failing_model_degrades_the_plan_instead_of_failing() -> None:
    def no_plan(instructions: str, prompt: str) -> dict[str, object]:
        if instructions.startswith("You turn a question into the sub-questions"):
            return {"subquestions": "not a list"}
        return respond(instructions, prompt)

    world = World()
    store = ResearchStore.in_memory()
    result = await research(
        QUESTION,
        mode="scholarly",
        model=ScriptedLLM(no_plan).model,
        store=store,
        sources=SourceConfig(scholarly=SCHOLARLY_ONLY),
        source_client=world.client(store),
    )
    assert any(d.startswith("planning failed") for d in result.degraded)
    assert [s.text for s in result.subquestions] == [QUESTION]


def _successors(node_id: str) -> set[str]:
    """The nodes *node_id* can route to, through any decision in between."""
    found: set[str] = set()
    for path in graph.edges_by_source.get(NodeID(node_id), []):
        for item in path.items:
            if not isinstance(item, DestinationMarker):
                continue
            target = graph.nodes[item.destination_id]
            branches = getattr(target, "branches", None)
            if branches is None:
                found.add(item.destination_id)
                continue
            found |= {d.id for b in branches for d in b.destinations}
    return found


def test_the_topology_is_the_specified_one() -> None:
    assert _successors("Plan") == {"Search"}
    assert _successors("Search") == {"Screen"}
    assert _successors("Screen") == {"Expand", "Read"}
    assert _successors("Expand") == {"Screen"}
    assert _successors("Read") == {"Coverage"}
    assert _successors("Coverage") == {"Search", "Write"}
    assert _successors("Write") == {"Verify"}
    assert _successors("Verify") == {"Finish"}
    assert _successors("Finish") == {"__end__"}


async def test_a_source_client_of_another_store_is_refused() -> None:
    client = World().client(ResearchStore.in_memory())
    with pytest.raises(ValueError, match="must belong"):
        await research(
            QUESTION,
            mode="scholarly",
            model=world_llm().model,
            store=ResearchStore.in_memory(),
            source_client=client,
        )


async def test_a_failed_run_is_recorded_in_the_store_before_it_raises() -> None:
    def broken(instructions: str, prompt: str) -> dict[str, object]:
        if instructions.startswith("You write search queries"):
            raise RuntimeError("a bug in the engine, not a model failure")
        return respond(instructions, prompt)

    store = ResearchStore.in_memory()
    with pytest.raises(RuntimeError):
        await research(
            QUESTION,
            mode="scholarly",
            model=ScriptedLLM(broken).model,
            store=store,
            sources=SourceConfig(scholarly=SCHOLARLY_ONLY),
            source_client=World().client(store),
        )
    (record,) = store.list_runs()
    assert record.finished_at is not None
    assert record.degraded and record.degraded[0].startswith("run failed")
