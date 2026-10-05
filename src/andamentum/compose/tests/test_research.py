"""Research in compose: findings and passages become sources and facts.

``deep_research.v2.research`` is stubbed at its package boundary
(``_research_helpers.FakeResearch``); the engine's own tests cover the engine.
The compose side runs for real: the graph, the evidence worker, the store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose import compose_bundle, write_document
from andamentum.compose import store as bundle_store
from andamentum.compose.evidence import stored_source_path
from andamentum.compose.genres import GENRES, Genre
from andamentum.compose.render import sources_body
from andamentum.compose.research import (
    ResearchSetup,
    ResearchSetupError,
    attribute,
    citation_label,
    degraded_notes,
    research_facts,
    research_mode,
    research_sources,
)
from andamentum.compose.schemas import Roles
from andamentum.deep_research.v2 import SourceConfig

from ._research_helpers import (
    BLOG_QUOTE,
    STUDY_QUOTE,
    FakeResearch,
    blog,
    make_result,
    study,
)
from .test_integration import SOURCE, Script

WEB = ResearchSetup(sources=SourceConfig(searxng_url="http://searx.test"))


# -- the mode follows the genre -----------------------------------------------------


def test_the_mode_is_the_genres_unless_the_setup_overrides_it() -> None:
    chosen = {name: research_mode(g, WEB) for name, g in GENRES.items()}
    assert chosen == {
        "email": "web",
        "memo": "web",
        "howto": "web",
        "proposal": "both",
        "report": "scholarly",
        "article": "scholarly",
        "grant": "scholarly",
    }
    forced = ResearchSetup(sources=SourceConfig(), mode="both")
    assert research_mode(GENRES["memo"], forced) == "both"


def test_a_genre_file_without_a_mode_asks_for_the_web_unless_it_cites() -> None:
    base = GENRES["email"].model_dump()
    plain = Genre.model_validate({**base, "research_mode": None, "cites": False})
    citing = Genre.model_validate({**base, "research_mode": None, "cites": True})
    assert plain.research == "web" and citing.research == "both"


# -- sources and facts from findings --------------------------------------------------


def test_each_work_with_a_finding_is_a_source_holding_its_passages(
    tmp_path: Path,
) -> None:
    result = make_result()
    sources = research_sources(result, bundle_dir=tmp_path, first_number=3)
    assert [s.id for s in sources] == ["src3", "src4"]
    assert [s.kind for s in sources] == ["research", "research"]
    assert [s.work_id for s in sources] == ["w1", "w2"]
    first = stored_source_path(tmp_path, sources[0]).read_text(encoding="utf-8")
    assert first.startswith("# Structured session logs")
    assert STUDY_QUOTE in first
    assert sources[1].path == "https://apollo-docs.example/latency"
    # nothing machine-specific reaches the label or the path
    assert all(str(tmp_path) not in (s.path + s.label) for s in sources)


def test_a_fact_is_a_finding_whose_quote_is_located_in_the_source_text(
    tmp_path: Path,
) -> None:
    result = make_result()
    source = research_sources(result, bundle_dir=tmp_path, first_number=1)[1]
    text = stored_source_path(tmp_path, source).read_text(encoding="utf-8")
    facts, dropped = research_facts(result, source, text, first_number=7)
    assert dropped == []
    assert [f.id for f in facts] == ["f7"]
    fact = facts[0]
    assert fact.quote == BLOG_QUOTE
    assert text[fact.start : fact.end] == BLOG_QUOTE
    assert fact.statement == attribute(
        "The Apollo cache halves median latency.", title=source.title
    )
    assert "not this study" in fact.statement


def test_only_the_works_a_caller_found_usable_become_sources(tmp_path: Path) -> None:
    sources = research_sources(
        make_result(), bundle_dir=tmp_path, first_number=1, only={"w1"}
    )
    assert [s.work_id for s in sources] == ["w1"]


def test_a_finding_whose_quote_is_not_in_the_text_is_dropped_and_said(
    tmp_path: Path,
) -> None:
    result = make_result()
    source = research_sources(result, bundle_dir=tmp_path, first_number=1)[1]
    facts, dropped = research_facts(result, source, "an unrelated text", first_number=1)
    assert facts == [] and "not found" in dropped[0]


def test_the_label_is_a_citation_a_reader_can_use() -> None:
    assert citation_label(study()) == (
        "Ada Okafor, Ravi Rowe, Lin Quay et al. (2020). Structured session logs and "
        "the fidelity of school interventions. Journal of Invented Studies. "
        "https://doi.org/10.5555/saltlog.2020.1"
    )
    assert citation_label(blog()) == (
        "Apollo cache latency guide. https://apollo-docs.example/latency"
    )


def test_the_sources_list_prints_the_citation(tmp_path: Path) -> None:
    sources = research_sources(make_result(), bundle_dir=tmp_path, first_number=1)
    body = sources_body({"src1": 1, "src2": 2}, sources=sources)
    assert body.splitlines()[0].startswith("1. Ada Okafor")
    assert "https://apollo-docs.example/latency" in body.splitlines()[1]


def test_notes_say_when_research_was_degraded_or_empty() -> None:
    assert degraded_notes(make_result()) == []
    notes = degraded_notes(make_result(degraded=["openalex: http 500"]))
    assert "degraded" in notes[0] and "openalex: http 500" in notes[0]
    empty = make_result().model_copy(update={"findings": []})
    assert "no usable passages" in degraded_notes(empty)[0]


# -- through the real graph ------------------------------------------------------------


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch):
    from andamentum.compose import assess as assess_module
    from andamentum.compose.checks.critic import CriticResult
    from andamentum.compose.schemas import CheckResult

    class Counting(Script):
        def __init__(self) -> None:
            super().__init__()
            self.extractions = 0

        async def _respond(self, **kw):  # type: ignore[no-untyped-def]
            from andamentum.compose.evidence import RawFacts

            if kw["output_type"] is RawFacts:
                self.extractions += 1
            return await super()._respond(**kw)

    s = Counting()
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", s)
    monkeypatch.setattr("andamentum.compose.checks.reader.run_agent_with_usage", s)
    monkeypatch.setattr("andamentum.compose.checks.grounding.run_agent_with_usage", s)

    class Scored:
        overall = "meets"

    async def judge(answer: str, **kw: object) -> Scored:
        return Scored()

    async def critic(bundle: object, **_: object) -> CriticResult:
        return CriticResult(
            check=CheckResult(name="critic", passed=True, details=[]),
            items=[],
            major=0,
            moderate=0,
            review_path=Path("review.json"),
        )

    monkeypatch.setattr("andamentum.compose.checks.reader.judge_score", judge)
    monkeypatch.setattr(assess_module, "run_critic", critic)
    return s


def _write(tmp_path: Path, **kw: object):
    notes = tmp_path / "notes.md"
    notes.write_text(SOURCE, encoding="utf-8")
    return write_document(
        "Email Jane Doe that the Apollo Labs demo moved to 19 March",
        out=tmp_path / "demo.compose",
        roles=Roles.all("test:model"),
        genre="email",
        sources=[str(notes)],
        formats=["txt"],
        **kw,  # type: ignore[arg-type]
    )


async def test_research_runs_once_in_the_genres_mode_over_the_bundles_store(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.v2.research", fake)
    result = await _write(tmp_path, research=WEB)

    bundle = tmp_path / "demo.compose"
    (call,) = fake.calls
    assert call["question"].startswith("Email Jane Doe")  # the user's own task text
    assert call["mode"] == "web"  # email is a web genre
    assert call["store"] == bundle / "research.db"
    assert call["model"] == "test:model"
    assert call["sources"] == WEB.sources

    sources = bundle_store.load_sources(bundle)
    assert [s.kind for s in sources] == ["file", "research", "research"]
    facts = bundle_store.load_facts(bundle)
    research = [f for f in facts if f.source_id in {"src2", "src3"}]
    assert [f.id for f in facts] == ["f1", "f2", "f3", "f4"]
    assert {f.quote for f in research} == {STUDY_QUOTE, BLOG_QUOTE}
    # the findings were not sent through the extraction model again
    assert script.extractions == 1
    assert bundle_store.load_research(bundle).run_id == "r_test"
    assert result.usage["drafter"] > 0


async def test_research_tokens_are_counted_in_the_usage(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("andamentum.deep_research.v2.research", FakeResearch())
    with_research = await _write(tmp_path, research=WEB)
    baseline = await write_document(
        "Email Jane Doe that the Apollo Labs demo moved to 19 March",
        out=tmp_path / "plain.compose",
        roles=Roles.all("test:model"),
        genre="email",
        sources=[str(tmp_path / "notes.md")],
        formats=["txt"],
    )
    assert with_research.usage["drafter"] - baseline.usage["drafter"] == 1200


async def test_a_resumed_run_reuses_the_saved_research(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.v2.research", fake)
    first = await _write(tmp_path, research=WEB)
    assert len(fake.calls) == 1
    again = await compose_bundle(
        first.bundle, roles=Roles.all("test:model"), formats=["txt"]
    )
    assert len(fake.calls) == 1  # no setup needed, nothing searched again
    assert again.stop_reason == "goal_met"


async def test_a_failed_research_run_cannot_be_resumed_without_its_settings(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "andamentum.deep_research.v2.research",
        FakeResearch(error=RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        await _write(tmp_path, research=WEB)
    with pytest.raises(ResearchSetupError, match="--research"):
        await compose_bundle(
            tmp_path / "demo.compose", roles=Roles.all("test:model"), formats=["txt"]
        )
    # with the settings and a working engine the same bundle completes
    monkeypatch.setattr("andamentum.deep_research.v2.research", FakeResearch())
    done = await compose_bundle(
        tmp_path / "demo.compose",
        roles=Roles.all("test:model"),
        formats=["txt"],
        research=WEB,
    )
    assert done.stop_reason == "goal_met"


async def test_web_research_without_a_searxng_address_is_refused(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.v2.research", fake)
    with pytest.raises(ResearchSetupError, match="SearxNG"):
        await _write(tmp_path, research=ResearchSetup(sources=SourceConfig()))
    assert fake.calls == []


async def test_the_mode_can_be_overridden_and_scholarly_needs_no_searxng(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.v2.research", fake)
    await _write(
        tmp_path, research=ResearchSetup(sources=SourceConfig(), mode="scholarly")
    )
    assert fake.calls[0]["mode"] == "scholarly"


async def test_a_degraded_run_is_reported_not_hidden(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeResearch(make_result(degraded=["openalex: http 500"]))
    monkeypatch.setattr("andamentum.deep_research.v2.research", fake)
    result = await _write(tmp_path, research=WEB)
    assert any("degraded" in q for q in result.open_questions)


async def test_without_research_nothing_is_searched(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeResearch()
    monkeypatch.setattr("andamentum.deep_research.v2.research", fake)
    await _write(tmp_path)
    assert fake.calls == []
    assert not (tmp_path / "demo.compose" / "research.db").exists()
    assert not bundle_store.has(tmp_path / "demo.compose", bundle_store.RESEARCH)


async def test_the_real_engine_over_an_invented_world_feeds_real_findings(
    tmp_path: Path, script, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine runs for real (only the network and the models are stand-ins);
    compose gets findings whose quotes are verbatim slices of stored passages,
    and the project's research.db is a real store."""
    from andamentum.core import fetch_gate, url_safety
    from andamentum.deep_research.v2 import ResearchStore
    from andamentum.deep_research.v2 import research as real_research
    from andamentum.deep_research.v2.tests.engine_world import (
        World,
        world_decisions,
        world_llm,
    )

    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()

    async def in_the_world(question: str, *, store: Path, **kwargs: object):
        opened = ResearchStore(store)
        try:
            kwargs.pop("model")
            kwargs.pop(
                "decision_model"
            )  # compose passes a name; the world's stub stands in
            return await real_research(
                question,
                model=world_llm().model,
                decision_model=world_decisions()[0],
                store=opened,
                source_client=World().client(opened),
                **kwargs,  # type: ignore[arg-type]
            )
        finally:
            opened.close()

    monkeypatch.setattr("andamentum.deep_research.v2.research", in_the_world)
    setup = ResearchSetup(
        sources=SourceConfig(scholarly=("openalex",)), mode="scholarly"
    )
    await _write(tmp_path, research=setup)

    bundle = tmp_path / "demo.compose"
    assert (bundle / "research.db").is_file()
    result = bundle_store.load_research(bundle)
    assert result.findings
    passages = {p.passage_id: p.text for p in result.passages}
    assert all(f.quote in passages[f.passage_id] for f in result.findings)
    facts = bundle_store.load_facts(bundle)
    sources = {s.id: s for s in bundle_store.load_sources(bundle)}
    research_facts_ = [f for f in facts if sources[f.source_id].kind == "research"]
    assert {f.quote for f in research_facts_} == {f.quote for f in result.findings}
    for fact in research_facts_:
        text = stored_source_path(bundle, sources[fact.source_id]).read_text(
            encoding="utf-8"
        )
        assert text[fact.start : fact.end] == fact.quote
    # the retracted work the world serves is never a source
    assert all("removes all request latency" not in s.title for s in sources.values())
