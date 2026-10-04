"""Orchestrator behaviour with every worker stubbed: stop rules, resume, routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from andamentum.compose import graph as graph_module
from andamentum.compose.assess import Assessment
from andamentum.compose.checks.critic import CriticResult
from andamentum.compose.checks.reader import ReaderResult
from andamentum.compose.genres import GENRES
from andamentum.compose.graph import ComposeDeps, run_graph
from andamentum.compose.log import ActionLog
from andamentum.compose.render import ProvenanceRequired
from andamentum.compose.route import Routed
from andamentum.compose.schemas import (
    CheckResult,
    Outline,
    Roles,
    SectionPlan,
    WorkItem,
)
from andamentum.compose.tests._stub import make_brief
from andamentum.compose.bundle import Bundle

ROLES = Roles.all("test:model")
ITEM = WorkItem(level="section", section_id="s01", reason="too vague", source="critic")
OUTLINE_ITEM = WorkItem(
    level="outline", section_id=None, reason="missing step", source="reader"
)


def assessment(
    *,
    met: bool,
    items: list[WorkItem] | None = None,
    outline: list[WorkItem] | None = None,
) -> Assessment:
    reader = ReaderResult(
        check=CheckResult(name="reader", passed=True, details=[]),
        misses=[],
        correct=2,
        total=2,
    )
    critic = CriticResult(
        check=CheckResult(name="critic", passed=met, details=[]),
        items=[],
        major=0 if met else 1,
        moderate=0,
        review_path=Path("review.json"),
    )
    return Assessment(
        round=0,
        checks=[
            CheckResult(name="critic", passed=met, details=["x"] if not met else [])
        ],
        reader=reader,
        critic=critic,
        routed=Routed(
            by_section={"s01": items} if items else {}, outline=outline or []
        ),
        goal_met=met,
        grounding_unsupported=0,
    )


class Workers:
    """Records and scripts every worker the graph calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, verdicts: list[Assessment]):
        self.verdicts = list(verdicts)
        self.calls: list[str] = []
        self.draft_body = "Apollo Labs starts the pilot on 1 March."
        self.revised: list[str] = []
        self.revise_bodies: list[str] | None = None
        monkeypatch.setattr(graph_module, "compile_brief", self.compile_brief)
        monkeypatch.setattr(graph_module, "extract_facts", self.extract_facts)
        monkeypatch.setattr(graph_module, "make_outline", self.make_outline)
        monkeypatch.setattr(graph_module, "amend_outline", self.amend_outline)
        monkeypatch.setattr(graph_module, "draft_section", self.draft_section)
        monkeypatch.setattr(graph_module, "revise_section", self.revise_section)
        monkeypatch.setattr(graph_module, "assess", self.assess)

    async def compile_brief(self, task: str, **kw: Any):
        self.calls.append("brief")
        return make_brief(
            genre="email", word_range=(1, 300), email_subject="Pilot", must_include=[]
        )

    async def extract_facts(self, *a: Any, **kw: Any):
        self.calls.append("facts")
        return []

    async def make_outline(self, *a: Any, **kw: Any) -> Outline:
        self.calls.append("outline")
        return Outline(sections=[self._plan("Body")])

    async def amend_outline(self, outline: Outline, items: Any, **kw: Any) -> Outline:
        self.calls.append("amend")
        return Outline(sections=[self._plan("Body"), self._plan("Next step")])

    @staticmethod
    def _plan(title: str) -> SectionPlan:
        return SectionPlan(
            title=title, purpose="p", key_points=["k"], fact_ids=[], target_words=50
        )

    async def draft_section(self, plan: SectionPlan, **kw: Any) -> str:
        self.calls.append(f"draft:{plan.title}")
        return self.draft_body

    async def revise_section(self, body: str, items: Any, **kw: Any) -> str:
        self.calls.append("revise")
        if self.revise_bodies is not None:
            new = self.revise_bodies.pop(0)
        else:
            new = f"{body} Revised {len(self.revised) + 1}."
        self.revised.append(new)
        return new

    async def assess(self, bundle: Any, **kw: Any) -> Assessment:
        self.calls.append("assess")
        return self.verdicts.pop(0)


def make_deps(tmp_path: Path, **overrides: Any) -> ComposeDeps:
    bundle = Bundle.create(tmp_path / "doc.compose", title="Apollo pilot")
    base: dict[str, Any] = dict(
        roles=ROLES,
        bundle_dir=bundle.path,
        task="Write to Jane Doe about the Apollo Labs pilot",
        genre_spec="email",
        source_paths=(),
        log=ActionLog(bundle.path),
        progress=lambda line: None,
        max_rounds=4,
        formats=("txt",),
    )
    base.update(overrides)
    return ComposeDeps(**base)


async def test_goal_met_needs_two_passes_over_the_same_text(tmp_path, monkeypatch):
    workers = Workers(
        monkeypatch, verdicts=[assessment(met=True), assessment(met=True)]
    )
    result = await run_graph(make_deps(tmp_path))
    assert result.stop_reason == "goal_met" and result.goal_met
    assert len(result.rounds) == 2
    assert "revise" not in workers.calls
    assert [p.name for p in result.outputs] == ["email.txt"]
    assert (tmp_path / "doc.compose" / "report.md").is_file()


async def test_one_passing_round_followed_by_a_failure_resets_the_streak(
    tmp_path, monkeypatch
):
    workers = Workers(
        monkeypatch,
        verdicts=[
            assessment(met=True),
            assessment(met=False, items=[ITEM]),
            assessment(met=True),
            assessment(met=True),
        ],
    )
    result = await run_graph(make_deps(tmp_path, max_rounds=6))
    assert result.stop_reason == "goal_met"
    assert len(result.rounds) == 4
    assert workers.calls.count("revise") == 1
    assert result.rounds[1].changed_sections == ["s01"]


async def test_no_progress_when_revision_changes_nothing(tmp_path, monkeypatch):
    workers = Workers(monkeypatch, verdicts=[assessment(met=False, items=[ITEM])])
    workers.revise_bodies = [workers.draft_body]
    result = await run_graph(make_deps(tmp_path))
    assert result.stop_reason == "no_progress" and not result.goal_met
    assert len(result.rounds) == 1
    assert result.outputs  # the document is still written


async def test_no_progress_when_text_returns_to_an_earlier_round(tmp_path, monkeypatch):
    workers = Workers(
        monkeypatch,
        verdicts=[
            assessment(met=False, items=[ITEM]),
            assessment(met=False, items=[ITEM]),
        ],
    )
    workers.revise_bodies = ["A different text entirely.", workers.draft_body]
    result = await run_graph(make_deps(tmp_path, max_rounds=6))
    assert result.stop_reason == "no_progress"
    assert len(result.rounds) == 2


async def test_cap_reached_stops_the_loop(tmp_path, monkeypatch):
    workers = Workers(
        monkeypatch, verdicts=[assessment(met=False, items=[ITEM]) for _ in range(3)]
    )
    result = await run_graph(make_deps(tmp_path, max_rounds=3))
    assert result.stop_reason == "cap_reached" and not result.goal_met
    assert len(result.rounds) == 3
    assert workers.calls.count("revise") == 2  # no revision after the last check


async def test_resume_skips_finished_nodes(tmp_path, monkeypatch):
    first = Workers(monkeypatch, verdicts=[assessment(met=True), assessment(met=True)])
    deps = make_deps(tmp_path)
    await run_graph(deps)
    assert first.calls[:4] == ["brief", "outline", "draft:Body", "assess"]

    second = Workers(monkeypatch, verdicts=[assessment(met=True), assessment(met=True)])
    result = await run_graph(deps)
    assert second.calls == ["assess", "assess"]
    assert result.stop_reason == "goal_met"
    assert len(result.rounds) == 4  # earlier rounds are kept
    assert result.brief.title == "Apollo pilot"


async def test_resume_drafts_only_sections_still_planned(tmp_path, monkeypatch):
    workers = Workers(
        monkeypatch, verdicts=[assessment(met=True), assessment(met=True)]
    )
    deps = make_deps(tmp_path)
    await run_graph(deps)
    bundle = Bundle.open(deps.bundle_dir)
    section = bundle.section("s01")
    bundle.write_section(
        section.id, body="", expected_revision=section.revision, status="planned"
    )

    again = Workers(monkeypatch, verdicts=[assessment(met=True), assessment(met=True)])
    await run_graph(deps)
    assert again.calls.count("draft:Body") == 1
    assert "brief" not in again.calls and "outline" not in again.calls
    assert workers.calls  # first run did the work


async def test_nothing_drafted_is_unresolvable(tmp_path, monkeypatch):
    workers = Workers(monkeypatch, verdicts=[])
    workers.draft_body = ""
    result = await run_graph(make_deps(tmp_path))
    assert result.stop_reason == "unresolvable"
    assert result.outputs == []
    assert "assess" not in workers.calls


async def test_outline_items_amend_once_then_become_open_questions(
    tmp_path, monkeypatch
):
    workers = Workers(
        monkeypatch,
        verdicts=[
            assessment(met=False, outline=[OUTLINE_ITEM]),
            assessment(met=False, outline=[OUTLINE_ITEM]),
            assessment(met=False, outline=[OUTLINE_ITEM]),
        ],
    )
    result = await run_graph(make_deps(tmp_path, max_rounds=3))
    assert workers.calls.count("amend") == 1
    assert "draft:Next step" in workers.calls
    assert any("missing step" in q for q in result.open_questions)
    assert result.stop_reason in ("no_progress", "cap_reached")


async def test_provenance_off_is_refused_for_a_provenance_genre(tmp_path, monkeypatch):
    Workers(monkeypatch, verdicts=[])
    assert GENRES["grant"].provenance_required
    deps = make_deps(tmp_path, genre_spec="grant", provenance=False)
    with pytest.raises(ProvenanceRequired):
        await run_graph(deps)
