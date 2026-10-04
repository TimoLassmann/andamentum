"""The real workers wired through the real graph, with only the model boundary
scripted: ingest, brief, evidence, plan, draft, deterministic checks, grounding,
reader, routing, revision and rendering all run for real. The critic is the one
stub, because it needs a whetstone model run; its own tests cover it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from andamentum.compose import assess as assess_module
from andamentum.compose import write_document
from andamentum.compose.checks.critic import CriticResult
from andamentum.compose.checks.grounding import GroundingVerdict
from andamentum.compose.checks.reader import ReaderAnswers
from andamentum.compose.evidence import RawFact, RawFacts
from andamentum.compose.brief import BriefDraft
from andamentum.compose.schemas import (
    CheckResult,
    Outline,
    ReaderQuestion,
    Roles,
    SectionPlan,
    WorkItem,
)
from andamentum.compose.write import SectionBody

from ._stub import USAGE

SOURCE = (
    "# Demo notes\n\n"
    "The Apollo Labs demo moves to 19 March at 10:00 in Meeting Room 4.\n"
    "Harbor Foods must confirm the attendees by 10 March.\n"
)
PADDING = (
    "The session runs for forty five minutes and our engineer will walk through "
    "the product, the pricing and the rollout plan, then leave time for questions "
    "from everyone who attends on the day, so please bring your questions along."
)
FIRST = (
    "The Apollo Labs demo moves to 19 March at 10:00. [[f1]] "
    f"Please confirm the attendees soon. {PADDING}"
)
SECOND = (
    "The Apollo Labs demo moves to 19 March at 10:00. [[f1]] "
    f"Please confirm the attendees by 10 March. [[f2]] {PADDING}"
)


class Script:
    """Scripted replacement for run_agent_with_usage, dispatching on output type."""

    def __init__(self) -> None:
        self.draft_calls = 0
        self.revise_calls = 0
        self.grounding_calls = 0

    async def __call__(self, model: Any, **kw: Any) -> tuple[BaseModel, Any]:
        return await self._respond(**kw), USAGE

    async def _respond(self, **kw: Any) -> BaseModel:
        kind = kw["output_type"]
        if kind is BriefDraft:
            return BriefDraft(
                genre="email",
                title="Demo date change",
                audience="Jane Doe at Harbor Foods",
                purpose="Move the demo and get attendees confirmed",
                key_messages=["The demo moved to 19 March"],
                must_include=["19 March"],
                constraints=[],
                reader_questions=[
                    ReaderQuestion(question="New date?", expected_answer="19 March"),
                    ReaderQuestion(question="Confirm by?", expected_answer="10 March"),
                ],
                email_subject="Demo moved to 19 March",
            )
        if kind is RawFacts:
            return RawFacts(
                facts=[
                    RawFact(
                        statement="The demo is on 19 March",
                        quote="moves to 19 March at 10:00",
                    ),
                    RawFact(
                        statement="Attendees by 10 March",
                        quote="confirm the attendees by 10 March",
                    ),
                    RawFact(statement="invented", quote="not in the source at all"),
                ]
            )
        if kind is Outline:
            return Outline(
                sections=[
                    SectionPlan(
                        title="Body",
                        purpose="Say what changed and what is needed",
                        key_points=["19 March", "confirm attendees by 10 March"],
                        fact_ids=["f1", "f2"],
                        target_words=60,
                    )
                ]
            )
        if kind is SectionBody:
            return SectionBody(body=self._body(kw))
        if kind is GroundingVerdict:
            self.grounding_calls += 1
            return GroundingVerdict(unsupported=[])
        if kind is ReaderAnswers:
            return ReaderAnswers(answers=["19 March", "10 March"])
        raise AssertionError(f"unscripted call for {kind}")

    def _body(self, kw: dict[str, Any]) -> str:
        message = kw["user_message"]
        if "Please confirm the attendees soon." in message:  # a revision call
            self.revise_calls += 1
            return SECOND
        self.draft_calls += 1
        return FIRST


class Scored:
    overall = "meets"


async def test_real_workers_through_the_real_graph(tmp_path, monkeypatch):
    script = Script()
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", script)
    monkeypatch.setattr("andamentum.compose.checks.reader.run_agent_with_usage", script)
    monkeypatch.setattr(
        "andamentum.compose.checks.grounding.run_agent_with_usage", script
    )

    async def judge(answer: str, **kw: Any) -> Scored:
        return Scored()

    monkeypatch.setattr("andamentum.compose.checks.reader.judge_score", judge)

    critic_rounds: list[int] = []

    async def critic(
        bundle: Any, *, genre: Any, model: str, round_no: int, log: Any, **_: Any
    ):
        critic_rounds.append(round_no)
        # Round 1 raises one major finding about the vague sentence; later rounds are clean.
        major = 1 if round_no == 1 else 0
        section = bundle.sections()[0]
        items = (
            [
                WorkItem(
                    level="section",
                    section_id=section.id,
                    reason="[major] The deadline is vague",
                    source="critic",
                )
            ]
            if major
            else []
        )
        return CriticResult(
            check=CheckResult(name="critic", passed=not major, details=[]),
            items=items,
            major=major,
            moderate=0,
            review_path=Path("review.json"),
        )

    monkeypatch.setattr(assess_module, "run_critic", critic)

    source = tmp_path / "notes.md"
    source.write_text(SOURCE, encoding="utf-8")
    result = await write_document(
        "Email Jane Doe that the Apollo Labs demo moved to 19 March",
        out=tmp_path / "demo.compose",
        roles=Roles.all("test:model"),
        genre="email",
        sources=[str(source)],
        max_rounds=5,
        formats=["txt", "md"],
    )

    assert result.stop_reason == "goal_met", result.rounds[-1].checks
    assert script.draft_calls == 1 and script.revise_calls == 1
    assert critic_rounds == [1, 2, 3]
    assert [r.goal_met for r in result.rounds] == [False, True, True]
    assert result.rounds[0].changed_sections == ["s01"]

    bundle = tmp_path / "demo.compose"
    for name in ("brief.json", "outline.json", "rounds.json", "log.jsonl", "report.md"):
        assert (bundle / name).is_file(), name
    assert (bundle / "evidence" / "facts.json").is_file()
    email = (bundle / "out" / "email.txt").read_text(encoding="utf-8")
    assert email.startswith("Subject: Demo moved to 19 March\n\n")
    assert "[[" not in email
    assert "10 March" in email
    assert (bundle / "out" / "document.md").is_file()
    report = (bundle / "report.md").read_text(encoding="utf-8")
    assert "goal_met" in report and "drafter: test:model" in report
    assert (
        script.grounding_calls == 2
    )  # drafted text, then the revised text; the re-check is cached

    facts = (bundle / "evidence" / "facts.json").read_text(encoding="utf-8")
    assert "invented" not in facts  # a fact whose quote is not in the source is dropped


async def test_resuming_a_finished_bundle_makes_no_early_calls(tmp_path, monkeypatch):
    from andamentum.compose import compose_bundle

    script = Script()
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", script)
    monkeypatch.setattr("andamentum.compose.checks.reader.run_agent_with_usage", script)
    monkeypatch.setattr(
        "andamentum.compose.checks.grounding.run_agent_with_usage", script
    )

    async def judge(answer: str, **kw: Any) -> Scored:
        return Scored()

    async def critic(
        bundle: Any, *, genre: Any, model: str, round_no: int, log: Any, **_: Any
    ):
        return CriticResult(
            check=CheckResult(name="critic", passed=True, details=[]),
            items=[],
            major=0,
            moderate=0,
            review_path=Path("review.json"),
        )

    monkeypatch.setattr("andamentum.compose.checks.reader.judge_score", judge)
    monkeypatch.setattr(assess_module, "run_critic", critic)
    source = tmp_path / "notes.md"
    source.write_text(SOURCE, encoding="utf-8")
    roles = Roles.all("test:model")
    first = await write_document(
        "Email Jane Doe that the demo moved to 19 March",
        out=tmp_path / "d.compose",
        roles=roles,
        genre="email",
        sources=[str(source)],
        formats=["txt"],
    )
    drafts_before = script.draft_calls
    again = await compose_bundle(first.bundle, roles=roles, formats=["txt"])
    assert again.stop_reason == "goal_met"
    assert script.draft_calls == drafts_before
    assert len(again.rounds) == len(first.rounds) + 2


class ShortDraftScript(Script):
    """First draft is too short for the email word range; only a length item fixes it."""

    SHORT = (
        "The Apollo Labs demo moves to 19 March at 10:00. [[f1]] "
        "Please confirm the attendees by 10 March. [[f2]]"
    )

    def _body(self, kw: dict[str, Any]) -> str:
        if "lengthen this section" in kw["user_message"]:
            self.revise_calls += 1
            return SECOND
        self.draft_calls += 1
        return self.SHORT


async def test_a_too_short_draft_is_lengthened_by_a_section_revision(
    tmp_path, monkeypatch
):
    script = ShortDraftScript()
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", script)
    monkeypatch.setattr("andamentum.compose.checks.reader.run_agent_with_usage", script)
    monkeypatch.setattr(
        "andamentum.compose.checks.grounding.run_agent_with_usage", script
    )

    async def judge(answer: str, **kw: Any) -> Scored:
        return Scored()

    async def critic(
        bundle: Any, *, genre: Any, model: str, round_no: int, log: Any, **_: Any
    ):
        return CriticResult(
            check=CheckResult(name="critic", passed=True, details=[]),
            items=[],
            major=0,
            moderate=0,
            review_path=Path("review.json"),
        )

    monkeypatch.setattr("andamentum.compose.checks.reader.judge_score", judge)
    monkeypatch.setattr(assess_module, "run_critic", critic)
    source = tmp_path / "notes.md"
    source.write_text(SOURCE, encoding="utf-8")
    result = await write_document(
        "Email Jane Doe that the Apollo Labs demo moved to 19 March",
        out=tmp_path / "short.compose",
        roles=Roles.all("test:model"),
        genre="email",
        sources=[str(source)],
        max_rounds=5,
        formats=["txt"],
    )
    assert script.revise_calls == 1
    assert result.rounds[0].changed_sections == ["s01"]
    assert result.stop_reason == "goal_met", result.rounds[-1].checks
