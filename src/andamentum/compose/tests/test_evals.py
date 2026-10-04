"""The synthetic evaluation tasks are well formed and the report renders."""

from __future__ import annotations

from andamentum.compose.evals.runner import (
    ArmMetrics,
    EvalReport,
    TaskEval,
    load_tasks,
)
from andamentum.compose.genres import GENRES
from andamentum.compose.schemas import Roles


def test_two_synthetic_tasks_per_genre_with_existing_sources():
    tasks = load_tasks()
    assert len(tasks) == 2 * len(GENRES) == 14
    for genre in GENRES:
        assert sum(1 for t in tasks if t.genre == genre) == 2
    for task in tasks:
        assert task.task
        assert task.sources, f"{task.genre}/{task.name} has no sources"
        for source in task.sources:
            assert source.is_file(), source


def test_report_and_grant_tasks_cover_figures_and_call_text():
    tasks = load_tasks()
    for t in tasks:
        if t.genre in ("report", "article"):
            assert any(s.suffix == ".csv" for s in t.sources)
        if t.genre == "grant":
            assert any(s.name == "call.md" for s in t.sources)


def test_load_tasks_rejects_unknown_genre():
    import pytest

    with pytest.raises(ValueError, match="unknown genre"):
        load_tasks(genres=["poem"])


def _arm(failed: list[str]) -> ArmMetrics:
    return ArmMetrics(
        checks_passed=["total_words"],
        checks_failed=failed,
        reader_correct=2,
        reader_total=3,
        grounding_unsupported=0,
        words=210,
        rounds=2,
        stop_reason="goal_met",
        tokens={},
        calls={"drafter": 4},
    )


def test_report_markdown_names_both_arms_and_the_winner():
    report = EvalReport(
        roles=Roles.all("m"),
        tasks=[
            TaskEval(
                genre="email",
                name="late-shipment",
                compose=_arm([]),
                baseline=_arm(["reader"]),
                winner="compose",
                judge_confidence=0.8,
                needs_review=False,
            )
        ],
    )
    text = report.markdown()
    assert "| email | late-shipment | compose |" in text
    assert "| email | late-shipment | baseline | reader |" in text
    assert "compose 1, baseline 0, tie 0" in text
    assert "| not reported |" in text
    arm = _arm([]).model_copy(update={"tokens": {"drafter": 120, "judge": 30}})
    assert (
        "150 (drafter 120, judge 30)"
        in EvalReport(
            roles=Roles.all("m"),
            tasks=[
                TaskEval(
                    genre="email",
                    name="x",
                    compose=arm,
                    baseline=arm,
                    winner="tie",
                    judge_confidence=0.5,
                    needs_review=False,
                )
            ],
        ).markdown()
    )
