"""Tests for the K5 fix: degenerate-question guard at graph entry.

The freeze sheet's K5: ``andamentum-epistemic stage <name> --question Q``
silently produced 4 sub-investigations of nothing meaningful, burned
API budget, and left a degenerate DB on disk (``run1.db`` is the
artifact). The guard fires inside ``run_epistemic_graph`` only on the
fresh-objective branch — resume / targeted runs ignore the question
and must not trip the validator.

These tests pin:
  1. Trivial inputs ("Q", "", "  ", single token) raise
     DegenerateQuestionError.
  2. Real research questions pass.
  3. The validator is skipped when an objective_id is supplied or an
     existing objective is found in the DB (resume modes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.epistemic.graph import (
    DegenerateQuestionError,
    _validate_research_question,
    run_epistemic_graph,
)
from andamentum.epistemic.graph.nodes import PrepareObjective


pytestmark = pytest.mark.asyncio


# ── Validator unit tests ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_input",
    [
        "",
        " ",
        "\t\n",
        "Q",
        "?",
        "Why",  # single token, even if real word
        "abc",  # too short and single token
    ],
)
async def test_validator_rejects_degenerate(bad_input: str) -> None:
    with pytest.raises(DegenerateQuestionError):
        _validate_research_question(bad_input)


@pytest.mark.parametrize(
    "good_input",
    [
        "Is exercise good for cardiovascular health?",
        "Does metformin extend lifespan?",
        "Is X safe?",  # 11 chars, 3 words — borderline but real
        "Why does this matter?",
    ],
)
async def test_validator_accepts_real_questions(good_input: str) -> None:
    # No exception
    _validate_research_question(good_input)


# ── Integration: graph entry rejects degenerate input ────────────────


async def test_run_graph_rejects_q_single_letter(tmp_path: Path, monkeypatch) -> None:
    """The K5 case verbatim: ``--question Q`` on a fresh DB used to
    produce a degenerate decomposition. Now it raises before any DB
    write or LLM call."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    with pytest.raises(DegenerateQuestionError) as excinfo:
        await run_epistemic_graph(
            question="Q",
            database_name="k5_q",
            db_dir=str(tmp_path),
            model="fake:test-model",
            embedding_model="fake-embeddings",
            decompose=True,
            stop_after=PrepareObjective,
        )
    assert "too short" in str(excinfo.value).lower() or "single token" in str(excinfo.value).lower()


async def test_run_graph_accepts_real_question_then_resumes_without_check(
    tmp_path: Path, monkeypatch
) -> None:
    """The validator only fires when creating a fresh objective. Once
    the DB has an objective, subsequent calls (which conventionally
    pass a placeholder like ``"(resumed)"``) must skip the validator —
    otherwise stage-runner resume would be unreachable."""
    from andamentum.epistemic.tests.conftest import FakeAgentRunner
    import andamentum.epistemic.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "DefaultAgentRunner",
        lambda model: FakeAgentRunner(),  # noqa: ARG005
    )

    # First call creates the objective from a real question.
    r1 = await run_epistemic_graph(
        question="Is exercise good for cardiovascular health?",
        database_name="k5_resume",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        decompose=True,
        stop_after=PrepareObjective,
    )
    assert r1.objective_id is not None

    # Second call uses the resume placeholder. Validator must NOT fire
    # because the DB already has an objective.
    r2 = await run_epistemic_graph(
        question="(resumed)",  # placeholder — would fail the validator
        database_name="k5_resume",
        db_dir=str(tmp_path),
        model="fake:test-model",
        embedding_model="fake-embeddings",
        decompose=True,
        stop_after=PrepareObjective,
    )
    assert r2.objective_id == r1.objective_id, (
        "Resume must reuse the existing objective, not create a new "
        "one — and the placeholder question must not trip the "
        "degenerate-question guard."
    )
