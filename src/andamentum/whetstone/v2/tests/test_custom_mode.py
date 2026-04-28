"""Tests for whetstone v2 custom-criteria mode.

Covers:

* Schema round-trip for CustomEvaluation.
* CustomReviewer node populates state from a mocked agent (using the
  build_dynamic_output_agent shim).
* CustomReviewer raises a clear error when custom_criteria is empty.
* End-to-end mode="custom" with mocked agent returns a populated
  ReviewResult.
* Markdown / HTML renderers surface the custom evaluations.
* CLI parses --criteria correctly (semicolon-separated and repeated).
"""

from __future__ import annotations

import unittest.mock as mock
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic_graph import GraphRunContext

from andamentum.whetstone.v2 import (
    CustomEvaluation,
    ReviewResult,
    render_html,
    render_markdown,
    review_document,
)
from andamentum.whetstone.v2.deps import ReviewDeps
from andamentum.whetstone.v2.dynamic_schemas import create_custom_evaluation_model
from andamentum.whetstone.v2.nodes.custom_reviewer import CustomReviewer
from andamentum.whetstone.v2.state import ReviewState


PAPER = """## Introduction

This paper studies bipedal walking with novel actor-critic methods.
Prior work (Smith 2020) is reviewed at length.

## Methods

We compare PPO and SAC variants. Hyperparameters are reported.
"""

CRITERIA = ["originality", "depth of literature", "clarity of methods"]


# ── Schema round-trip ─────────────────────────────────────────────────


class TestSchemaRoundtrip:
    def test_custom_evaluation_roundtrip(self) -> None:
        e = CustomEvaluation(
            criterion="originality", status="pass", notes="strong novelty."
        )
        restored = CustomEvaluation.model_validate(e.model_dump())
        assert restored == e

    def test_custom_evaluation_status_enum(self) -> None:
        with pytest.raises(Exception):
            CustomEvaluation.model_validate(
                {"criterion": "x", "status": "maybe", "notes": "n"}
            )


# ── CustomReviewer node ───────────────────────────────────────────────


@dataclass
class _FakeRunResult:
    output: Any


class _FakeAgent:
    def __init__(self, output: Any):
        self.output = output
        self.calls: list[str] = []

    async def run(self, prompt: str) -> _FakeRunResult:
        self.calls.append(prompt)
        return _FakeRunResult(output=self.output)


class _FakeContext:
    def __init__(self, state: ReviewState, deps: ReviewDeps):
        self.state = state
        self.deps = deps


def _as_ctx(fake: _FakeContext) -> GraphRunContext[ReviewState, ReviewDeps]:
    return cast(GraphRunContext[ReviewState, ReviewDeps], fake)


def _build_filled_dynamic_output(criteria: list[str]) -> Any:
    """Build a runtime-schema instance with canned per-criterion verdicts."""
    model = create_custom_evaluation_model(criteria)
    fields = {
        "originality_status": "pass",
        "originality_notes": "novel actor-critic combo.",
        "depth_of_literature_status": "fail",
        "depth_of_literature_notes": "only one citation found.",
        "clarity_of_methods_status": "unclear",
        "clarity_of_methods_notes": "hyperparameters reported but training schedule absent.",
        "overall_assessment": "Mixed quality across criteria.",
    }
    return model(**fields)


class TestCustomReviewerNode:
    async def test_runs_one_call_with_dynamic_schema(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="custom",
            custom_criteria=list(CRITERIA),
        )
        deps = ReviewDeps(model="fake:test")
        fake = _FakeAgent(output=_build_filled_dynamic_output(CRITERIA))

        called_with: dict[str, Any] = {}

        def fake_build(name: str, model: Any, output_type: Any) -> _FakeAgent:
            assert name == "custom_reviewer"
            called_with["output_type"] = output_type
            return fake

        with mock.patch(
            "andamentum.whetstone.v2.nodes.custom_reviewer.build_dynamic_output_agent",
            side_effect=fake_build,
        ):
            ctx = _FakeContext(state, deps)
            end_marker = await CustomReviewer().run(_as_ctx(ctx))

        assert state.llm_calls == 1
        assert len(state.custom_evaluations) == 3
        assert [e.criterion for e in state.custom_evaluations] == CRITERIA
        assert state.custom_evaluations[0].status == "pass"
        assert state.custom_evaluations[1].status == "fail"
        assert state.custom_evaluations[2].status == "unclear"
        # Each criterion is mirrored as a CheckableItem with source="custom".
        assert len(state.checkable_items) == 3
        assert all(it.source == "custom" for it in state.checkable_items)
        # The runtime model was passed as output_type.
        assert called_with["output_type"].__name__ == "CustomReviewerOutput"
        # Returns End[ReviewResult].
        out: ReviewResult = end_marker.data
        assert len(out.custom_evaluations) == 3
        assert "Custom-criteria review summary" in out.summary

    async def test_empty_criteria_raises(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="custom",
            custom_criteria=[],
        )
        deps = ReviewDeps(model="fake:test")

        with mock.patch(
            "andamentum.whetstone.v2.nodes.custom_reviewer.build_dynamic_output_agent",
        ) as patched:
            ctx = _FakeContext(state, deps)
            with pytest.raises(ValueError, match="custom_criteria is empty"):
                await CustomReviewer().run(_as_ctx(ctx))
            patched.assert_not_called()

    async def test_failure_surfaces_in_summary(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="custom",
            custom_criteria=list(CRITERIA),
        )
        deps = ReviewDeps(model="fake:test")

        class _FailingAgent:
            async def run(self, prompt: str) -> _FakeRunResult:
                raise RuntimeError("simulated failure")

        def fake_build(name: str, model: Any, output_type: Any):
            return _FailingAgent()

        with mock.patch(
            "andamentum.whetstone.v2.nodes.custom_reviewer.build_dynamic_output_agent",
            side_effect=fake_build,
        ):
            ctx = _FakeContext(state, deps)
            end_marker = await CustomReviewer().run(_as_ctx(ctx))

        # No verdicts but the result is still emitted.
        assert state.custom_evaluations == []
        assert "review call failed" in state.summary.lower()
        out: ReviewResult = end_marker.data
        assert out.custom_evaluations == []


# ── End-to-end ────────────────────────────────────────────────────────


async def test_e2e_custom_mode() -> None:
    """Full pipeline run with mocked custom_reviewer agent."""
    fake_output = _build_filled_dynamic_output(CRITERIA)

    def fake_build(name: str, model: Any, output_type: Any) -> _FakeAgent:
        assert name == "custom_reviewer"
        return _FakeAgent(output=fake_output)

    import andamentum.whetstone.v2.agents as agents_mod
    import andamentum.whetstone.v2.nodes.custom_reviewer as cr_mod

    with (
        mock.patch.multiple(agents_mod, build_dynamic_output_agent=fake_build),
        mock.patch.multiple(cr_mod, build_dynamic_output_agent=fake_build),
    ):
        result = await review_document(
            PAPER,
            model="fake:test",
            mode="custom",
            custom_criteria=tuple(CRITERIA),
        )

    assert len(result.custom_evaluations) == 3
    assert {e.status for e in result.custom_evaluations} == {
        "pass",
        "fail",
        "unclear",
    }
    # 1 LLM call, regardless of number of criteria.
    assert result.metrics.llm_calls == 1
    # Standard review-mode + panel fields are empty.
    assert result.findings == []
    assert result.expert_profiles == []
    assert result.guideline_evaluations == []
    # Summary surfaces the bucketed view.
    assert "Custom-criteria review summary" in result.summary


async def test_e2e_custom_mode_requires_criteria() -> None:
    with pytest.raises(ValueError, match="requires custom_criteria"):
        await review_document(
            PAPER,
            model="fake:test",
            mode="custom",
            custom_criteria=None,
        )


async def test_e2e_custom_mode_rejects_blank_criteria() -> None:
    with pytest.raises(ValueError, match="empty / whitespace"):
        await review_document(
            PAPER,
            model="fake:test",
            mode="custom",
            custom_criteria=("  ",),
        )


async def test_e2e_custom_mode_requires_model() -> None:
    with pytest.raises(ValueError, match="requires a model"):
        await review_document(
            PAPER,
            model=None,
            mode="custom",
            custom_criteria=("originality",),
        )


# ── Mode argument cross-checks ────────────────────────────────────────


async def test_review_mode_rejects_custom_criteria() -> None:
    with pytest.raises(ValueError, match="mode='review'"):
        await review_document(
            PAPER,
            model="fake:test",
            mode="review",
            custom_criteria=("originality",),
        )


async def test_review_mode_rejects_guidelines() -> None:
    with pytest.raises(ValueError, match="mode='review'"):
        await review_document(
            PAPER,
            model="fake:test",
            mode="review",
            guidelines="some text",
        )


# ── Renderer tests ────────────────────────────────────────────────────


def _custom_result() -> ReviewResult:
    return ReviewResult(
        summary="## Custom-criteria review summary\n\nMixed quality.",
        custom_evaluations=[
            CustomEvaluation(
                criterion="originality",
                status="pass",
                notes="Genuinely new combination.",
            ),
            CustomEvaluation(
                criterion="depth of literature",
                status="fail",
                notes="Only Smith 2020 is cited.",
            ),
            CustomEvaluation(
                criterion="clarity of methods",
                status="unclear",
                notes="Training schedule omitted.",
            ),
        ],
    )


def test_markdown_renderer_surfaces_custom_evaluations() -> None:
    md = render_markdown(_custom_result())
    assert "Custom-criteria evaluation (3)" in md
    assert "FAIL (1)" in md
    assert "UNCLEAR (1)" in md
    assert "PASS (1)" in md
    assert "depth of literature" in md
    assert "Only Smith 2020 is cited." in md


def test_html_renderer_surfaces_custom_evaluations() -> None:
    html = render_html(_custom_result())
    assert "Custom-criteria evaluation" in html
    assert "depth of literature" in html
    assert "Only Smith 2020 is cited." in html


def test_renderers_unaffected_when_no_custom_evaluations() -> None:
    result = ReviewResult(summary="Some summary")
    md = render_markdown(result)
    html = render_html(result)
    assert "Custom-criteria" not in md
    assert "Custom-criteria" not in html


# ── CLI argument parsing ──────────────────────────────────────────────


def test_cli_parses_criteria_semicolon_separated() -> None:
    from andamentum.whetstone.v2.cli import (
        _build_parser,
        _parse_criteria,
        _validate_args,
    )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "custom",
            "--model",
            "openai:gpt-5.4-nano",
            "--criteria",
            "originality; depth of literature; clarity of methods",
        ]
    )
    _validate_args(args)
    parsed = _parse_criteria(args.criteria)
    assert parsed == [
        "originality",
        "depth of literature",
        "clarity of methods",
    ]


def test_cli_parses_criteria_repeated_flags() -> None:
    from andamentum.whetstone.v2.cli import (
        _build_parser,
        _parse_criteria,
        _validate_args,
    )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "custom",
            "--model",
            "openai:gpt-5.4-nano",
            "--criteria",
            "originality",
            "--criteria",
            "depth of literature",
        ]
    )
    _validate_args(args)
    parsed = _parse_criteria(args.criteria)
    assert parsed == ["originality", "depth of literature"]


def test_cli_validates_custom_with_no_llm() -> None:
    from andamentum.whetstone.v2.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "custom",
            "--no-llm",
            "--criteria",
            "originality",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_cli_validates_criteria_required_when_mode_custom() -> None:
    from andamentum.whetstone.v2.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "custom",
            "--model",
            "openai:gpt-5.4-nano",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_cli_rejects_criteria_with_other_modes() -> None:
    from andamentum.whetstone.v2.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "review",
            "--model",
            "openai:gpt-5.4-nano",
            "--criteria",
            "originality",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)
