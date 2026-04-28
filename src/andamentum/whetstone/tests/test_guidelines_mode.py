"""Tests for whetstone v2 guidelines mode.

Covers:

* Schema round-trips (CheckableItem, GuidelineEvaluation).
* ExtractCheckableItems node populates state from a mocked agent.
* ExtractCheckableItems raises a clear error when guidelines_text is empty.
* EvaluateGuidelineItems runs N parallel calls and populates state.
* EvaluateGuidelineItems degrades per-item failures to "unclear" rather
  than aborting the whole run.
* End-to-end mode="guidelines" with mocked agents returns a populated
  ReviewResult with checkable_items + guideline_evaluations.
* Markdown / HTML renderers surface the items, status, and notes.
* CLI accepts --guidelines @file (text fixture).
"""

from __future__ import annotations

import unittest.mock as mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_graph import GraphRunContext

from andamentum.whetstone import (
    CheckableItem,
    GuidelineEvaluation,
    ReviewResult,
    render_html,
    render_markdown,
    review_document,
)
from andamentum.whetstone.agents import ExtractedItemsList
from andamentum.whetstone.deps import ReviewDeps
from andamentum.whetstone.nodes.evaluate_guideline_items import (
    EvaluateGuidelineItems,
)
from andamentum.whetstone.nodes.extract_checkable_items import (
    ExtractCheckableItems,
)
from andamentum.whetstone.schemas import SectionCard
from andamentum.whetstone.state import ReviewState


PAPER = """## 1 Introduction

This paper studies bipedal walking. Word count of abstract: ~200.
We had N = 50 participants and cite prior work [1, 42].

## 2 Methods

We compare two variants of RL on the same benchmark.

## References

[1] First Author. (2020). Title one.
"""

GUIDELINES = """\
Author guidelines for the Imaginary Journal of Robotics

Manuscripts must include a structured abstract of no more than 250 words.
Figures should be supplied in vector format. A data availability
statement is required. Author contributions section is required.
References should follow Vancouver style. Manuscripts should not exceed
8000 words including references.
"""


# ── Schema round-trip tests ───────────────────────────────────────────


class TestSchemaRoundtrip:
    def test_checkable_item_roundtrip(self) -> None:
        item = CheckableItem(name="Abstract ≤ 250 words", source="guidelines")
        restored = CheckableItem.model_validate(item.model_dump())
        assert restored == item

    def test_checkable_item_source_enum(self) -> None:
        with pytest.raises(Exception):
            CheckableItem.model_validate({"name": "x", "source": "bogus"})

    def test_guideline_evaluation_roundtrip(self) -> None:
        e = GuidelineEvaluation(
            item_name="Abstract ≤ 250 words",
            status="pass",
            notes="The abstract is 220 words.",
            category="abstract",
        )
        restored = GuidelineEvaluation.model_validate(e.model_dump())
        assert restored == e

    def test_guideline_evaluation_status_enum(self) -> None:
        with pytest.raises(Exception):
            GuidelineEvaluation.model_validate(
                {"item_name": "x", "status": "maybe", "notes": "n"}
            )


# ── Per-node tests ─────────────────────────────────────────────────────


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


class TestExtractCheckableItemsNode:
    async def test_extracts_items_from_agent(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="guidelines",
            guidelines_text=GUIDELINES,
        )
        deps = ReviewDeps(model="fake:test")
        fake = _FakeAgent(
            output=ExtractedItemsList(
                items=[
                    "Abstract ≤ 250 words",
                    "Figures supplied in vector format",
                    "Data availability statement present",
                ]
            )
        )

        with mock.patch(
            "andamentum.whetstone.nodes.extract_checkable_items.build_pydantic_ai_agent",
            return_value=fake,
        ):
            ctx = _FakeContext(state, deps)
            next_node = await ExtractCheckableItems().run(_as_ctx(ctx))

        assert len(state.checkable_items) == 3
        assert all(it.source == "guidelines" for it in state.checkable_items)
        assert state.checkable_items[0].name == "Abstract ≤ 250 words"
        assert state.llm_calls == 1
        assert isinstance(next_node, EvaluateGuidelineItems)

    async def test_dedupes_and_strips(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="guidelines",
            guidelines_text=GUIDELINES,
        )
        deps = ReviewDeps(model="fake:test")
        fake = _FakeAgent(
            output=ExtractedItemsList(items=["Abstract", " Abstract", "  ", "Figures"])
        )

        with mock.patch(
            "andamentum.whetstone.nodes.extract_checkable_items.build_pydantic_ai_agent",
            return_value=fake,
        ):
            ctx = _FakeContext(state, deps)
            await ExtractCheckableItems().run(_as_ctx(ctx))

        names = [it.name for it in state.checkable_items]
        assert names == ["Abstract", "Figures"]

    async def test_empty_guidelines_raises(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="guidelines",
            guidelines_text="   ",
        )
        deps = ReviewDeps(model="fake:test")

        with mock.patch(
            "andamentum.whetstone.nodes.extract_checkable_items.build_pydantic_ai_agent",
        ) as patched:
            ctx = _FakeContext(state, deps)
            with pytest.raises(ValueError, match="guidelines_text is empty"):
                await ExtractCheckableItems().run(_as_ctx(ctx))
            patched.assert_not_called()


class TestEvaluateGuidelineItemsNode:
    async def test_runs_one_evaluation_per_item(self) -> None:
        items = [
            CheckableItem(name="Abstract ≤ 250 words", source="guidelines"),
            CheckableItem(name="Vector figures", source="guidelines"),
            CheckableItem(name="Data availability statement", source="guidelines"),
        ]
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="guidelines",
            document_map=[
                SectionCard(
                    section_id="sec_001",
                    title="Intro",
                    one_line_gist="intro",
                ),
            ],
            checkable_items=list(items),
        )
        deps = ReviewDeps(model="fake:test")

        idx = {"i": 0}
        from typing import Literal as _L

        statuses: list[_L["pass", "fail", "unclear"]] = ["pass", "fail", "unclear"]

        def fake_build(name: str, model: Any) -> _FakeAgent:
            assert name == "guideline_item_evaluator"
            i = idx["i"]
            idx["i"] += 1
            return _FakeAgent(
                output=GuidelineEvaluation(
                    item_name=items[i].name,
                    status=statuses[i],
                    notes=f"Notes for item {i}",
                    category="",
                )
            )

        with mock.patch(
            "andamentum.whetstone.nodes.evaluate_guideline_items.build_pydantic_ai_agent",
            side_effect=fake_build,
        ):
            ctx = _FakeContext(state, deps)
            end_marker = await EvaluateGuidelineItems().run(_as_ctx(ctx))

        assert len(state.guideline_evaluations) == 3
        assert state.llm_calls == 3
        # Returns End[ReviewResult]
        out: ReviewResult = end_marker.data
        assert len(out.guideline_evaluations) == 3
        assert {e.status for e in out.guideline_evaluations} == {
            "pass",
            "fail",
            "unclear",
        }

    async def test_partial_failure_becomes_unclear(self) -> None:
        items = [
            CheckableItem(name="A", source="guidelines"),
            CheckableItem(name="B", source="guidelines"),
        ]
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="guidelines",
            checkable_items=list(items),
        )
        deps = ReviewDeps(model="fake:test")

        idx = {"i": 0}

        class _MaybeFailAgent:
            async def run(self, prompt: str) -> _FakeRunResult:
                i = idx["i"]
                idx["i"] += 1
                if i == 0:
                    raise RuntimeError("simulated failure")
                return _FakeRunResult(
                    output=GuidelineEvaluation(item_name="B", status="pass", notes="ok")
                )

        with mock.patch(
            "andamentum.whetstone.nodes.evaluate_guideline_items.build_pydantic_ai_agent",
            return_value=_MaybeFailAgent(),
        ):
            ctx = _FakeContext(state, deps)
            await EvaluateGuidelineItems().run(_as_ctx(ctx))

        assert len(state.guideline_evaluations) == 2
        # Find the failing one — its status should be "unclear".
        failed = [e for e in state.guideline_evaluations if e.item_name == "A"]
        assert len(failed) == 1
        assert failed[0].status == "unclear"
        assert "Evaluation failed" in failed[0].notes

    async def test_no_items_skips(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="guidelines",
            checkable_items=[],
        )
        deps = ReviewDeps(model="fake:test")

        with mock.patch(
            "andamentum.whetstone.nodes.evaluate_guideline_items.build_pydantic_ai_agent",
        ) as patched:
            ctx = _FakeContext(state, deps)
            end_marker = await EvaluateGuidelineItems().run(_as_ctx(ctx))
            patched.assert_not_called()

        out: ReviewResult = end_marker.data
        assert out.guideline_evaluations == []
        assert state.summary  # something noting the skip


# ── End-to-end with mocked agents ─────────────────────────────────────


@pytest.fixture
def guidelines_canned() -> dict[str, Any]:
    return {
        "extract_checkable_items": ExtractedItemsList(
            items=[
                "Abstract ≤ 250 words",
                "Figures in vector format",
                "Data availability statement present",
            ]
        ),
        "guideline_evaluations": [
            GuidelineEvaluation(
                item_name="Abstract ≤ 250 words",
                status="pass",
                notes="220-word abstract present.",
            ),
            GuidelineEvaluation(
                item_name="Figures in vector format",
                status="fail",
                notes="Figures appear as raster (PNG).",
            ),
            GuidelineEvaluation(
                item_name="Data availability statement present",
                status="unclear",
                notes="No statement found in standard locations.",
            ),
        ],
    }


def _make_guidelines_fake_build(canned: dict[str, Any]):
    eval_idx = {"i": 0}

    def fake_build(name: str, model: Any) -> _FakeAgent:
        if name == "extract_checkable_items":
            return _FakeAgent(output=canned["extract_checkable_items"])
        if name == "guideline_item_evaluator":
            outs = canned["guideline_evaluations"]
            i = eval_idx["i"]
            eval_idx["i"] = (i + 1) % len(outs)
            return _FakeAgent(output=outs[i])
        raise AssertionError(f"agent {name!r} called but no canned output set")

    return fake_build


async def test_e2e_guidelines_mode(guidelines_canned: dict[str, Any]) -> None:
    fake_build = _make_guidelines_fake_build(guidelines_canned)

    import andamentum.whetstone.agents as agents_mod
    import andamentum.whetstone.nodes.evaluate_guideline_items as eg_mod
    import andamentum.whetstone.nodes.extract_checkable_items as ex_mod

    with (
        mock.patch.multiple(agents_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(ex_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(eg_mod, build_pydantic_ai_agent=fake_build),
    ):
        result = await review_document(
            PAPER,
            model="fake:test",
            mode="guidelines",
            guidelines=GUIDELINES,
        )

    assert len(result.checkable_items) == 3
    assert len(result.guideline_evaluations) == 3
    statuses = {e.status for e in result.guideline_evaluations}
    assert statuses == {"pass", "fail", "unclear"}
    # 1 (extract) + 3 (evaluate) = 4 LLM calls.
    assert result.metrics.llm_calls == 4
    # Standard review-mode fields are empty.
    assert result.findings == []
    assert result.expert_profiles == []
    # Summary contains the headline.
    assert "Guideline checklist summary" in result.summary


async def test_e2e_guidelines_mode_requires_guidelines() -> None:
    with pytest.raises(ValueError, match="non-empty guidelines"):
        await review_document(
            PAPER,
            model="fake:test",
            mode="guidelines",
            guidelines="",
        )


async def test_e2e_guidelines_mode_requires_model() -> None:
    with pytest.raises(ValueError, match="requires a model"):
        await review_document(
            PAPER,
            model=None,
            mode="guidelines",
            guidelines=GUIDELINES,
        )


# ── Renderer tests ─────────────────────────────────────────────────────


def _guidelines_result() -> ReviewResult:
    return ReviewResult(
        summary="## Guideline checklist summary\n\nEvaluated **3** rules.",
        checkable_items=[
            CheckableItem(name="Abstract ≤ 250 words", source="guidelines"),
            CheckableItem(name="Vector figures", source="guidelines"),
            CheckableItem(name="Data availability", source="guidelines"),
        ],
        guideline_evaluations=[
            GuidelineEvaluation(
                item_name="Abstract ≤ 250 words",
                status="pass",
                notes="220 words.",
            ),
            GuidelineEvaluation(
                item_name="Vector figures",
                status="fail",
                notes="PNG only.",
            ),
            GuidelineEvaluation(
                item_name="Data availability",
                status="unclear",
                notes="Not found.",
            ),
        ],
    )


def test_markdown_renderer_surfaces_guideline_evaluations() -> None:
    md = render_markdown(_guidelines_result())
    assert "Journal-guideline checks (3)" in md
    assert "FAIL (1)" in md
    assert "UNCLEAR (1)" in md
    assert "PASS (1)" in md
    assert "Vector figures" in md
    assert "PNG only." in md
    assert "Abstract ≤ 250 words" in md


def test_html_renderer_surfaces_guideline_evaluations() -> None:
    html = render_html(_guidelines_result())
    assert "Journal-guideline checks" in html
    assert "Vector figures" in html
    assert "PNG only." in html


def test_renderers_unaffected_when_no_guideline_evaluations() -> None:
    result = ReviewResult(summary="Some summary")
    md = render_markdown(result)
    html = render_html(result)
    assert "Journal-guideline" not in md
    assert "Journal-guideline" not in html


# ── CLI argument parsing ──────────────────────────────────────────────


def test_cli_parses_guidelines_at_file(tmp_path: Path) -> None:
    """--guidelines @file reads file contents."""
    from andamentum.whetstone.cli import (
        _build_parser,
        _resolve_guidelines,
        _validate_args,
    )

    guidelines_file = tmp_path / "journal.txt"
    guidelines_file.write_text(GUIDELINES, encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "guidelines",
            "--model",
            "openai:gpt-5.4-nano",
            "--guidelines",
            f"@{guidelines_file}",
        ]
    )
    _validate_args(args)
    text = _resolve_guidelines(args.guidelines)
    assert "vector format" in text
    assert "Vancouver style" in text


def test_cli_validates_guidelines_with_no_llm() -> None:
    from andamentum.whetstone.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "guidelines",
            "--no-llm",
            "--guidelines",
            "some text",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_cli_validates_guidelines_required_when_mode_guidelines() -> None:
    from andamentum.whetstone.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "guidelines",
            "--model",
            "openai:gpt-5.4-nano",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)
