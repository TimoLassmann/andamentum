"""Tests for whetstone v2 panel mode.

Covers:

* Each new agent's output schema serialises correctly (pydantic round-trip).
* ExtractKeywords node populates state.disciplines from a mocked agent.
* GenerateExpertPanel produces N ExpertProfiles in parallel.
* ExpertReview produces N ExpertReviews in parallel, each tied to a profile.
* PanelSynthesise produces a single PanelSynthesis from N reviews.
* End-to-end: ``await review_document(paper, mode="panel", model="fake:test")``
  with mocked agents returns a ReviewResult populated with
  expert_profiles, expert_reviews, panel_synthesis.
* Markdown renderer surfaces expert names, scores, and recommendations.
* HTML renderer same.

Mocking pattern: same as ``test_pipeline_e2e.py`` — patch
``build_pydantic_ai_agent`` in every node module that imports it.
"""

from __future__ import annotations

import unittest.mock as mock
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic_graph import GraphRunContext

from andamentum.whetstone import (
    ExpertProfile,
    ExpertReview,
    PanelSynthesis,
    render_html,
    render_markdown,
    review_document,
)
from andamentum.whetstone.agents import KeywordExtractionOutput
from andamentum.whetstone.deps import ReviewDeps
from andamentum.whetstone.nodes.expert_review import (
    ExpertReview as ExpertReviewNode,
)
from andamentum.whetstone.nodes.extract_keywords import ExtractKeywords
from andamentum.whetstone.nodes.generate_expert_panel import GenerateExpertPanel
from andamentum.whetstone.nodes.panel_synthesise import PanelSynthesise
from andamentum.whetstone.schemas import ReviewResult, SectionCard
from andamentum.whetstone.state import ReviewState


PAPER = """## 1 Introduction

This paper studies Reinforcement Learning (RL) applied to bipedal walking.
We had N = 50 participants in our user study, and cite prior work [1, 42].
As shown in Figure 1, the results are striking.

## 2 Methods

We compare two variants of RL on the same benchmark.
Across N=48 trials, the new method outperforms baselines significantly.
Figure 1: Comparison of accuracy across methods.

## References

[1] First Author. (2020). Title one.
[2] Second Author. (2021).
"""


# ── Test fixtures: canned output objects ───────────────────────────────


def _profile(name: str, discipline: str) -> ExpertProfile:
    return ExpertProfile(
        name=name,
        position=f"Professor, {discipline}, Imaginary University",
        education=f"PhD in {discipline}, MIT, 2005",
        contributions=f"Major contributions to {discipline}: A, B, C.",
        research=f"Currently studies {discipline}.",
        discipline=discipline,
    )


def _review(name: str, discipline: str, score: int = 7) -> ExpertReview:
    return ExpertReview(
        expert_name=name,
        discipline=discipline,
        overall_score=score,
        overall_assessment=f"Overall assessment from {name}.",
        scientific_rigor_score=score,
        scientific_rigor_justification="Rigour is adequate.",
        methodology_score=score,
        methodology_justification="Methodology is reasonable.",
        novelty_score=score,
        novelty_justification="Novelty is moderate.",
        clarity_score=score,
        clarity_justification="Clarity is good.",
        strengths=["Clear structure", "Reasonable claims"],
        weaknesses=["Some unsupported claims"],
        recommendation="Minor Revisions",
        recommendation_justification="Acceptable with revisions.",
    )


def _synthesis(n: int = 3) -> PanelSynthesis:
    return PanelSynthesis(
        average_overall_score=7.0,
        score_range="6-8",
        number_of_experts=n,
        consensus_strengths=["Solid approach"],
        consensus_weaknesses=["Missing baseline"],
        divergent_opinions=["Disagreement on novelty"],
        scientific_rigor_summary="Rigour is adequate across the panel.",
        methodology_summary="Methodology was praised by 2 of 3 experts.",
        novelty_summary="Novelty is contested.",
        clarity_summary="Clarity is uniformly judged good.",
        overall_recommendation="Minor Revisions",
        recommendation_justification="Panel converges on minor revisions.",
        confidence_level="medium",
        key_decision_factors=["Solid methodology", "Limited novelty"],
        review_summary=(
            "The panel finds this a competent submission with limited "
            "but real contribution. Minor revisions are needed primarily "
            "around the missing baseline comparison and a clearer novelty "
            "framing. The four experts agreed on the basic shape of the "
            "needed revisions, with one dissenting view on the novelty "
            "axis. Overall the panel recommends minor revisions."
        ),
    )


# ── Schema round-trip tests ───────────────────────────────────────────


class TestSchemaRoundtrip:
    def test_expert_profile_roundtrip(self) -> None:
        p = _profile("Dr Imagined", "Robotics")
        dumped = p.model_dump()
        restored = ExpertProfile.model_validate(dumped)
        assert restored == p

    def test_expert_review_roundtrip(self) -> None:
        r = _review("Dr Imagined", "Robotics", score=8)
        dumped = r.model_dump()
        restored = ExpertReview.model_validate(dumped)
        assert restored == r
        # Score bounds enforced.
        with pytest.raises(Exception):
            ExpertReview.model_validate({**dumped, "overall_score": 11})

    def test_panel_synthesis_roundtrip(self) -> None:
        s = _synthesis(n=4)
        dumped = s.model_dump()
        restored = PanelSynthesis.model_validate(dumped)
        assert restored == s

    def test_review_result_carries_panel_fields(self) -> None:
        r = ReviewResult(
            expert_profiles=[_profile("A", "X"), _profile("B", "Y")],
            expert_reviews=[_review("A", "X"), _review("B", "Y")],
            panel_synthesis=_synthesis(n=2),
        )
        assert len(r.expert_profiles) == 2
        assert len(r.expert_reviews) == 2
        assert r.panel_synthesis is not None
        # Round-trip
        restored = ReviewResult.model_validate(r.model_dump())
        assert restored.expert_profiles == r.expert_profiles
        assert restored.panel_synthesis == r.panel_synthesis


# ── Per-node tests with a fake agent factory ───────────────────────────


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
    """Minimal stand-in for pydantic-graph's GraphRunContext.

    ``GraphRunContext`` is a thin holder for ``state`` + ``deps`` plus
    pydantic-graph internals we don't need in tests. Cast the instance
    via ``_as_ctx`` when passing it to a node's ``run``.
    """

    def __init__(self, state: ReviewState, deps: ReviewDeps):
        self.state = state
        self.deps = deps


def _as_ctx(fake: _FakeContext) -> GraphRunContext[ReviewState, ReviewDeps]:
    """Cast a _FakeContext to GraphRunContext for the type checker."""
    return cast(GraphRunContext[ReviewState, ReviewDeps], fake)


class TestExtractKeywordsNode:
    async def test_extracts_disciplines_from_agent(self) -> None:
        state = ReviewState(source="dummy", markdown=PAPER, mode="panel", n_experts=4)
        deps = ReviewDeps(model="fake:test")

        fake = _FakeAgent(
            output=KeywordExtractionOutput(
                disciplines=[
                    "Computational Robotics",
                    "Reinforcement Learning",
                    "Cognitive Science",
                ]
            )
        )

        with mock.patch(
            "andamentum.whetstone.nodes.extract_keywords.build_pydantic_ai_agent",
            return_value=fake,
        ):
            ctx = _FakeContext(state, deps)
            next_node = await ExtractKeywords().run(_as_ctx(ctx))

        assert state.disciplines == [
            "Computational Robotics",
            "Reinforcement Learning",
            "Cognitive Science",
        ]
        assert state.llm_calls == 1
        # Should advance to GenerateExpertPanel.
        assert isinstance(next_node, GenerateExpertPanel)

    async def test_skips_llm_when_disciplines_provided(self) -> None:
        state = ReviewState(
            source="dummy",
            markdown=PAPER,
            mode="panel",
            panel_disciplines=["Statistics", "Robotics"],
        )
        deps = ReviewDeps(model="fake:test")

        # Patch the builder to assert it is NOT called.
        with mock.patch(
            "andamentum.whetstone.nodes.extract_keywords.build_pydantic_ai_agent",
        ) as patched:
            ctx = _FakeContext(state, deps)
            await ExtractKeywords().run(_as_ctx(ctx))
            patched.assert_not_called()

        assert state.disciplines == ["Statistics", "Robotics"]
        assert state.llm_calls == 0

    async def test_dedupes_and_strips(self) -> None:
        state = ReviewState(source="dummy", markdown=PAPER, mode="panel")
        deps = ReviewDeps(model="fake:test")

        fake = _FakeAgent(
            output=KeywordExtractionOutput(
                disciplines=["Physics", " Physics", "  ", "Chemistry"]
            )
        )

        with mock.patch(
            "andamentum.whetstone.nodes.extract_keywords.build_pydantic_ai_agent",
            return_value=fake,
        ):
            ctx = _FakeContext(state, deps)
            await ExtractKeywords().run(_as_ctx(ctx))

        assert state.disciplines == ["Physics", "Chemistry"]


class TestGenerateExpertPanelNode:
    async def test_generates_n_profiles_in_parallel(self) -> None:
        disciplines = ["A", "B", "C", "D"]
        state = ReviewState(
            source="dummy",
            mode="panel",
            n_experts=4,
            disciplines=list(disciplines),
        )
        deps = ReviewDeps(model="fake:test")

        # Each call returns a profile whose discipline echoes the input.
        # We can't peek at the prompt to extract it, so we return a
        # generic profile and let the node's defensive code restore
        # the discipline.
        call_count = {"n": 0}

        def fake_build(name: str, model: Any) -> _FakeAgent:
            assert name == "expert_generator"
            call_count["n"] += 1
            return _FakeAgent(
                output=ExpertProfile(
                    name=f"Dr {call_count['n']}",
                    position="Prof",
                    education="PhD",
                    contributions="x",
                    research="y",
                    discipline="",  # blank — node should restore from prompt
                )
            )

        with mock.patch(
            "andamentum.whetstone.nodes.generate_expert_panel.build_pydantic_ai_agent",
            side_effect=fake_build,
        ):
            ctx = _FakeContext(state, deps)
            next_node = await GenerateExpertPanel().run(_as_ctx(ctx))

        assert len(state.expert_profiles) == 4
        assert state.llm_calls == 4
        # Each profile got its discipline restored from the discipline list.
        # The order of completion is non-deterministic but the SET should match.
        restored = {p.discipline for p in state.expert_profiles}
        assert restored == set(disciplines)
        assert isinstance(next_node, ExpertReviewNode)

    async def test_caps_at_n_experts(self) -> None:
        # 6 disciplines, n_experts=3 → only 3 generations.
        state = ReviewState(
            source="dummy",
            mode="panel",
            n_experts=3,
            disciplines=["A", "B", "C", "D", "E", "F"],
        )
        deps = ReviewDeps(model="fake:test")

        builds = {"n": 0}

        def fake_build(name: str, model: Any) -> _FakeAgent:
            builds["n"] += 1
            return _FakeAgent(
                output=ExpertProfile(
                    name=f"Dr {builds['n']}",
                    position="Prof",
                    education="PhD",
                    contributions="x",
                    research="y",
                    discipline="X",
                )
            )

        with mock.patch(
            "andamentum.whetstone.nodes.generate_expert_panel.build_pydantic_ai_agent",
            side_effect=fake_build,
        ):
            ctx = _FakeContext(state, deps)
            await GenerateExpertPanel().run(_as_ctx(ctx))

        assert len(state.expert_profiles) == 3
        assert builds["n"] == 3

    async def test_partial_failure_continues(self) -> None:
        state = ReviewState(
            source="dummy",
            mode="panel",
            n_experts=3,
            disciplines=["A", "B", "C"],
        )
        deps = ReviewDeps(model="fake:test")

        results = [
            ExpertProfile(
                name="A",
                position="P",
                education="E",
                contributions="c",
                research="r",
                discipline="A",
            ),
            None,  # second call raises
            ExpertProfile(
                name="C",
                position="P",
                education="E",
                contributions="c",
                research="r",
                discipline="C",
            ),
        ]
        idx = {"i": 0}

        class _MaybeFailAgent:
            async def run(self, prompt: str) -> _FakeRunResult:
                i = idx["i"]
                idx["i"] += 1
                out = results[i]
                if out is None:
                    raise RuntimeError("simulated profile failure")
                return _FakeRunResult(output=out)

        with mock.patch(
            "andamentum.whetstone.nodes.generate_expert_panel.build_pydantic_ai_agent",
            return_value=_MaybeFailAgent(),
        ):
            ctx = _FakeContext(state, deps)
            await GenerateExpertPanel().run(_as_ctx(ctx))

        # Only 2 succeeded, but the run continues.
        assert len(state.expert_profiles) == 2


class TestExpertReviewNode:
    async def test_runs_one_review_per_profile(self) -> None:
        profiles = [
            _profile("Dr A", "Statistics"),
            _profile("Dr B", "Robotics"),
            _profile("Dr C", "Cognitive Science"),
        ]
        state = ReviewState(
            source="dummy",
            mode="panel",
            markdown=PAPER,
            document_map=[
                SectionCard(section_id="sec_001", title="Intro", one_line_gist="intro"),
            ],
            expert_profiles=list(profiles),
        )
        deps = ReviewDeps(model="fake:test")

        idx = {"i": 0}

        def fake_build(name: str, model: Any) -> _FakeAgent:
            assert name == "expert_reviewer"
            i = idx["i"]
            idx["i"] += 1
            return _FakeAgent(
                output=_review(profiles[i].name, profiles[i].discipline, score=7 + i)
            )

        with mock.patch(
            "andamentum.whetstone.nodes.expert_review.build_pydantic_ai_agent",
            side_effect=fake_build,
        ):
            ctx = _FakeContext(state, deps)
            next_node = await ExpertReviewNode().run(_as_ctx(ctx))

        assert len(state.expert_reviews) == 3
        assert state.llm_calls == 3
        # Each review's discipline matches its profile.
        review_disciplines = sorted(r.discipline for r in state.expert_reviews)
        assert review_disciplines == sorted(p.discipline for p in profiles)
        assert isinstance(next_node, PanelSynthesise)

    async def test_no_profiles_skips(self) -> None:
        state = ReviewState(
            source="dummy",
            mode="panel",
            expert_profiles=[],
        )
        deps = ReviewDeps(model="fake:test")

        with mock.patch(
            "andamentum.whetstone.nodes.expert_review.build_pydantic_ai_agent",
        ) as patched:
            ctx = _FakeContext(state, deps)
            next_node = await ExpertReviewNode().run(_as_ctx(ctx))
            patched.assert_not_called()

        assert state.expert_reviews == []
        assert isinstance(next_node, PanelSynthesise)


class TestPanelSynthesiseNode:
    async def test_synthesises_from_reviews(self) -> None:
        reviews = [_review("A", "X", 6), _review("B", "Y", 7), _review("C", "Z", 8)]
        state = ReviewState(source="dummy", mode="panel", expert_reviews=list(reviews))
        deps = ReviewDeps(model="fake:test")

        fake = _FakeAgent(output=_synthesis(n=3))

        with mock.patch(
            "andamentum.whetstone.nodes.panel_synthesise.build_pydantic_ai_agent",
            return_value=fake,
        ):
            ctx = _FakeContext(state, deps)
            end_marker = await PanelSynthesise().run(_as_ctx(ctx))

        assert state.panel_synthesis is not None
        assert state.panel_synthesis.number_of_experts == 3
        # ``summary`` stays empty when panel_synthesis is set; renderers
        # format the synthesis directly to avoid duplicate output.
        assert state.summary == ""
        assert state.llm_calls == 1
        # Returns End[ReviewResult]
        out: ReviewResult = end_marker.data
        assert out.panel_synthesis is not None
        assert len(out.expert_reviews) == 3

    async def test_no_reviews_skips_synthesis(self) -> None:
        state = ReviewState(source="dummy", mode="panel", expert_reviews=[])
        deps = ReviewDeps(model="fake:test")

        with mock.patch(
            "andamentum.whetstone.nodes.panel_synthesise.build_pydantic_ai_agent",
        ) as patched:
            ctx = _FakeContext(state, deps)
            end_marker = await PanelSynthesise().run(_as_ctx(ctx))
            patched.assert_not_called()

        assert state.panel_synthesis is None
        out: ReviewResult = end_marker.data
        assert out.panel_synthesis is None
        # We still emit a summary noting the skip.
        assert state.summary

    async def test_synthesis_failure_surfaces_message(self) -> None:
        reviews = [_review("A", "X")]
        state = ReviewState(source="dummy", mode="panel", expert_reviews=list(reviews))
        deps = ReviewDeps(model="fake:test")

        class _FailingAgent:
            async def run(self, prompt: str) -> _FakeRunResult:
                raise RuntimeError("simulated synthesis failure")

        with mock.patch(
            "andamentum.whetstone.nodes.panel_synthesise.build_pydantic_ai_agent",
            return_value=_FailingAgent(),
        ):
            ctx = _FakeContext(state, deps)
            await PanelSynthesise().run(_as_ctx(ctx))

        assert state.panel_synthesis is None  # crashed
        # Per-expert reviews still in the result.
        assert state.expert_reviews == reviews
        assert "synthesis call failed" in state.summary.lower()


# ── End-to-end with mocked agents ─────────────────────────────────────


@pytest.fixture
def panel_canned() -> dict[str, Any]:
    return {
        "extract_keywords": KeywordExtractionOutput(
            disciplines=["Robotics", "Reinforcement Learning", "Statistics"]
        ),
        "expert_generator_outputs": [
            _profile("Dr Alice", "Robotics"),
            _profile("Dr Bob", "Reinforcement Learning"),
            _profile("Dr Carol", "Statistics"),
        ],
        "expert_reviewer_outputs": [
            _review("Dr Alice", "Robotics", score=7),
            _review("Dr Bob", "Reinforcement Learning", score=8),
            _review("Dr Carol", "Statistics", score=6),
        ],
        "panel_synthesise": _synthesis(n=3),
    }


def _make_panel_fake_build(canned: dict[str, Any]):
    """Build a fake_build that returns one canned output per agent name.

    For the multi-call agents (expert_generator, expert_reviewer) we
    return outputs in order from a list — one per call.
    """
    gen_idx = {"i": 0}
    rev_idx = {"i": 0}

    def fake_build(name: str, model: Any) -> _FakeAgent:
        if name == "expert_generator":
            outs = canned["expert_generator_outputs"]
            i = gen_idx["i"]
            gen_idx["i"] = (i + 1) % len(outs)
            return _FakeAgent(output=outs[i])
        if name == "expert_reviewer":
            outs = canned["expert_reviewer_outputs"]
            i = rev_idx["i"]
            rev_idx["i"] = (i + 1) % len(outs)
            return _FakeAgent(output=outs[i])
        if name in canned:
            return _FakeAgent(output=canned[name])
        raise AssertionError(f"agent {name!r} called but no canned output set")

    return fake_build


async def test_e2e_panel_mode(panel_canned: dict[str, Any]) -> None:
    fake_build = _make_panel_fake_build(panel_canned)

    import andamentum.whetstone.agents as agents_mod
    import andamentum.whetstone.nodes.expert_review as er_mod
    import andamentum.whetstone.nodes.extract_keywords as ek_mod
    import andamentum.whetstone.nodes.generate_expert_panel as gep_mod
    import andamentum.whetstone.nodes.panel_synthesise as ps_mod

    with (
        mock.patch.multiple(agents_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(ek_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(gep_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(er_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(ps_mod, build_pydantic_ai_agent=fake_build),
    ):
        result = await review_document(
            PAPER,
            model="fake:test",
            mode="panel",
            n_experts=3,
        )

    # Panel-mode fields populated.
    assert len(result.expert_profiles) == 3
    assert len(result.expert_reviews) == 3
    assert result.panel_synthesis is not None
    assert result.panel_synthesis.number_of_experts == 3

    # Standard review-mode fields are not (or are empty).
    assert result.findings == []
    assert result.edits == []

    # ``summary`` is empty in panel mode; recommendation lives in panel_synthesis.
    assert result.summary == ""
    assert result.panel_synthesis.overall_recommendation == "Minor Revisions"

    # LLM call count = 1 (extract) + 3 (generate) + 3 (review) + 1 (synth) = 8.
    assert result.metrics.llm_calls == 8


async def test_e2e_panel_mode_with_explicit_disciplines(
    panel_canned: dict[str, Any],
) -> None:
    """Explicit disciplines skip the keyword-extraction LLM call.

    Total LLM calls drop from 8 to 7.
    """
    fake_build = _make_panel_fake_build(panel_canned)

    import andamentum.whetstone.agents as agents_mod
    import andamentum.whetstone.nodes.expert_review as er_mod
    import andamentum.whetstone.nodes.extract_keywords as ek_mod
    import andamentum.whetstone.nodes.generate_expert_panel as gep_mod
    import andamentum.whetstone.nodes.panel_synthesise as ps_mod

    with (
        mock.patch.multiple(agents_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(ek_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(gep_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(er_mod, build_pydantic_ai_agent=fake_build),
        mock.patch.multiple(ps_mod, build_pydantic_ai_agent=fake_build),
    ):
        result = await review_document(
            PAPER,
            model="fake:test",
            mode="panel",
            n_experts=3,
            panel_disciplines=("Robotics", "Reinforcement Learning", "Statistics"),
        )

    assert len(result.expert_profiles) == 3
    # 0 (extract — skipped) + 3 + 3 + 1 = 7.
    assert result.metrics.llm_calls == 7


# ── Renderer tests ─────────────────────────────────────────────────────


def _panel_result() -> ReviewResult:
    return ReviewResult(
        summary="",
        expert_profiles=[
            _profile("Dr Alice", "Robotics"),
            _profile("Dr Bob", "Statistics"),
        ],
        expert_reviews=[
            _review("Dr Alice", "Robotics", score=7),
            _review("Dr Bob", "Statistics", score=8),
        ],
        panel_synthesis=_synthesis(n=2),
    )


def test_markdown_renderer_surfaces_panel_output() -> None:
    md = render_markdown(_panel_result())

    # Headlines.
    assert "Panel synthesis" in md
    assert "Expert reviews (2)" in md
    assert "Expert biosketches (2)" in md

    # Expert names + disciplines + scores + recommendations.
    assert "Dr Alice" in md
    assert "Robotics" in md
    assert "Dr Bob" in md
    assert "Statistics" in md
    assert "7/10" in md  # Alice's overall score
    assert "8/10" in md  # Bob's overall score
    assert "Minor Revisions" in md  # both reviews + the synthesis recommendation


def test_html_renderer_surfaces_panel_output() -> None:
    html = render_html(_panel_result())

    assert "Panel synthesis" in html
    assert "Expert reviews" in html
    assert "Expert biosketches" in html
    # Names and disciplines appear in the rendered HTML.
    assert "Dr Alice" in html
    assert "Robotics" in html
    assert "Dr Bob" in html
    assert "Statistics" in html
    # Recommendation appears (in the synthesis card or per-expert cards).
    assert "Minor Revisions" in html


def test_renderers_unaffected_when_no_panel_fields() -> None:
    """A standard review-mode ReviewResult should render exactly as before
    (no panel sections appearing)."""
    result = ReviewResult(
        summary="Some summary",
        document_map=[
            SectionCard(section_id="sec_001", title="Intro", one_line_gist="x"),
        ],
    )
    md = render_markdown(result)
    html = render_html(result)

    assert "Panel synthesis" not in md
    assert "Expert reviews" not in md
    assert "Expert biosketches" not in md

    assert "Panel synthesis" not in html
    assert "Expert reviews" not in html
    assert "Expert biosketches" not in html


# ── CLI argument parsing ──────────────────────────────────────────────


def test_cli_validates_panel_with_no_llm() -> None:
    """--mode panel requires --model."""
    from andamentum.whetstone.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        ["paper.md", "--out", "out.md", "--mode", "panel", "--no-llm"]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_cli_accepts_panel_mode() -> None:
    """--mode panel parses cleanly with --model."""
    from andamentum.whetstone.cli import _build_parser, _validate_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            "out.md",
            "--mode",
            "panel",
            "--model",
            "openai:gpt-5.4-nano",
            "--n-experts",
            "5",
            "--panel-disciplines",
            "Statistics, Robotics",
        ]
    )
    _validate_args(args)  # should not raise
    assert args.mode == "panel"
    assert args.n_experts == 5
    assert args.panel_disciplines == "Statistics, Robotics"
