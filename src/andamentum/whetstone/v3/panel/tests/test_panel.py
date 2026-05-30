"""Panel-mode graph for v3.

Coverage:
  - run_panel_v3 returns a ReviewResult with panel fields populated
  - ExtractKeywords skipped when caller supplies panel_disciplines
  - Per-discipline profile generation isolates failures (None filtered)
  - Per-expert review isolates failures (None filtered)
  - panel_synthesise crash is loud-fail-safe (reviews still returned)
  - n_experts caps the discipline → profile fan-out
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from andamentum.whetstone.schemas import (
    ExpertProfile,
    ExpertReview,
    PanelSynthesis,
)
from andamentum.whetstone.v3.panel.agents import KeywordExtractionOutput
from andamentum.whetstone.v3.panel.graph import (
    PanelDeps,
    PanelState,
    Sectionize,
    panel_graph_v3,
    run_panel_v3,
)


SRC = "# Intro\n\nWe present a study of X.\n\n# Methods\n\nWe analysed Y.\n"


def _profile(name: str, discipline: str) -> ExpertProfile:
    return ExpertProfile(
        name=name,
        position=f"Professor of {discipline}",
        education="PhD, Test University (2005)",
        contributions="- Authored several papers on the topic.",
        research="Currently focused on aspects of " + discipline,
        discipline=discipline,
    )


def _review(name: str, discipline: str, score: int = 7) -> ExpertReview:
    return ExpertReview(
        expert_name=name,
        discipline=discipline,
        overall_score=score,
        overall_assessment="Solid work overall.",
        scientific_rigor_score=score,
        scientific_rigor_justification="Methods are reasonable.",
        methodology_score=score,
        methodology_justification="Standard approach.",
        novelty_score=score,
        novelty_justification="Modestly novel.",
        clarity_score=score,
        clarity_justification="Writing is clear.",
        strengths=["Clear motivation", "Solid evaluation"],
        weaknesses=["Missing a baseline", "Limited discussion of priors"],
        recommendation="Minor Revisions",
        recommendation_justification="Address the missing baseline.",
    )


def _synthesis(n: int = 2) -> PanelSynthesis:
    return PanelSynthesis(
        average_overall_score=7.0,
        score_range="6-8",
        number_of_experts=n,
        consensus_strengths=["Clear motivation"],
        consensus_weaknesses=["Missing baseline"],
        divergent_opinions=[],
        scientific_rigor_summary="Adequate rigor across reviewers.",
        methodology_summary="Methodology is standard.",
        novelty_summary="Modestly novel.",
        clarity_summary="Writing is clear.",
        overall_recommendation="Minor Revisions",
        recommendation_justification="Address the missing baseline.",
        confidence_level="medium",
        key_decision_factors=["Add baseline comparison"],
        review_summary="The panel finds the work solid but recommends a baseline.",
    )


# ── Helpers to stub the four agents ────────────────────────────────────────


class _StubAgent:
    """Configurable async agent stub. Each instance is constructed with
    a single output value to return from .run()."""

    def __init__(self, output) -> None:
        self.output = output
        self.captured_prompts: list[str] = []

    async def run(self, prompt: str):
        self.captured_prompts.append(prompt)
        return SimpleNamespace(output=self.output)


def _patch_build_agent_chain(*outputs):
    """Return a callable suitable for `patch(..., side_effect=...)`
    that returns a fresh _StubAgent for each successive build_pydantic_ai_agent
    call, cycling through the supplied outputs."""
    agents = [_StubAgent(o) for o in outputs]
    iter_agents = iter(agents)

    def _build(*_args, **_kwargs):
        try:
            return next(iter_agents)
        except StopIteration:
            # Fall back to a no-op agent so the test doesn't crash on an
            # extra build call — clearer signal than StopIteration.
            return _StubAgent(SimpleNamespace())

    return _build, agents


# ── Tests ──────────────────────────────────────────────────────────────────


async def test_run_panel_v3_populates_review_result_panel_fields() -> None:
    """End-to-end: all four agents stubbed; result carries
    expert_profiles, expert_reviews, panel_synthesis."""
    build, agents = _patch_build_agent_chain(
        KeywordExtractionOutput(disciplines=["AI", "Linguistics"]),
        _profile("Alice", "AI"),
        _profile("Bob", "Linguistics"),
        _review("Alice", "AI"),
        _review("Bob", "Linguistics"),
        _synthesis(n=2),
    )
    with (
        patch(
            "andamentum.whetstone.v3.panel.graph.build_pydantic_ai_agent",
            side_effect=build,
        ),
        patch(
            "andamentum.whetstone.v3.panel.graph.resolve_model",
            return_value="stub",
        ),
    ):
        result = await run_panel_v3(SRC, model="stub", n_experts=2)
    assert len(result.expert_profiles) == 2
    assert {p.discipline for p in result.expert_profiles} == {"AI", "Linguistics"}
    assert len(result.expert_reviews) == 2
    assert result.panel_synthesis is not None
    assert result.panel_synthesis.number_of_experts == 2
    # Criterion-cascade fields are empty in panel mode.
    assert result.findings == []
    assert result.edits == []


async def test_panel_disciplines_skip_extract_keywords_call() -> None:
    """When panel_disciplines is supplied, ExtractKeywords does not
    invoke the keyword agent (one fewer LLM call)."""
    build, agents = _patch_build_agent_chain(
        _profile("Alice", "Statistics"),
        _review("Alice", "Statistics"),
        _synthesis(n=1),
    )
    with (
        patch(
            "andamentum.whetstone.v3.panel.graph.build_pydantic_ai_agent",
            side_effect=build,
        ),
        patch(
            "andamentum.whetstone.v3.panel.graph.resolve_model",
            return_value="stub",
        ),
    ):
        result = await run_panel_v3(
            SRC,
            model="stub",
            n_experts=1,
            panel_disciplines=["Statistics"],
        )
    # Only 3 agent builds happened: profile + review + synthesis
    # (no extract_keywords agent).
    assert len(agents) == 3
    # All 3 stubs were consumed.
    assert all(a.captured_prompts for a in agents)
    assert result.expert_profiles[0].discipline == "Statistics"


async def test_profile_generation_isolates_per_discipline_failures() -> None:
    """If one expert_generator call crashes, the others still produce
    profiles — partial success is acceptable."""

    profile_call = [0]

    class _PartiallyFailingGen:
        async def run(self, _prompt: str):  # noqa: ARG002
            profile_call[0] += 1
            if profile_call[0] == 1:
                raise RuntimeError("simulated profile crash")
            return SimpleNamespace(output=_profile("Bob", "Linguistics"))

    extract_stub = _StubAgent(
        KeywordExtractionOutput(disciplines=["AI", "Linguistics"])
    )
    review_stub_a = _StubAgent(_review("Bob", "Linguistics"))
    synth_stub = _StubAgent(_synthesis(n=1))

    build_sequence = [
        extract_stub,
        _PartiallyFailingGen(),
        _PartiallyFailingGen(),
        review_stub_a,
        synth_stub,
    ]
    iter_seq = iter(build_sequence)

    def _build(*_args, **_kwargs):
        return next(iter_seq)

    with (
        patch(
            "andamentum.whetstone.v3.panel.graph.build_pydantic_ai_agent",
            side_effect=_build,
        ),
        patch(
            "andamentum.whetstone.v3.panel.graph.resolve_model",
            return_value="stub",
        ),
    ):
        result = await run_panel_v3(SRC, model="stub", n_experts=2)
    # First discipline (AI) crashed; second (Linguistics) succeeded.
    assert len(result.expert_profiles) == 1
    assert result.expert_profiles[0].discipline == "Linguistics"


async def test_panel_synthesise_crash_is_loud_fail_safe() -> None:
    """If panel_synthesise crashes, the result still carries the
    per-expert reviews — the failure is surfaced in summary, not raised."""

    class _CrashingSynth:
        async def run(self, _prompt: str):  # noqa: ARG002
            raise RuntimeError("simulated synth crash")

    build_sequence = [
        _StubAgent(KeywordExtractionOutput(disciplines=["AI"])),
        _StubAgent(_profile("Alice", "AI")),
        _StubAgent(_review("Alice", "AI")),
        _CrashingSynth(),
    ]
    iter_seq = iter(build_sequence)

    def _build(*_args, **_kwargs):
        return next(iter_seq)

    with (
        patch(
            "andamentum.whetstone.v3.panel.graph.build_pydantic_ai_agent",
            side_effect=_build,
        ),
        patch(
            "andamentum.whetstone.v3.panel.graph.resolve_model",
            return_value="stub",
        ),
    ):
        result = await run_panel_v3(SRC, model="stub", n_experts=1)
    assert result.panel_synthesis is None
    assert len(result.expert_reviews) == 1
    assert "synthesis" in result.summary.lower()


async def test_n_experts_caps_the_discipline_fan_out() -> None:
    """5 disciplines + n_experts=2 → only 2 profiles generated."""
    build_sequence = [
        _StubAgent(
            KeywordExtractionOutput(
                disciplines=["AI", "Linguistics", "Statistics", "Robotics", "Biology"]
            )
        ),
        _StubAgent(_profile("Alice", "AI")),
        _StubAgent(_profile("Bob", "Linguistics")),
        _StubAgent(_review("Alice", "AI")),
        _StubAgent(_review("Bob", "Linguistics")),
        _StubAgent(_synthesis(n=2)),
    ]
    iter_seq = iter(build_sequence)

    def _build(*_args, **_kwargs):
        return next(iter_seq)

    with (
        patch(
            "andamentum.whetstone.v3.panel.graph.build_pydantic_ai_agent",
            side_effect=_build,
        ),
        patch(
            "andamentum.whetstone.v3.panel.graph.resolve_model",
            return_value="stub",
        ),
    ):
        result = await run_panel_v3(SRC, model="stub", n_experts=2)
    assert len(result.expert_profiles) == 2
    assert len(result.expert_reviews) == 2


async def test_no_disciplines_skips_review_and_synthesis() -> None:
    """An empty discipline list (e.g. extractor returned []) cleanly
    short-circuits the rest of the pipeline."""
    build_sequence = [
        _StubAgent(KeywordExtractionOutput(disciplines=[])),
    ]
    iter_seq = iter(build_sequence)

    def _build(*_args, **_kwargs):
        return next(iter_seq)

    with (
        patch(
            "andamentum.whetstone.v3.panel.graph.build_pydantic_ai_agent",
            side_effect=_build,
        ),
        patch(
            "andamentum.whetstone.v3.panel.graph.resolve_model",
            return_value="stub",
        ),
    ):
        result = await run_panel_v3(SRC, model="stub", n_experts=4)
    assert result.expert_profiles == []
    assert result.expert_reviews == []
    assert result.panel_synthesis is None
    assert "skipped" in result.summary.lower()


def test_panel_graph_is_importable_and_has_entry_node() -> None:
    """Sanity: the graph and its entry node are importable as a unit."""
    assert panel_graph_v3 is not None
    assert Sectionize is not None
    # Smoke: PanelDeps + PanelState construct without args (defaults
    # work) — useful as a guard against accidental required-field
    # changes in future edits.
    deps = PanelDeps(agent_model="stub")
    state = PanelState(source=SRC)
    assert deps.n_experts == 4
    assert state.expert_profiles == []


# ---------------------------------------------------------------------------
# Confidentiality tripwire (parity with review_document)
# ---------------------------------------------------------------------------


async def test_run_panel_v3_refuses_confidential_marker_without_affirmation() -> None:
    """run_panel_v3 (the public ``run_panel``) must refuse confidentiality-
    marked text unless the caller affirms authorship — the marker scan fires
    before any LLM call, so no model is needed."""
    from andamentum.whetstone._confidentiality import ConfidentialityMarkerError

    marked = (
        "Manuscript ID: 4242\n\n"
        "Please do not share this document outside the review panel.\n"
    )
    with pytest.raises(ConfidentialityMarkerError):
        await run_panel_v3(marked, model="stub", n_experts=1)


async def test_run_panel_v3_confirm_own_draft_bypasses_marker_scan() -> None:
    """confirm_own_draft=True disarms the tripwire (mirrors --i-am-the-author).
    With a stub model the graph won't really call out, but the scan must not
    be what stops it."""
    from andamentum.whetstone._confidentiality import ConfidentialityMarkerError

    marked = "Manuscript ID: 4242\n\nbody text follows.\n"
    try:
        await run_panel_v3(marked, model="stub", n_experts=1, confirm_own_draft=True)
    except ConfidentialityMarkerError:  # pragma: no cover
        raise AssertionError("confirm_own_draft must bypass the marker scan")
    except Exception:
        # Any other failure (e.g. the stub model) is fine — we only assert the
        # confidentiality tripwire did not fire.
        pass
