"""A named reviewer, from the entry point to the finished result.

Step two built the profiles and the resolver; this file pins what happens when
one is actually handed to a pipeline. Four properties, in the order they would
break:

1. **No reviewer means no change.** ``reviewer=None`` produces a run with no
   brief in any prompt and no reviewer block on the result — the run whetstone
   produced before profiles existed.
2. **A reviewer's selections reach the calls.** Its criteria run, its probes
   run in its own order, and its brief is in front of every one of them.
3. **A misfit warns and proceeds.** The run completes and the warning is on
   the result, where a reader of the report will see it — losing it to a
   terminal scrollback is the failure this is guarding.
4. **An empty criterion set costs no criterion calls and no gap rounds.** The
   line editor ships one, and the gap loop used to run its full round cap over
   nothing and label what it found with a criterion the caller never selected.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from andamentum.whetstone.v3.checklist.bank import probe_for
from andamentum.whetstone.v3.checklist.graph import run_checklist_v3
from andamentum.whetstone.v3.checklist.run_probes import _PresenceFindings
from andamentum.whetstone.v3.consolidate import _Consolidation
from andamentum.whetstone.v3.extract import _ClaimSpans, _Requote
from andamentum.whetstone.v3.extract_criteria import (
    _ExtractedCriterion,
    _ExtractionResult,
)
from andamentum.whetstone.v3.gaps import _DemandList
from andamentum.whetstone.v3.graph import run_review_v3
from andamentum.whetstone.v3.review import _CriterionFindings
from andamentum.whetstone.v3.reviewers import PROFILES
from andamentum.whetstone.v3.synth import StructuredReview

SRC = (
    "# Introduction\n\nWe propose a fast method.\n\n"
    "# Results\n\nWe achieve 95% accuracy on the benchmark.\n"
)


# ── stubs ───────────────────────────────────────────────────────────────────


def _router(defn: Any, _model: Any) -> Any:
    """The non-criterion agents: extraction, gaps, consolidation, synthesis."""
    name = defn.name

    class _Agent:
        def output_validator(self, fn: Any) -> Any:
            return fn

        async def run(self, _prompt: str, **_kwargs: Any) -> Any:
            if name == "v3_extract_claims":
                out: Any = _ClaimSpans(claims=["We propose a fast method."])
            elif name == "v3_requote":
                out = _Requote(quote="")
            elif name == "v3_extract_criteria":
                out = _ExtractionResult(
                    criteria=[
                        _ExtractedCriterion(
                            name="House rule", questions=["Is the house rule kept?"]
                        )
                    ]
                )
            elif name == "v3_gap_analysis":
                out = _DemandList(demands=[])
            elif name == "v3_consolidate":
                out = _Consolidation(groups=[])
            elif name in ("v3_synthesise", "v3_critique_revise"):
                out = StructuredReview(
                    synopsis="A short methods paper.",
                    strengths=["clear contribution"],
                    weaknesses=["no baseline"],
                )
            else:  # pragma: no cover - a new agent would have to be routed
                raise AssertionError(f"unexpected agent: {name}")
            return types.SimpleNamespace(output=out)

    return _Agent()


def _criterion_recorder(prompts: list[str]) -> Callable[..., Any]:
    def factory(_criterion: Any, _agent_model: str) -> Any:
        class _Agent:
            def output_validator(self, fn: Any) -> Any:
                return fn

            async def run(self, prompt: str, **_kwargs: Any) -> Any:
                prompts.append(prompt)
                return types.SimpleNamespace(output=_CriterionFindings(findings=[]))

        return _Agent()

    return factory


def _probe_recorder(prompts: list[str]) -> Callable[..., Any]:
    def factory(*, output_type: type, instructions: str, agent_model: str) -> Any:
        class _Agent:
            async def run(self, prompt: str, **_kwargs: Any) -> Any:
                prompts.append(prompt)
                return types.SimpleNamespace(output=_PresenceFindings(findings=[]))

        return _Agent()

    return factory


async def _review(*, prompts: list[str], **kwargs: Any) -> Any:
    with ExitStack() as stack:
        for module in ("extract", "extract_criteria", "gaps", "consolidate", "synth"):
            stack.enter_context(
                patch(
                    f"andamentum.whetstone.v3.{module}.build_pydantic_ai_agent",
                    new=_router,
                )
            )
            stack.enter_context(
                patch(
                    f"andamentum.whetstone.v3.{module}.resolve_model",
                    new=lambda _x: None,
                )
            )
        stack.enter_context(
            patch(
                "andamentum.whetstone.v3.review._build_agent",
                new=_criterion_recorder(prompts),
            )
        )
        return await run_review_v3(SRC, model="stub", cap=1, **kwargs)


async def _checklist(*, prompts: list[str], **kwargs: Any) -> Any:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "andamentum.whetstone.v3.extract.build_pydantic_ai_agent",
                new=_router,
            )
        )
        stack.enter_context(
            patch("andamentum.whetstone.v3.extract.resolve_model", new=lambda _x: None)
        )
        stack.enter_context(
            patch(
                "andamentum.whetstone.v3.checklist.run_probes._build_agent",
                new=_probe_recorder(prompts),
            )
        )
        return await run_checklist_v3(SRC, model="stub", **kwargs)


# ── 1. no reviewer means no change ──────────────────────────────────────────


async def test_no_reviewer_leaves_the_prompts_and_the_result_alone() -> None:
    prompts: list[str] = []
    result = await _review(prompts=prompts, document_type="academic")
    assert prompts, "the cascade must still have run"
    assert not any("REVIEWER BRIEF" in prompt for prompt in prompts)
    assert result.reviewer_header is None


async def test_the_default_profile_is_the_same_as_no_reviewer() -> None:
    """`--reviewer default` must not be a second code path."""
    without: list[str] = []
    with_default: list[str] = []
    await _review(prompts=without, document_type="academic")
    await _review(
        prompts=with_default,
        document_type="academic",
        reviewer=PROFILES["default"],
    )
    assert without == with_default


async def test_no_reviewer_leaves_a_checklist_run_alone() -> None:
    prompts: list[str] = []
    result = await _checklist(prompts=prompts, document_type="academic")
    assert prompts
    assert not any("REVIEWER BRIEF" in prompt for prompt in prompts)
    assert result.reviewer_header is None


# ── 2. a reviewer's selections reach the calls ──────────────────────────────


async def test_a_reviewer_supplies_its_criteria_and_its_stance() -> None:
    profile = PROFILES["methods"]
    prompts: list[str] = []
    await _review(prompts=prompts, document_type="academic", reviewer=profile)

    assert len(prompts) == len(profile.criteria or ())
    for prompt, criterion in zip(prompts, profile.criteria or ()):
        assert prompt.startswith("REVIEWER BRIEF")
        assert profile.brief in prompt
        assert f"CRITERION: {criterion.name}" in prompt


async def test_a_reviewer_supplies_its_probes_in_its_own_order() -> None:
    profile = PROFILES["reader"]
    prompts: list[str] = []
    result = await _checklist(
        prompts=prompts, document_type="academic", reviewer=profile
    )

    # One call per probe, in the profile's order, each led by the stance. The
    # probe ids are resolved per document type, so the titles are compared
    # against the same resolution the pipeline performed.
    expected = [
        probe_for(probe_id, document_type="academic").title
        for probe_id in profile.probes or ()
    ]
    titles = [
        prompt.split("CHECK: ", 1)[1].splitlines()[0].strip() for prompt in prompts
    ]
    assert titles == expected
    assert all(prompt.startswith("REVIEWER BRIEF") for prompt in prompts)
    # And the coverage table has one row per selected probe, in that order.
    rows = [title for title in expected if title in result.summary]
    assert rows == expected


def _coverage_rows(summary: str) -> list[str]:
    """The first cell of every row in the rendered coverage table."""
    return [
        line.split("|")[1].strip()
        for line in summary.splitlines()
        if line.startswith("|") and not set(line) <= set("|- ")
    ]


async def test_the_coverage_table_rows_are_the_profiles_probe_list_in_order() -> None:
    profile = PROFILES["methods"]
    result = await _checklist(prompts=[], document_type="academic", reviewer=profile)
    expected = [
        probe_for(probe_id, document_type="academic").title
        for probe_id in profile.probes or ()
    ]
    # The header row of the markdown table is the one row that is not a probe.
    assert _coverage_rows(result.summary)[1:] == expected


async def test_two_runs_of_one_reviewer_tabulate_identically() -> None:
    """Comparability row for row is what a named bank buys over a typed list."""
    first = await _checklist(
        prompts=[], document_type="academic", reviewer=PROFILES["claims"]
    )
    second = await _checklist(
        prompts=[], document_type="academic", reviewer=PROFILES["claims"]
    )
    assert _coverage_rows(first.summary) == _coverage_rows(second.summary)
    assert first.summary == second.summary


async def test_guidelines_survive_a_reviewer_that_carries_criteria() -> None:
    """The mutual-exclusion check reads the caller's arguments, not the profile.

    ``criteria`` and ``guidelines_text`` together are still a ``ValueError``.
    A profile supplying criteria is not the caller supplying them, so someone
    who passes guidelines alongside ``--reviewer claims`` must get their
    guidelines — not an error about an argument they did not use.
    """
    prompts: list[str] = []
    result = await _review(
        prompts=prompts,
        document_type="academic",
        reviewer=PROFILES["claims"],
        guidelines_text="Check that the house rule is kept.",
    )
    assert len(prompts) == 1
    assert "CRITERION: House rule" in prompts[0]
    # The stance still arrives: the guidelines replaced the questions, not the
    # reviewer.
    assert prompts[0].startswith("REVIEWER BRIEF")
    header = result.reviewer_header
    assert header is not None
    criteria_line = next(
        line for line in header.configuration if line.label == "criteria"
    )
    assert criteria_line.source == "supplied guidelines"


async def test_the_reviewer_header_reaches_the_result() -> None:
    profile = PROFILES["claims"]
    result = await _review(prompts=[], document_type="academic", reviewer=profile)
    header = result.reviewer_header
    assert header is not None
    assert header.identity == profile.identity
    assert header.brief == profile.brief
    assert header.version
    assert [line.label for line in header.configuration] == [
        "criteria",
        "probes",
        "editor criteria",
        "author",
    ]


async def test_a_checklist_run_carries_the_header_too() -> None:
    result = await _checklist(
        prompts=[], document_type="academic", reviewer=PROFILES["framing"]
    )
    assert result.reviewer_header is not None
    assert result.reviewer_header.identity == "Framing reviewer"


# ── 3. a misfit warns and proceeds ──────────────────────────────────────────


async def test_a_misfitting_reviewer_still_runs_and_says_so() -> None:
    prompts: list[str] = []
    result = await _review(
        prompts=prompts, document_type="creative", reviewer=PROFILES["methods"]
    )
    assert prompts, "a reviewer that does not fit is a warning, not a refusal"
    header = result.reviewer_header
    assert header is not None
    assert "methods reviewer on a document classified as creative" in header.fit_warning
    assert "--document-type" in header.fit_warning


async def test_a_fitting_reviewer_warns_about_nothing() -> None:
    result = await _review(
        prompts=[], document_type="academic", reviewer=PROFILES["methods"]
    )
    assert result.reviewer_header is not None
    assert result.reviewer_header.fit_warning == ""


# ── 4. an empty criterion set is cheap ──────────────────────────────────────


async def test_the_line_editor_makes_no_criterion_call_and_no_gap_round() -> None:
    prompts: list[str] = []
    calls: list[Any] = []

    async def _gap_loop(*args: Any, **kwargs: Any) -> list[Any]:  # pragma: no cover
        calls.append(kwargs)
        return list(args[1])

    with patch("andamentum.whetstone.v3.graph.gap_loop", new=_gap_loop):
        result = await _review(
            prompts=prompts, document_type="academic", reviewer=PROFILES["editor"]
        )

    assert prompts == []
    assert calls == [], "the gap loop has no coverage to re-examine"
    # The run still completed: claim extraction, synthesis and self-critique
    # are unconditional, which is why the profile's description says so.
    assert result.summary
    assert result.reviewer_header is not None
