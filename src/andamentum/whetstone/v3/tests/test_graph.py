"""End-to-end smoke test of the v3 graph with all agents stubbed."""

from __future__ import annotations

import types
from contextlib import ExitStack
from unittest.mock import patch

from andamentum.whetstone.v3.consolidate import _Consolidation
from andamentum.whetstone.v3.extract import _ClaimSpans, _Requote
from andamentum.whetstone.v3.gaps import _DemandList
from andamentum.whetstone.v3.graph import run_review_v3
from andamentum.whetstone.v3.review import _CriterionFindings, _RawFinding
from andamentum.whetstone.v3.synth import StructuredReview

SRC = (
    "# Introduction\n\nWe propose a fast method.\n\n"
    "# Results\n\nWe achieve 95% accuracy on the benchmark.\n"
)


def _router(defn, _model):
    """Stub for the legacy build_pydantic_ai_agent path (extract, gaps,
    consolidate, synth — everything except review, which now constructs
    its own Agent via ``review._build_agent``)."""
    name = defn.name

    class _Agent:
        def output_validator(self, fn):  # absorbs the lock-and-refine validator
            return fn

        async def run(self, _prompt, **_kwargs):
            if name == "v3_extract_claims":
                out = _ClaimSpans(
                    claims=[
                        "We propose a fast method.",
                        "We achieve 95% accuracy on the benchmark.",
                    ]
                )
            elif name == "v3_requote":
                out = _Requote(quote="")  # give up → unmatched claims drop
            elif name == "v3_gap_analysis":
                out = _DemandList(demands=[])  # no gaps → loop exits at once
            elif name == "v3_consolidate":
                out = _Consolidation(groups=[])  # no merges in this smoke test
            elif name in ("v3_synthesise", "v3_critique_revise"):
                out = StructuredReview(
                    synopsis="A short methods paper.",
                    strengths=["clear contribution"],
                    weaknesses=["accuracy claim lacks a baseline"],
                )
            else:
                out = _CriterionFindings(findings=[])
            return types.SimpleNamespace(output=out)

    return _Agent()


def _build_review_agent_stub(_criterion, _agent_model):
    """Stub for the new ``review._build_agent`` helper. Returns one finding
    per criterion regardless of which criterion is asking — the test
    only cares that the cascade produces *something* per stage."""

    class _Agent:
        def output_validator(self, fn):  # absorbs the lock-and-refine validator
            return fn

        async def run(self, _prompt, **_kwargs):
            return types.SimpleNamespace(
                output=_CriterionFindings(
                    findings=[
                        _RawFinding(
                            issue="accuracy unsupported by a baseline",
                            quote="We achieve 95% accuracy on the benchmark.",
                            severity="major",
                        )
                    ]
                )
            )

    return _Agent()


async def test_graph_runs_end_to_end_to_review_result() -> None:
    mods_using_legacy = ["extract", "gaps", "consolidate", "synth"]
    with ExitStack() as stack:
        for m in mods_using_legacy:
            stack.enter_context(
                patch(
                    f"andamentum.whetstone.v3.{m}.build_pydantic_ai_agent", new=_router
                )
            )
            stack.enter_context(
                patch(f"andamentum.whetstone.v3.{m}.resolve_model", new=lambda x: None)
            )
        # review.py no longer uses build_pydantic_ai_agent; it constructs
        # its own Agent via _build_agent. Stub that directly.
        stack.enter_context(
            patch(
                "andamentum.whetstone.v3.review._build_agent",
                new=_build_review_agent_stub,
            )
        )
        # Explicit document_type="academic" — the test asserts the
        # SPECS criterion category set, so we must skip the classifier
        # (which with model="stub" would default to "general" → GENERAL
        # criterion set with different category names).
        result = await run_review_v3(SRC, model="stub", cap=1, document_type="academic")

    # It produced a ReviewResult the renderers can consume.
    assert result.summary.startswith("## Summary")
    assert "accuracy claim lacks a baseline" in result.summary
    # Findings survived verification and carry located, section-relative quotes.
    assert result.findings
    f = result.findings[0]
    assert f.quotes and f.quotes[0].text == "We achieve 95% accuracy on the benchmark."
    assert f.category in {
        c.lower()
        for c in ("Story", "Presentation", "Evaluations", "Correctness", "Significance")
    }
    # Document map reflects the sections.
    assert result.document_map
