"""The brief slot in the three prompts a reviewer profile can reach.

A profile's brief is the one thing no argument can express, and it costs a
change to prompt text that every existing run also goes through. So the
property under test here is not "the brief appears" — it is that **an empty
brief changes nothing**: the default reviewer's criterion, probe and editor
prompts have to be byte-identical to the prompts those functions built before
profiles existed, or every run in the field starts producing different findings
because a feature nobody enabled was added.

The pin is structural rather than a stored copy of each prompt. Each test
asserts that the only difference a brief makes is a prefix, and that an empty
brief renders no prefix at all:

    prompt(brief=B) == brief_block(B) + prompt(brief="")

which forces ``prompt(brief="") == prompt-with-no-slot`` without a golden file
that would go stale the first time an unrelated line of the scaffold changes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from andamentum.whetstone.v3.checklist.bank import probe_bank_for
from andamentum.whetstone.v3.checklist.run_probes import _build_prompt
from andamentum.whetstone.v3.criteria import criterion_set_for
from andamentum.whetstone.v3.digest import build_document_model
from andamentum.whetstone.v3.editor import EditorOutput, _run_editor_on_section
from andamentum.whetstone.v3.model import Section
from andamentum.whetstone.v3.prompt_blocks import brief_block
from andamentum.whetstone.v3.review import _CriterionFindings, review_criterion
from andamentum.whetstone.v3.reviewers import PROFILES

BRIEF = PROFILES["methods"].brief

SOURCE = (
    "# Introduction\n\nWe propose a fast method.\n\n"
    "# Results\n\nWe achieve 95% accuracy on the benchmark.\n"
)

SECTIONS = [
    Section(
        id="s1", title="Introduction", text="We propose a fast method.", start=0, end=45
    ),
    Section(
        id="s2",
        title="Results",
        text="We achieve 95% accuracy on the benchmark.",
        start=45,
        end=len(SOURCE),
    ),
]


def _model():
    return build_document_model(SOURCE, SECTIONS, [])


# ── the block itself ────────────────────────────────────────────────────────


def test_an_empty_brief_renders_nothing_at_all() -> None:
    assert brief_block("") == ""


def test_whitespace_is_not_a_brief() -> None:
    """Otherwise a profile with a stray newline would silently change every
    prompt it reaches while looking, in the profile table, like no brief."""
    assert brief_block("   \n  ") == ""


def test_a_brief_renders_one_block_carrying_its_own_words() -> None:
    block = brief_block(BRIEF)
    assert block.count("REVIEWER BRIEF") == 1
    assert BRIEF in block
    assert block.endswith("\n\n")


# ── the criterion prompt ────────────────────────────────────────────────────


class _CapturingAgent:
    """Stands in for the criterion agent; keeps the prompt it was handed."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def output_validator(self, fn: Any) -> Any:
        return fn

    async def run(self, prompt: str, **_kwargs: Any) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(output=_CriterionFindings(findings=[]))


async def _criterion_prompt(brief: str) -> str:
    agent = _CapturingAgent()
    with patch("andamentum.whetstone.v3.review._build_agent", return_value=agent):
        await review_criterion(
            criterion_set_for("academic")[0],
            _model(),
            agent_model="stub:none",
            brief=brief,
        )
    return agent.prompts[0]


async def test_the_criterion_prompt_is_unchanged_by_an_empty_brief() -> None:
    plain = await _criterion_prompt("")
    assert plain.startswith("CRITERION: ")
    assert "REVIEWER BRIEF" not in plain

    briefed = await _criterion_prompt(BRIEF)
    assert briefed == brief_block(BRIEF) + plain


# ── the probe prompt ────────────────────────────────────────────────────────


def test_the_probe_prompt_is_unchanged_by_an_empty_brief() -> None:
    probe = probe_bank_for("academic")[0]
    model = _model()

    plain = _build_prompt(probe, model)
    assert plain.startswith("CHECK: ")
    assert "REVIEWER BRIEF" not in plain
    assert _build_prompt(probe, model, brief="") == plain

    briefed = _build_prompt(probe, model, brief=BRIEF)
    assert briefed == brief_block(BRIEF) + plain


def test_two_probe_prompts_with_the_same_brief_are_identical() -> None:
    """Checklist mode sells comparability run to run; a stance must not make
    the prompt depend on anything but the probe, the document and itself."""
    probe = probe_bank_for("academic")[0]
    model = _model()
    assert _build_prompt(probe, model, brief=BRIEF) == _build_prompt(
        probe, model, brief=BRIEF
    )


# ── the editor prompt ───────────────────────────────────────────────────────


class _CapturingEditorAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def output_validator(self, fn: Any) -> Any:
        return fn

    async def run(self, prompt: str, **_kwargs: Any) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(output=EditorOutput(edits=[]))


async def _editor_prompt(brief: str) -> str:
    agent = _CapturingEditorAgent()
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=agent,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        await _run_editor_on_section(
            SECTIONS[0],
            ["clarity"],
            [],
            agent_model="stub:none",
            brief=brief,
        )
    return agent.prompts[0]


async def test_the_editor_prompt_is_unchanged_by_an_empty_brief() -> None:
    plain = await _editor_prompt("")
    assert plain.startswith("SECTION TITLE: ")
    assert "REVIEWER BRIEF" not in plain

    briefed = await _editor_prompt(BRIEF)
    assert briefed == brief_block(BRIEF) + plain


# ── the default reviewer carries no brief to carry ──────────────────────────


def test_the_default_reviewer_has_nothing_to_render() -> None:
    assert brief_block(PROFILES["general"].brief) == ""


@pytest.mark.parametrize("name", [n for n in PROFILES if n != "general"])
def test_every_specialist_renders_exactly_one_block(name: str) -> None:
    block = brief_block(PROFILES[name].brief)
    assert block and block.count("REVIEWER BRIEF") == 1
