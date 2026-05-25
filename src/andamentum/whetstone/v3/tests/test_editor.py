"""Editor phase for v3.

The agent + anchoring logic is identical to v2's editor; these tests
cover the v3-specific wiring: the EditSections node fires only when
`editor=True`, populates `state.edits`, feeds ReviewResult.edits with
section-local char offsets, and silently drops EditProposals whose
original_text cannot be anchored in the section source.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from andamentum.whetstone.v3.editor import (
    DEFAULT_EDITOR_CRITERIA,
    EditorOutput,
    EditProposal,
    run_editor,
)
from andamentum.whetstone.v3.model import Section


# ── Fixtures ────────────────────────────────────────────────────────────────


SECTIONS = [
    Section(
        id="s1",
        title="Introduction",
        text=(
            "The data shows a clear trend. Many studies have looked at "
            "this problem from different angles."
        ),
        start=0,
        end=120,
    ),
    Section(
        id="s2",
        title="Methods",
        text=(
            "We performed an analysis of the corpus using standard "
            "techniques. The corpus contained 1247 documents."
        ),
        start=120,
        end=260,
    ),
]


class _FakeAgent:
    """Minimal AsyncAgent stand-in for the v3 editor.

    Each instance is configured with a list of EditorOutput payloads
    to return on successive `.run()` calls (cycles if exhausted)."""

    def __init__(self, outputs: list[EditorOutput]) -> None:
        if not outputs:
            outputs = [EditorOutput(edits=[])]
        self._outputs = outputs
        self._calls = 0
        self.captured_prompts: list[str] = []

    async def run(self, prompt: str):
        self.captured_prompts.append(prompt)
        out = self._outputs[self._calls % len(self._outputs)]
        self._calls += 1
        return SimpleNamespace(output=out)


def _patch_editor(outputs: list[EditorOutput]) -> _FakeAgent:
    """Helper to construct a fake agent + the patches that route both
    build_pydantic_ai_agent and resolve_model in v3.editor through it."""
    fake = _FakeAgent(outputs)
    return fake


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_run_editor_empty_sections_returns_empty() -> None:
    """Defensive: an empty section list must not call the agent."""
    fake = _patch_editor([])
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        edits = await run_editor(
            [], criteria=DEFAULT_EDITOR_CRITERIA, agent_model="stub"
        )
    assert edits == []
    assert fake._calls == 0


async def test_run_editor_anchors_proposals_to_section_text() -> None:
    """An EditProposal whose original_text is verbatim in a section
    becomes an Edit with valid section-local char offsets that
    round-trip through section.text[char_start:char_end]."""
    fake = _patch_editor(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        title="Subject-verb agreement",
                        rationale="Plural noun takes plural verb.",
                        original_text="The data shows a clear trend.",
                        new_text="The data show a clear trend.",
                    )
                ]
            ),
            EditorOutput(edits=[]),
        ]
    )
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        edits = await run_editor(SECTIONS, criteria=["grammar"], agent_model="stub")
    assert len(edits) == 1
    e = edits[0]
    assert e.section_id == "s1"
    assert e.original_text == "The data shows a clear trend."
    assert e.new_text == "The data show a clear trend."
    # Round-trip: section.text[char_start:char_end] should equal original_text.
    s1 = SECTIONS[0]
    assert s1.text[e.char_start : e.char_end] == e.original_text


async def test_run_editor_drops_unanchored_proposals_silently() -> None:
    """A proposal whose original_text isn't present in any section is
    dropped — locate returns None, the helper logs at debug, no Edit
    is produced, and no exception escapes."""
    fake = _patch_editor(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        title="Fabricated",
                        rationale="LLM made this up.",
                        original_text="THIS PHRASE DOES NOT APPEAR ANYWHERE",
                        new_text="anything",
                    )
                ]
            ),
            EditorOutput(edits=[]),
        ]
    )
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        edits = await run_editor(
            SECTIONS, criteria=DEFAULT_EDITOR_CRITERIA, agent_model="stub"
        )
    assert edits == []


async def test_editor_criteria_flows_into_prompt() -> None:
    """Caller-supplied editor_criteria must reach the user prompt the
    agent sees (so the model knows which checks to apply)."""
    fake = _patch_editor([EditorOutput(edits=[])])
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        await run_editor(
            SECTIONS[:1],
            criteria=["voice", "tense"],
            agent_model="stub",
        )
    assert any("voice" in p and "tense" in p for p in fake.captured_prompts), (
        f"editor_criteria did not reach prompt: {fake.captured_prompts!r}"
    )


async def test_run_editor_isolates_per_section_failures() -> None:
    """If one section's agent call crashes, the others still produce
    edits — a crash should not abort the whole pass."""

    class _PartiallyFailing:
        def __init__(self) -> None:
            self._calls = 0

        async def run(self, _prompt: str):  # noqa: ARG002
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("simulated agent crash on section 1")
            return SimpleNamespace(
                output=EditorOutput(
                    edits=[
                        EditProposal(
                            title="Concision",
                            rationale="Replace nominalisation with a verb.",
                            original_text="performed an analysis of",
                            new_text="analysed",
                        )
                    ]
                )
            )

    fake = _PartiallyFailing()
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        edits = await run_editor(SECTIONS, criteria=["concision"], agent_model="stub")
    # First section crashed → no edits from it. Second section produced 1.
    assert len(edits) == 1
    assert edits[0].section_id == "s2"


async def test_run_editor_respects_concurrency_cap() -> None:
    """The 5-concurrent semaphore bounds parallel agent calls. Fan out
    over 12 synthetic sections and observe the peak concurrency never
    exceeds 5."""
    in_flight: list[int] = [0]
    peak: list[int] = [0]
    lock = asyncio.Lock()

    class _CountingAgent:
        async def run(self, _prompt: str):  # noqa: ARG002
            async with lock:
                in_flight[0] += 1
                if in_flight[0] > peak[0]:
                    peak[0] = in_flight[0]
            try:
                await asyncio.sleep(0.01)
            finally:
                async with lock:
                    in_flight[0] -= 1
            return SimpleNamespace(output=EditorOutput(edits=[]))

    many_sections = [
        Section(id=f"s{i}", title=f"Sec {i}", text=f"body {i}", start=0, end=1)
        for i in range(12)
    ]
    fake = _CountingAgent()
    with (
        patch(
            "andamentum.whetstone.v3.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.v3.editor.resolve_model", return_value="stub"),
    ):
        edits = await run_editor(
            many_sections, criteria=["grammar"], agent_model="stub"
        )
    assert edits == []
    assert peak[0] <= 5, f"concurrency cap violated: peak={peak[0]}"


async def test_editor_disabled_by_default_via_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: run_review_v3 with editor=False (the default) skips
    the EditSections node — no editor LLM call, no Edits on the result."""
    from andamentum.whetstone.v3 import graph as graph_mod

    called = {"editor": False}

    async def _spy_run_editor(*_args, **_kwargs):
        called["editor"] = True
        return []

    # Patch the symbol used by EditSections.run via its imported alias
    monkeypatch.setattr(graph_mod, "run_editor", _spy_run_editor)

    # Stub the whole graph: we only want to verify EditSections gating.
    # Easiest: directly construct the node + state + deps and call run().
    from andamentum.whetstone.v3.graph import EditSections, Finalize, V3Deps, V3State

    state = V3State(source="...")
    deps = V3Deps(agent_model="stub", editor_enabled=False)

    class _Ctx:
        def __init__(self, state, deps):
            self.state = state
            self.deps = deps

    node = EditSections()
    result = await node.run(_Ctx(state, deps))
    assert isinstance(result, Finalize)
    assert called["editor"] is False
    assert state.edits == []


async def test_editor_enabled_via_graph_invokes_run_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: editor=True fires the editor pass and populates
    state.edits before flowing to Finalize."""
    from andamentum.whetstone.v3 import graph as graph_mod
    from andamentum.whetstone.schemas import Edit

    sentinel = Edit(
        title="t",
        rationale="r",
        section_id="s1",
        char_start=0,
        char_end=4,
        original_text="abcd",
        new_text="ABCD",
    )

    async def _stub_run_editor(*_args, **_kwargs):
        return [sentinel]

    monkeypatch.setattr(graph_mod, "run_editor", _stub_run_editor)

    from andamentum.whetstone.v3.graph import EditSections, Finalize, V3Deps, V3State

    state = V3State(source="...")
    deps = V3Deps(agent_model="stub", editor_enabled=True)

    class _Ctx:
        def __init__(self, state, deps):
            self.state = state
            self.deps = deps

    node = EditSections()
    result = await node.run(_Ctx(state, deps))
    assert isinstance(result, Finalize)
    assert state.edits == [sentinel]
