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
from typing import Any
from unittest.mock import patch

import pytest

from andamentum.whetstone.pipeline.model import Span
from andamentum.whetstone.pipeline.review import Finding, Severity
from pydantic_ai import ModelRetry

from andamentum.whetstone.pipeline.editor import (
    DEFAULT_EDITOR_CRITERIA,
    EditorOutput,
    EditProposal,
    make_edit_validator,
    run_editor,
)
from andamentum.whetstone.pipeline.model import Section


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
        self.validator = None

    def output_validator(self, fn):
        """Record it. The real agent runs this on every completion; these
        fakes return a payload directly, so the tests that care about the
        validator call it themselves (see test_validator_*)."""
        self.validator = fn
        return fn

    async def run(self, prompt: str):
        self.captured_prompts.append(prompt)
        out = self._outputs[self._calls % len(self._outputs)]
        self._calls += 1
        return SimpleNamespace(output=out)


def _patch_editor(outputs: list[EditorOutput]) -> _FakeAgent:
    """Helper to construct a fake agent + the patches that route both
    build_pydantic_ai_agent and resolve_model in editor through it."""
    fake = _FakeAgent(outputs)
    return fake


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_run_editor_empty_sections_returns_empty() -> None:
    """Defensive: an empty section list must not call the agent."""
    fake = _patch_editor([])
    with (
        patch(
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
    ):
        edits, _declined = await run_editor(
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
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
    ):
        edits, _declined = await run_editor(
            SECTIONS, criteria=["grammar"], agent_model="stub"
        )
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
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
    ):
        edits, _declined = await run_editor(
            SECTIONS, criteria=DEFAULT_EDITOR_CRITERIA, agent_model="stub"
        )
    assert edits == []


async def test_editor_criteria_flows_into_prompt() -> None:
    """Caller-supplied editor_criteria must reach the user prompt the
    agent sees (so the model knows which checks to apply)."""
    fake = _patch_editor([EditorOutput(edits=[])])
    with (
        patch(
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
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

        def output_validator(self, fn):
            return fn

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
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
    ):
        edits, _declined = await run_editor(
            SECTIONS, criteria=["concision"], agent_model="stub"
        )
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
        def output_validator(self, fn):
            return fn

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
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
    ):
        edits, _declined = await run_editor(
            many_sections, criteria=["grammar"], agent_model="stub"
        )
    assert edits == []
    assert peak[0] <= 5, f"concurrency cap violated: peak={peak[0]}"


async def test_editor_disabled_by_default_via_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: run_review with editor=False (the default) skips
    the EditSections node — no editor LLM call, no Edits on the result."""
    from andamentum.whetstone.pipeline import graph as graph_mod

    called = {"editor": False}

    async def _spy_run_editor(*_args, **_kwargs):
        called["editor"] = True
        return []

    # Patch the symbol used by EditSections.run via its imported alias
    monkeypatch.setattr(graph_mod, "run_editor", _spy_run_editor)

    # Stub the whole graph: we only want to verify EditSections gating.
    # Easiest: directly construct the node + state + deps and call run().
    from andamentum.whetstone.pipeline.graph import EditSections, Finalize, ReviewDeps, ReviewState

    state = ReviewState(source="...")
    deps = ReviewDeps(agent_model="stub", editor_enabled=False)

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
    from andamentum.whetstone.pipeline import graph as graph_mod
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
        return [sentinel], []

    monkeypatch.setattr(graph_mod, "run_editor", _stub_run_editor)

    from andamentum.whetstone.pipeline.graph import EditSections, Finalize, ReviewDeps, ReviewState

    state = ReviewState(source="...")
    deps = ReviewDeps(agent_model="stub", editor_enabled=True)

    class _Ctx:
        def __init__(self, state, deps):
            self.state = state
            self.deps = deps

    node = EditSections()
    result = await node.run(_Ctx(state, deps))
    assert isinstance(result, Finalize)
    assert state.edits == [sentinel]


# ── the editor answering the review ─────────────────────────────────────


def _finding(
    section_id: str,
    start: int,
    end: int,
    *,
    criterion: str = "Story",
    severity: Severity = "major",
    issue: str = "the claim is unsupported",
    quote: str = "q",
):
    return Finding(
        criterion=criterion,
        issue=issue,
        quote=quote,
        severity=severity,
        span=Span(section_id=section_id, start=start, end=end),
    )


async def _run(outputs, findings, sections=None):
    fake = _patch_editor(outputs)
    with (
        patch(
            "andamentum.whetstone.pipeline.editor.build_pydantic_ai_agent",
            return_value=fake,
        ),
        patch("andamentum.whetstone.pipeline.editor.resolve_model", return_value="stub"),
    ):
        edits, declined = await run_editor(
            sections if sections is not None else SECTIONS,
            criteria=["clarity"],
            agent_model="stub",
            findings=findings,
        )
    return edits, declined, fake


async def test_the_findings_for_a_section_reach_its_prompt() -> None:
    """The whole point: the editor is told what the review found."""
    section = SECTIONS[0]
    finding = _finding(
        section.id,
        section.start,
        section.start + 5,
        issue="the sample size is never stated",
    )
    _edits, _declined, fake = await _run(
        [EditorOutput(edits=[])], [finding], sections=[section]
    )
    prompt = fake.captured_prompts[0]
    assert "REVIEWER FINDINGS" in prompt
    assert "the sample size is never stated" in prompt
    assert "[Story/major]" in prompt


async def test_a_section_with_no_findings_says_so() -> None:
    """Silence would read as an omission; the prompt states it explicitly."""
    _edits, _declined, fake = await _run(
        [EditorOutput(edits=[])], [], sections=[SECTIONS[0]]
    )
    assert "REVIEWER FINDINGS for this section: none" in fake.captured_prompts[0]


async def test_a_finding_with_no_span_is_never_sent() -> None:
    """It names something missing, or the document as a whole. Inviting a
    rewrite for it is inviting invention."""
    homeless = Finding(
        criterion="Story", issue="the limitations are never stated", quote="", span=None
    )
    _edits, declined, fake = await _run(
        [EditorOutput(edits=[])], [homeless], sections=[SECTIONS[0]]
    )
    assert "the limitations are never stated" not in fake.captured_prompts[0]
    # It is not counted as declined either — it was never asked.
    assert declined == []


async def test_declining_a_finding_records_the_reason() -> None:
    """`can_fix=false` is the right answer for a structural finding, and the
    reason has to survive to the author."""
    section = SECTIONS[0]
    finding = _finding(
        section.id,
        section.start,
        section.start + 5,
        issue="the evaluation is missing entirely",
    )
    edits, declined, _fake = await _run(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        addresses=1,
                        can_fix=False,
                        title="needs the author",
                        rationale="-",
                        why="this needs a new experiment, not a rewrite",
                    )
                ]
            )
        ],
        [finding],
        sections=[section],
    )
    assert edits == []
    assert len(declined) == 1
    assert declined[0].issue == "the evaluation is missing entirely"
    assert declined[0].why == "this needs a new experiment, not a rewrite"
    assert declined[0].criterion == "Story"


async def test_a_finding_the_editor_ignored_is_still_reported() -> None:
    """Silence is not a decision, and it would otherwise read as one."""
    section = SECTIONS[0]
    finding = _finding(section.id, section.start, section.start + 5)
    _edits, declined, _fake = await _run(
        [EditorOutput(edits=[])], [finding], sections=[section]
    )
    assert len(declined) == 1
    assert "did not respond" in declined[0].why


async def test_an_edit_answering_a_finding_says_which_one() -> None:
    section = SECTIONS[0]
    target = section.text.split()[0]
    finding = _finding(
        section.id, section.start, section.start + 5, issue="this phrasing overclaims"
    )
    edits, declined, _fake = await _run(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        addresses=1,
                        can_fix=True,
                        title="hedge the claim",
                        rationale="softens an overstated verb",
                        original_text=target,
                        new_text="Perhaps",
                    )
                ]
            )
        ],
        [finding],
        sections=[section],
    )
    assert len(edits) == 1
    # By id, not by copying the finding into the rationale: the renderers show
    # the rewrite under its finding, and the copy said the same thing twice.
    assert edits[0].finding_id == finding.id
    assert edits[0].rationale == "softens an overstated verb"
    assert declined == []


async def test_a_declined_finding_names_the_finding_it_declined() -> None:
    section = SECTIONS[0]
    finding = _finding(section.id, section.start, section.start + 5)
    _edits, declined, _fake = await _run(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        addresses=1,
                        can_fix=False,
                        title="needs data",
                        rationale="",
                        why="Report the sample size per group.",
                    )
                ]
            )
        ],
        [finding],
        sections=[section],
    )
    assert [d.finding_id for d in declined] == [finding.id]


async def test_a_rewrite_that_did_not_quote_the_document_leaves_it_unanswered() -> None:
    """The edit is dropped for being unanchorable — and the finding must not
    look handled because an attempt was made."""
    section = SECTIONS[0]
    finding = _finding(section.id, section.start, section.start + 5)
    edits, declined, _fake = await _run(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        addresses=1,
                        can_fix=True,
                        title="x",
                        rationale="y",
                        original_text="text that is nowhere in the section",
                        new_text="z",
                    )
                ]
            )
        ],
        [finding],
        sections=[section],
    )
    assert edits == []
    assert len(declined) == 1
    assert "verbatim" in declined[0].why


async def test_one_finding_may_need_two_changes() -> None:
    """A finding can span more than one sentence — "this paragraph overclaims
    in several places" is answered by several rewrites, not one.

    An earlier version deduplicated by finding number and threw the second
    away. Only genuine SPAN collisions are a problem, and the surgery layer
    already refuses those and reports them; refusing here as well discarded
    edits that never conflicted with anything.
    """
    section = SECTIONS[0]
    words = section.text.split()
    finding = _finding(section.id, section.start, section.start + 5)
    edits, _declined, _fake = await _run(
        [
            EditorOutput(
                edits=[
                    EditProposal(
                        addresses=1,
                        can_fix=True,
                        title="a",
                        rationale="a",
                        original_text=words[0],
                        new_text="A",
                    ),
                    EditProposal(
                        addresses=1,
                        can_fix=True,
                        title="b",
                        rationale="b",
                        original_text=words[1],
                        new_text="B",
                    ),
                ]
            )
        ],
        [finding],
        sections=[section],
    )
    assert len(edits) == 2
    assert {e.new_text for e in edits} == {"A", "B"}


# ── the output validator ────────────────────────────────────────────────────
#
# Everything the editor's schema relies on is a prompt rule, so the validator
# is the only thing standing between a small model's mistake and a silently
# discarded section. These call it directly: the fakes above return a payload
# without ever running it.


def _ctx(retry: int) -> Any:
    """Stand-in for pydantic-ai's RunContext. The validator reads two
    attributes off it; constructing a real one needs a live model."""
    return SimpleNamespace(retry=retry, partial_output=False)


SECTION_TEXT = "The evidence layer earns its place. The results are clear."


def _prop(
    *,
    title: str = "t",
    rationale: str = "r",
    addresses: int = 0,
    can_fix: bool = True,
    why: str = "",
    original_text: str = "",
    new_text: str = "",
) -> EditProposal:
    return EditProposal(
        title=title,
        rationale=rationale,
        addresses=addresses,
        can_fix=can_fix,
        why=why,
        original_text=original_text,
        new_text=new_text,
    )


async def test_validator_locks_a_good_entry_and_asks_only_about_the_bad_one():
    """A retry must be able to add, never to subtract.

    The review agent moved to lock-and-refine because a model asked to redo
    its whole output reintroduced errors into quotes it had already got right.
    The editor inherits both the pattern and the reason.
    """
    validate = make_edit_validator(SECTION_TEXT, 0)
    good = _prop(
        original_text="The results are clear.", new_text="The results are unambiguous."
    )
    bad = _prop(
        title="paraphrased", original_text="the results were clear", new_text="x"
    )

    with pytest.raises(ModelRetry) as exc:
        await validate(_ctx(0), EditorOutput(edits=[good, bad]))

    message = str(exc.value)
    assert "LOCKED (1 entry" in message
    assert "DO NOT re-emit" in message
    assert "the results were clear" in message  # names the text to fix
    assert "The results are clear." not in message  # never mentions the good one

    # The model drops the good entry on its retry. It survives anyway.
    out = await validate(_ctx(1), EditorOutput(edits=[]))
    assert [e.original_text for e in out.edits] == ["The results are clear."]


async def test_validator_names_an_unanswered_finding():
    """A finding with no entry used to vanish. Now the model is told."""
    validate = make_edit_validator(SECTION_TEXT, 2)
    answered = _prop(addresses=1, can_fix=False, why="needs a restructure")

    with pytest.raises(ModelRetry) as exc:
        await validate(_ctx(0), EditorOutput(edits=[answered]))
    assert "NOT YET ANSWERED: finding(s) 2" in str(exc.value)


async def test_validator_rejects_a_finding_number_that_does_not_exist():
    validate = make_edit_validator(SECTION_TEXT, 2)
    stray = _prop(addresses=7, original_text="The results are clear.", new_text="x")

    with pytest.raises(ModelRetry) as exc:
        await validate(_ctx(0), EditorOutput(edits=[stray]))
    assert "addresses=7" in str(exc.value)
    assert "use 1-2" in str(exc.value)


async def test_validator_gives_up_and_keeps_what_anchored():
    """Attempts are bounded. What survives is the lock, not the last attempt."""
    validate = make_edit_validator(SECTION_TEXT, 1)
    good = _prop(addresses=1, original_text="The results are clear.", new_text="Clear.")
    bad = _prop(original_text="not in the document", new_text="x")

    with pytest.raises(ModelRetry):
        await validate(_ctx(0), EditorOutput(edits=[good, bad]))
    with pytest.raises(ModelRetry):
        await validate(_ctx(1), EditorOutput(edits=[bad]))

    out = await validate(_ctx(2), EditorOutput(edits=[bad]))
    assert [e.new_text for e in out.edits] == ["Clear."]


async def test_validator_keeps_both_rewrites_for_one_finding():
    """Locked by (finding, quote), so a finding needing two changes keeps both."""
    validate = make_edit_validator(SECTION_TEXT, 1)
    first = _prop(
        addresses=1, original_text="The evidence layer earns its place.", new_text="A."
    )
    second = _prop(addresses=1, original_text="The results are clear.", new_text="B.")

    out = await validate(_ctx(0), EditorOutput(edits=[first, second]))
    assert {e.new_text for e in out.edits} == {"A.", "B."}
