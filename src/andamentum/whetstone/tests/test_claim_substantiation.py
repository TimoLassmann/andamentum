"""Tests for cross-section claim substantiation (full-text verification).

Both agents (digest_extractor, claim_support) are stubbed; no embeddings,
no Ollama, no cosine.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from andamentum.whetstone.agents.claim_support import ClaimSupport
from andamentum.whetstone.agents.digest_extractor import RawClaim, SectionClaims
from andamentum.whetstone.deps import ReviewDeps
from andamentum.whetstone.nodes.reconcile_claims import ReconcileClaims
from andamentum.whetstone.state import ReviewState
from andamentum.whetstone.structural.types import SectionRef


@dataclass
class _Result:
    output: object


class _StubAgents:
    """Routes build_pydantic_ai_agent by name to canned outputs."""

    def __init__(self, *, claims: SectionClaims, supported: bool, reason: str = ""):
        self.claims = claims
        self.supported = supported
        self.reason = reason

    def __call__(self, name: str, model):  # mimics build_pydantic_ai_agent
        claims, supported, reason = self.claims, self.supported, self.reason

        class _Agent:
            async def run(self, prompt: str):
                if name == "digest_extractor":
                    return _Result(output=claims)
                return _Result(output=ClaimSupport(supported=supported, reason=reason))

        return _Agent()


def _section(text: str, *, id="s1", title="Results") -> SectionRef:
    return SectionRef(id=id, title=title, text=text, char_start=0, char_end=len(text))


def _ctx(state: ReviewState, deps: ReviewDeps):
    @dataclass
    class _Ctx:
        state: ReviewState
        deps: ReviewDeps

    return _Ctx(state=state, deps=deps)


def _state_with(section: SectionRef) -> ReviewState:
    state = ReviewState(source="x")
    state.sections = [section]
    # Pretend CriticalRead already classified this as reviewable.
    state.reviewable_section_ids = {section.id}
    return state


async def test_claim_with_citation_not_flagged() -> None:
    text = "The method recovers 4x more associations [12]. We measured recall."
    claims = SectionClaims(
        claims=[RawClaim(text="recovers 4x more", quote="recovers 4x more associations", has_citation=True)]
    )
    state = _state_with(_section(text))
    deps = ReviewDeps(model="stub")

    with patch(
        "andamentum.whetstone.nodes.reconcile_claims.build_pydantic_ai_agent",
        new=_StubAgents(claims=claims, supported=False, reason="should not be called"),
    ):
        await ReconcileClaims().run(_ctx(state, deps))  # type: ignore[arg-type]

    subs = [f for f in state.challenged_findings if f.category == "substantiation"]
    assert subs == []  # citation substantiates → not flagged, no verify call


async def test_unsupported_claim_flagged_with_reason() -> None:
    text = "The system is robust to irrelevant input. Throughput was high."
    claims = SectionClaims(
        claims=[RawClaim(text="system is robust to irrelevant input",
                         quote="The system is robust to irrelevant input", has_citation=False)]
    )
    state = _state_with(_section(text))
    deps = ReviewDeps(model="stub")

    with patch(
        "andamentum.whetstone.nodes.reconcile_claims.build_pydantic_ai_agent",
        new=_StubAgents(claims=claims, supported=False, reason="no robustness experiment is reported"),
    ):
        await ReconcileClaims().run(_ctx(state, deps))  # type: ignore[arg-type]

    subs = [f for f in state.challenged_findings if f.category == "substantiation"]
    assert len(subs) == 1
    assert subs[0].confidence == "low"
    assert "no robustness experiment is reported" in subs[0].rationale


async def test_supported_claim_not_flagged() -> None:
    text = "The system is robust. Under 50% noise, precision held at 0.9."
    claims = SectionClaims(
        claims=[RawClaim(text="system is robust", quote="The system is robust", has_citation=False)]
    )
    state = _state_with(_section(text))
    deps = ReviewDeps(model="stub")

    with patch(
        "andamentum.whetstone.nodes.reconcile_claims.build_pydantic_ai_agent",
        new=_StubAgents(claims=claims, supported=True, reason="precision held at 0.9 under noise"),
    ):
        await ReconcileClaims().run(_ctx(state, deps))  # type: ignore[arg-type]

    subs = [f for f in state.challenged_findings if f.category == "substantiation"]
    assert subs == []


async def test_unanchorable_claim_dropped() -> None:
    text = "Real sentence in the section."
    claims = SectionClaims(
        claims=[RawClaim(text="hallucinated", quote="THIS QUOTE IS NOT IN THE TEXT", has_citation=False)]
    )
    state = _state_with(_section(text))
    deps = ReviewDeps(model="stub")

    with patch(
        "andamentum.whetstone.nodes.reconcile_claims.build_pydantic_ai_agent",
        new=_StubAgents(claims=claims, supported=False, reason="x"),
    ):
        await ReconcileClaims().run(_ctx(state, deps))  # type: ignore[arg-type]

    subs = [f for f in state.challenged_findings if f.category == "substantiation"]
    assert subs == []  # quote can't be anchored → dropped → nothing flagged


async def test_reference_section_excluded_from_claims() -> None:
    # A section the classifier marked non-reviewable yields no claims even if
    # the extractor would have returned some.
    text = "42. Smith J et al. Some Paper. Journal 2021;1:1-2."
    claims = SectionClaims(
        claims=[RawClaim(text="a claim", quote="Some Paper", has_citation=False)]
    )
    state = ReviewState(source="x")
    state.sections = [_section(text, id="refs", title="References")]
    state.reviewable_section_ids = set()  # classifier marked nothing reviewable

    deps = ReviewDeps(model="stub")
    with patch(
        "andamentum.whetstone.nodes.reconcile_claims.build_pydantic_ai_agent",
        new=_StubAgents(claims=claims, supported=False, reason="x"),
    ):
        await ReconcileClaims().run(_ctx(state, deps))  # type: ignore[arg-type]

    subs = [f for f in state.challenged_findings if f.category == "substantiation"]
    assert subs == []
