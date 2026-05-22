"""Tests for claim extraction + locate-with-retry verify (mocked agent)."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.v3.extract import _ClaimSpans, _Requote, build_claims
from andamentum.whetstone.v3.model import Section


def _section(text: str) -> Section:
    return Section(id="s1", title="S", text=text, start=0, end=len(text))


def _fake_agents(*, extract_claims: list[str], requote: str = ""):
    """Patch v3.extract._agent to route by agent name to canned outputs."""

    def _factory(name: str, prompt: str, output_model, model: str):
        class _Agent:
            async def run(self, _prompt: str):
                if name == "v3_extract_claims":
                    out = _ClaimSpans(claims=extract_claims)
                else:
                    out = _Requote(quote=requote)
                return types.SimpleNamespace(output=out)

        return _Agent()

    return patch("andamentum.whetstone.v3.extract._agent", new=_factory)


async def test_verbatim_claim_kept_with_source_text() -> None:
    src = "The method is fast. It also generalises well."
    with _fake_agents(extract_claims=["The method is fast."]):
        claims = await build_claims([_section(src)], src, model="stub")
    assert len(claims) == 1
    assert claims[0].quote == "The method is fast."
    assert src[claims[0].span.start : claims[0].span.end] == "The method is fast."


async def test_hallucinated_claim_dropped_after_retries() -> None:
    src = "The method is fast."
    # Extractor invents a claim absent from the text; requote keeps failing.
    with _fake_agents(
        extract_claims=["We achieve state-of-the-art on ImageNet."],
        requote="Still not in the source.",
    ):
        claims = await build_claims([_section(src)], src, model="stub")
    assert claims == []


async def test_requote_recovers_a_locatable_span() -> None:
    src = "Our approach reduces error by forty percent on the benchmark."
    # Extractor's first quote is absent; requote returns a real verbatim span.
    with _fake_agents(
        extract_claims=["error is reduced 40%"],
        requote="reduces error by forty percent",
    ):
        claims = await build_claims([_section(src)], src, model="stub")
    assert len(claims) == 1
    assert claims[0].quote == "reduces error by forty percent"
