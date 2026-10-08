"""Tests for WebSearchGatherer evidence gathering."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from andamentum.deep_research import Finding, Passage, ResearchResult
from andamentum.references import Work

from ..evidence_gathering import GATHER_MAX_ROUNDS, WebSearchGatherer
from ..operations import GatheredEvidence


def _work(url: str = "http://example.com/page1", title: str = "Test Page") -> Work:
    return Work(
        work_id="w1", url=url, title=title, venue="example.com", access="metadata"
    )


def _passage(text: str, *, work_id: str = "w1", start: int = 0) -> Passage:
    return Passage.make(work_id=work_id, start=start, text=text)


def _finding(
    passage: Passage,
    *,
    quote: str,
    number: int = 1,
    verified: bool = True,
    claim: str = "The page states the fact.",
) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id="sq1",
        claim=claim,
        quote=quote,
        quote_verified=verified,
        passage_id=passage.passage_id,
        work_id=passage.work_id,
    )


def _result(
    *, works: list[Work], passages: list[Passage], findings: list[Finding], **extra
) -> ResearchResult:
    return ResearchResult(
        run_id="r1",
        question="q",
        mode="web",
        works=works,
        passages=passages,
        findings=findings,
        **extra,
    )


def _run_with(result: ResearchResult):
    """Patch the entry point the gatherer calls; return the mock."""
    return patch(
        "andamentum.deep_research.research",
        new_callable=AsyncMock,
        return_value=result,
    )


class TestWebSearchGathererFindings:
    """WebSearchGatherer turns verified findings into evidence."""

    @pytest.mark.asyncio
    async def test_finding_becomes_evidence_with_context(self):
        text = "Before the quote. " * 100 + "Water boils at 100 C." + " After." * 100
        passage = _passage(text)
        result = _result(
            works=[_work()],
            passages=[passage],
            findings=[_finding(passage, quote="Water boils at 100 C.")],
        )

        gatherer = WebSearchGatherer(model="test", browser_cdp_url="http://chrome.test")
        with _run_with(result) as run:
            gathered = await gatherer.gather("web_search", "boiling point of water")

        assert len(gathered) == 1
        ev = gathered[0]
        assert "Water boils at 100 C." in ev.content
        # context is a window around the quote, not the whole passage
        assert len(ev.content) < len(text)
        assert ev.source_ref == "http://example.com/page1"
        assert ev.source_type == "web_search"
        assert ev.evidence_kind == "web_page"
        assert ev.quality_score is None
        sd = ev.structured_data
        assert sd["claim"] == "The page states the fact."
        assert sd["quote"] == "Water boils at 100 C."
        assert sd["quote_verified"] is True
        assert sd["title"] == "Test Page"
        assert sd["site"] == "example.com"

        args, kwargs = run.call_args
        assert args == ("boiling point of water",)
        assert kwargs["mode"] == "web"
        assert kwargs["model"] == "test"
        assert kwargs["budget"].max_rounds == GATHER_MAX_ROUNDS
        assert kwargs["sources"].browser_cdp_url == "http://chrome.test"

    @pytest.mark.asyncio
    async def test_unverified_finding_is_not_evidence_and_is_logged(self, caplog):
        passage = _passage("The page says something else entirely.")
        result = _result(
            works=[_work()],
            passages=[passage],
            findings=[
                _finding(passage, quote="a quote that is not there", verified=False),
                _finding(passage, quote="something else", number=2),
            ],
        )

        gatherer = WebSearchGatherer(model="test", browser_cdp_url="http://chrome.test")
        with _run_with(result), caplog.at_level("WARNING"):
            gathered = await gatherer.gather("web_search", "q")

        assert len(gathered) == 1
        assert gathered[0].structured_data["quote"] == "something else"
        assert "1 of 2 findings" in caplog.text

    @pytest.mark.asyncio
    async def test_no_verified_findings_returns_empty(self):
        passage = _passage("Some text.")
        result = _result(
            works=[_work()],
            passages=[passage],
            findings=[_finding(passage, quote="nope", verified=False)],
        )
        gatherer = WebSearchGatherer(model="test", browser_cdp_url="http://chrome.test")
        with _run_with(result):
            assert await gatherer.gather("web_search", "q") == []

    @pytest.mark.asyncio
    async def test_doi_link_when_the_work_has_no_url(self):
        work = Work(work_id="w1", title="A paper", doi="10.1234/abc", access="metadata")
        passage = _passage("A finding sits here.")
        result = _result(
            works=[work],
            passages=[passage],
            findings=[_finding(passage, quote="A finding sits here.")],
        )
        gatherer = WebSearchGatherer(model="test", browser_cdp_url="http://chrome.test")
        with _run_with(result):
            gathered = await gatherer.gather("web_search", "q")
        assert gathered[0].source_ref == "https://doi.org/10.1234/abc"

    @pytest.mark.asyncio
    async def test_finding_of_unknown_work_raises(self):
        passage = _passage("Text.")
        result = _result(
            works=[], passages=[passage], findings=[_finding(passage, quote="Text.")]
        )
        gatherer = WebSearchGatherer(model="test", browser_cdp_url="http://chrome.test")
        with _run_with(result), pytest.raises(RuntimeError, match="not among"):
            await gatherer.gather("web_search", "q")

    @pytest.mark.asyncio
    async def test_searxng_url_reaches_the_sources_and_nothing_is_started(
        self, monkeypatch
    ):
        # the environment names another instance: it must be ignored
        monkeypatch.setenv("SEARXNG_URL", "http://env.invalid:1")
        result = _result(works=[], passages=[], findings=[])
        with (
            _run_with(result) as run,
            patch("subprocess.run") as started,
        ):
            await WebSearchGatherer(
                model="test", searxng_url="http://searx.test"
            ).gather("web_search", "q")
        assert run.call_args.kwargs["sources"].searxng_url == "http://searx.test"
        started.assert_not_called()

    def test_without_an_address_construction_is_a_clear_error(self):
        with pytest.raises(ValueError, match="searxng_url"):
            WebSearchGatherer(model="test")

    def test_the_default_gatherer_needs_an_address_for_web_search(self):
        from ..evidence_gathering import get_default_gatherer

        with pytest.raises(ValueError, match="searxng_url"):
            get_default_gatherer(model="test")
        gatherer = get_default_gatherer(model="test", searxng_url="http://searx.test")
        assert gatherer.searxng_url == "http://searx.test"  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────────
# CompositeGatherer — silent fallback removal tests (Task 12)
# ──────────────────────────────────────────────────────────────────────────────


class TestCompositeGathererNoSilentFallback:
    """Assert that CompositeGatherer does NOT silently fall back to web search
    when a specifically-requested provider fails or returns empty."""

    async def test_specific_provider_failure_raises(self):
        """When the specifically-requested provider fails, CompositeGatherer must
        raise — not silently fall through to web search."""
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        class _FailingProvider:
            async def gather(self, query: str) -> list[GatheredEvidence]:
                raise RuntimeError("clinicaltrials.gov unreachable")

        class _UnreachableWebSearch:
            async def gather(
                self, source_type: str, query: str
            ) -> list[GatheredEvidence]:
                raise AssertionError(
                    "web search should not be called on provider failure"
                )

        gatherer = CompositeGatherer(
            web_search=_UnreachableWebSearch(),
            providers={"clinicaltrials": _FailingProvider()},
        )
        with pytest.raises(RuntimeError, match="clinicaltrials.gov unreachable"):
            await gatherer.gather("clinicaltrials", "diabetes prevention")

    async def test_specific_provider_empty_returns_empty(self):
        """When the specifically-requested provider returns empty, CompositeGatherer
        must return empty — not silently fall through to web search."""
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        class _EmptyProvider:
            async def gather(self, query: str) -> list[GatheredEvidence]:
                return []

        class _UnreachableWebSearch:
            async def gather(
                self, source_type: str, query: str
            ) -> list[GatheredEvidence]:
                raise AssertionError(
                    "web search should not be called on empty provider"
                )

        gatherer = CompositeGatherer(
            web_search=_UnreachableWebSearch(),
            providers={"chembl": _EmptyProvider()},
        )
        assert await gatherer.gather("chembl", "aspirin") == []

    async def test_all_aggregates_successes_logs_failures(self):
        """source_type='all' iterates all providers + web. Per-provider failures
        are logged but do not prevent aggregation of successes."""
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        good_result = GatheredEvidence(
            content="Good provider result",
            source_ref="good-source",
            source_type="openalex",
            quality_score=0.8,
        )
        web_result = GatheredEvidence(
            content="Web result",
            source_ref="web-source",
            source_type="web_search",
            quality_score=0.5,
        )

        class _GoodProvider:
            async def gather(self, query: str) -> list[GatheredEvidence]:
                return [good_result]

        class _RaisingProvider:
            async def gather(self, query: str) -> list[GatheredEvidence]:
                raise RuntimeError("openalex down")

        class _GoodWebSearch:
            async def gather(
                self, source_type: str, query: str
            ) -> list[GatheredEvidence]:
                return [web_result]

        gatherer = CompositeGatherer(
            web_search=_GoodWebSearch(),
            providers={"good": _GoodProvider(), "failing": _RaisingProvider()},
        )
        results = await gatherer.gather("all", "test query")

        contents = [r.content for r in results]
        assert "Good provider result" in contents
        assert "Web result" in contents
        assert len(results) == 2

    async def test_all_raises_when_every_call_fails(self):
        """source_type='all': if every provider AND web search fail, raise RuntimeError."""
        from andamentum.epistemic.evidence_gathering import CompositeGatherer

        class _FailingProvider:
            async def gather(self, query: str) -> list[GatheredEvidence]:
                raise RuntimeError("provider down")

        class _FailingWebSearch:
            async def gather(
                self, source_type: str, query: str
            ) -> list[GatheredEvidence]:
                raise ConnectionError("SearXNG down")

        gatherer = CompositeGatherer(
            web_search=_FailingWebSearch(),
            providers={"p1": _FailingProvider(), "p2": _FailingProvider()},
        )
        with pytest.raises(RuntimeError, match="All gather calls failed"):
            await gatherer.gather("all", "test query")


class TestWebSearchAddressIsRequiredAtTheStart:
    @pytest.mark.asyncio
    async def test_the_graph_refuses_to_start_without_an_address(self, tmp_path):
        from ..graph import run_epistemic_graph

        with pytest.raises(ValueError, match="searxng_url"):
            await run_epistemic_graph(
                question="Is exercise good for cardiovascular health?",
                database_name="no_address",
                db_dir=str(tmp_path),
                model="fake:test-model",
                embedding_model="fake-embeddings",
            )
        assert list(tmp_path.iterdir()) == []  # nothing was created

    @pytest.mark.asyncio
    async def test_preflight_skips_web_search_without_an_address(self):
        from ..preflight import preflight

        result = await preflight(model="fake:test-model", providers={})
        web = [c for c in result.checks if c.name == "WebSearch"]
        assert [c.status for c in web] == ["skip"]

    def test_the_commands_take_the_address(self):
        from ..cli import _build_parser as build_parser

        parser = build_parser()
        for command in (["ask", "q"], ["verify", "c"], ["preflight"]):
            args = parser.parse_args([*command, "--searxng-url", "http://searx.test"])
            assert args.searxng_url == "http://searx.test"
