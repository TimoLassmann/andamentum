"""Tests for the cross-provider evidence dedupe sweep."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.dedupe_evidence import (
    dedupe_evidence_by_source_ref,
    normalize_source_ref,
)
from andamentum.epistemic.entities import Evidence, Objective
from andamentum.epistemic.repository import EpistemicRepository


# ── normalize_source_ref ──────────────────────────────────────────────


class TestNormalizeSourceRef:
    def test_strips_doi_https_prefix(self):
        assert normalize_source_ref("https://doi.org/10.1234/X") == "10.1234/x"

    def test_strips_doi_dx_prefix(self):
        assert normalize_source_ref("http://dx.doi.org/10.1234/X") == "10.1234/x"

    def test_strips_pubmed_prefix(self):
        assert (
            normalize_source_ref("https://pubmed.ncbi.nlm.nih.gov/12345/")
            == "12345"
        )

    def test_lowercases(self):
        assert normalize_source_ref("10.1234/ABC") == "10.1234/abc"

    def test_strips_query_string(self):
        assert (
            normalize_source_ref("https://example.com/paper?utm=x")
            == "https://example.com/paper"
        )

    def test_strips_fragment(self):
        assert (
            normalize_source_ref("https://example.com/paper#section")
            == "https://example.com/paper"
        )

    def test_strips_trailing_slash(self):
        assert normalize_source_ref("10.1234/x/") == "10.1234/x"

    def test_empty_returns_empty(self):
        assert normalize_source_ref("") == ""
        assert normalize_source_ref(None) == ""

    def test_doi_variants_collapse_to_same_key(self):
        # All four variants of the same DOI should hash to identical keys.
        a = normalize_source_ref("10.1234/X")
        b = normalize_source_ref("10.1234/x")
        c = normalize_source_ref("https://doi.org/10.1234/X")
        d = normalize_source_ref("http://dx.doi.org/10.1234/x/")
        assert a == b == c == d


# ── dedupe_evidence_by_source_ref ─────────────────────────────────────


@pytest.fixture
async def repo(tmp_path: Path) -> EpistemicRepository:
    store = DocumentStore.for_database("test_dedupe", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def _make_objective(repo: EpistemicRepository) -> str:
    obj = Objective(description="test objective")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj.entity_id


def _make_evidence(
    obj_id: str,
    source_type: str,
    source_ref: str,
    *,
    content: str = "Some extracted content",
    extracted: bool = True,
    invalidated: bool = False,
) -> Evidence:
    return Evidence(
        objective_id=obj_id,
        source_type=source_type,
        source_ref=source_ref,
        extracted=extracted,
        extracted_content=content if extracted else "",
        invalidated=invalidated,
    )


class TestDedupeEvidenceBySourceRef:
    async def test_same_doi_three_providers_dedupes_to_one(self, repo):
        obj_id = await _make_objective(repo)
        # Three providers return the same paper.
        e_pubmed = _make_evidence(obj_id, "pubmed", "10.1234/abc")
        e_openalex = _make_evidence(obj_id, "openalex", "https://doi.org/10.1234/abc")
        e_europepmc = _make_evidence(
            obj_id, "europepmc", "10.1234/ABC", content="Slightly different summary"
        )
        await repo.save(e_pubmed)
        await repo.save(e_openalex)
        await repo.save(e_europepmc)

        n_groups, n_marked = await dedupe_evidence_by_source_ref(repo, obj_id)
        assert n_groups == 1
        assert n_marked == 2

        # Exactly one of the three remains non-invalidated.
        all_evidence = await repo.query("evidence", objective_id=obj_id)
        non_inv = [e for e in all_evidence if not e.invalidated]
        assert len(non_inv) == 1

    async def test_winner_has_longest_content(self, repo):
        obj_id = await _make_objective(repo)
        short = _make_evidence(obj_id, "pubmed", "10.1/x", content="short")
        long_ = _make_evidence(
            obj_id, "openalex", "10.1/x", content="much longer extracted summary"
        )
        await repo.save(short)
        await repo.save(long_)

        await dedupe_evidence_by_source_ref(repo, obj_id)

        kept = await repo.get("evidence", long_.entity_id)
        dropped = await repo.get("evidence", short.entity_id)
        assert kept.invalidated is False
        assert dropped.invalidated is True
        assert "duplicate" in (dropped.invalidation_reason or "").lower()

    async def test_invalidation_reason_lists_other_providers(self, repo):
        obj_id = await _make_objective(repo)
        e1 = _make_evidence(obj_id, "pubmed", "10.1/x", content="A")
        e2 = _make_evidence(obj_id, "openalex", "10.1/x", content="BB")
        e3 = _make_evidence(obj_id, "europepmc", "10.1/x", content="CCC")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        await dedupe_evidence_by_source_ref(repo, obj_id)

        # e3 has the longest content → kept; e1 and e2 → marked
        marked = [e for e in [await repo.get("evidence", e.entity_id) for e in [e1, e2]] if e.invalidated]
        assert len(marked) == 2
        # The losers' invalidation_reason should mention at least one other provider
        for m in marked:
            assert "pubmed" in m.invalidation_reason or "openalex" in m.invalidation_reason

    async def test_empty_source_refs_dont_dedupe_each_other(self, repo):
        obj_id = await _make_objective(repo)
        e1 = _make_evidence(obj_id, "web_search", "")
        e2 = _make_evidence(obj_id, "web_search", "")
        await repo.save(e1)
        await repo.save(e2)

        n_groups, n_marked = await dedupe_evidence_by_source_ref(repo, obj_id)
        assert n_groups == 0
        assert n_marked == 0

    async def test_already_invalidated_evidence_skipped(self, repo):
        obj_id = await _make_objective(repo)
        # e1 is already invalidated for some other reason.
        e1 = _make_evidence(obj_id, "pubmed", "10.1/x", invalidated=True)
        e2 = _make_evidence(obj_id, "openalex", "10.1/x")
        await repo.save(e1)
        await repo.save(e2)

        n_groups, n_marked = await dedupe_evidence_by_source_ref(repo, obj_id)
        # No duplicates among non-invalidated items (only e2 participates).
        assert n_groups == 0
        assert n_marked == 0

        # e1 stays invalidated, e2 stays valid.
        e1_after = await repo.get("evidence", e1.entity_id)
        e2_after = await repo.get("evidence", e2.entity_id)
        assert e1_after.invalidated is True
        assert e2_after.invalidated is False

    async def test_distinct_papers_not_deduped(self, repo):
        obj_id = await _make_objective(repo)
        e1 = _make_evidence(obj_id, "pubmed", "10.1/a")
        e2 = _make_evidence(obj_id, "pubmed", "10.1/b")
        e3 = _make_evidence(obj_id, "openalex", "10.1/c")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        n_groups, n_marked = await dedupe_evidence_by_source_ref(repo, obj_id)
        assert n_groups == 0
        assert n_marked == 0

    async def test_unextracted_evidence_skipped(self, repo):
        obj_id = await _make_objective(repo)
        # An extracted dupe and an unextracted stub with the same ref.
        # Only extracted items should participate in dedupe.
        e1 = _make_evidence(obj_id, "pubmed", "10.1/x", extracted=True)
        stub = _make_evidence(obj_id, "openalex", "10.1/x", extracted=False)
        await repo.save(e1)
        await repo.save(stub)

        n_groups, n_marked = await dedupe_evidence_by_source_ref(repo, obj_id)
        # Stub is ignored (extracted=False), so e1 has no peer to merge with.
        assert n_groups == 0
        assert n_marked == 0


class TestVouchingPropagatesAsCorroborationCount:
    """Phase B: when N independent providers return the same item, the
    surviving Evidence's ``corroboration_count`` reflects N. Downstream
    consumers (compute_posterior, IBE chain) already weight evidence by
    ``1 + log(corroboration_count)``, so populating it from provider
    count makes Reichenbach common-cause vouching propagate
    automatically.
    """

    async def test_three_providers_same_doi_yields_count_three(self, repo):
        obj_id = await _make_objective(repo)
        e_pubmed = _make_evidence(obj_id, "pubmed", "10.1234/abc")
        e_openalex = _make_evidence(
            obj_id, "openalex", "https://doi.org/10.1234/abc"
        )
        e_europepmc = _make_evidence(obj_id, "europepmc", "10.1234/ABC")
        await repo.save(e_pubmed)
        await repo.save(e_openalex)
        await repo.save(e_europepmc)

        await dedupe_evidence_by_source_ref(repo, obj_id)

        # Whichever item survived (longest content / oldest tie-break)
        # should carry corroboration_count=3 and a non-empty
        # corroborating_sources list.
        all_ev = await repo.query("evidence", objective_id=obj_id)
        survivors = [e for e in all_ev if not e.invalidated]
        assert len(survivors) == 1
        winner = survivors[0]
        assert winner.corroboration_count == 3
        assert len(winner.corroborating_sources) >= 1

    async def test_two_providers_yields_count_two(self, repo):
        obj_id = await _make_objective(repo)
        e1 = _make_evidence(obj_id, "pubmed", "10.5/y")
        e2 = _make_evidence(obj_id, "openalex", "10.5/y")
        await repo.save(e1)
        await repo.save(e2)

        await dedupe_evidence_by_source_ref(repo, obj_id)

        all_ev = await repo.query("evidence", objective_id=obj_id)
        winner = next(e for e in all_ev if not e.invalidated)
        assert winner.corroboration_count == 2

    async def test_single_provider_unchanged(self, repo):
        obj_id = await _make_objective(repo)
        e = _make_evidence(obj_id, "pubmed", "10.6/z")
        await repo.save(e)

        await dedupe_evidence_by_source_ref(repo, obj_id)

        loaded = await repo.get("evidence", e.entity_id)
        # No duplicates seen → corroboration_count stays at the default
        # (1). Phase B only INCREASES the count from provider vouching;
        # it never decreases below an existing value (other paths like
        # the HDBSCAN cluster sweep can also raise it).
        assert loaded.corroboration_count == 1
