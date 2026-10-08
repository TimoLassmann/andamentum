"""Field-ownership guard + Tier-1 contract tests for Reichenbach corroboration.

Principle-fidelity plan §4.4 (docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md):
``corroboration_count`` must have exactly one writer — the Reichenbach
computation in ``operations.claims.select_top_k_evidence`` — computed from
``independence_group`` membership, never from raw HDBSCAN cluster size
(that redundancy signal now lives in the honestly-named
``retrieval_cluster_size``).

``test_corroboration_count_has_exactly_one_writer`` below is the guard: a
source-level AST scan in the style already established by
``test_verdict_single_source.py`` / ``test_doubt_scale_single_source.py``
("these are source-level scans, so they catch the accumulation itself
rather than one instance of its symptom"). It is what would have caught
``dedupe_evidence.py``'s now-removed second writer the day it was added.

The rest of this file is Tier-1 (plan §4.2b): pure/deterministic contract
tests for the new independence-based computation. No model, no fakes —
``deduplicate_evidence`` (the one genuinely model-backed step, via
embeddings) is stubbed with hand-built ``EvidenceCluster`` objects, exactly
the pattern ``test_no_silent_fallbacks.py`` already uses to isolate this
same function's OTHER contract (every cluster gets a representative). That
is not "faking the model" (plan §4.2b's rule under test) — it is holding a
separately-tested, already-deterministic clustering primitive fixed so the
NEW logic built on top of it can be checked on its own.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from andamentum.document_store import Store
from andamentum.epistemic.dedup import EvidenceCluster
from andamentum.epistemic.entities import Evidence
from andamentum.epistemic.operations.claims import (
    _independence_group,
    select_top_k_evidence,
)
from andamentum.epistemic.repository import EpistemicRepository

_SRC = Path(__file__).resolve().parent.parent  # .../src/andamentum/epistemic


# ── Field-ownership guard ───────────────────────────────────────────────

#: The one module allowed to write corroboration_count. Relative to _SRC,
#: matching how offenders are reported below.
_OWNER = "operations/claims.py"


def _package_sources() -> list[Path]:
    """Every shipped module in ``andamentum.epistemic`` (tests excluded)."""
    return [
        p
        for p in _SRC.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    ]


def _attribute_write_sites(field_name: str) -> dict[str, list[int]]:
    """Files where ``<expr>.<field_name> = ...`` or ``+=`` etc. appears as
    an attribute WRITE, mapped to the line numbers it occurs at.

    Deliberately narrow: only assignment TARGETS count. A constructor
    keyword (``Evidence(corroboration_count=...)``), a read
    (``getattr(ev, "corroboration_count", 1)``), or a docstring/comment
    mentioning the name are not writes and must not trip this scan —
    entities/evidence.py itself both defines the field and restores it
    from a keyword argument in ``_from_metadata``, neither of which is an
    attribute-assignment target.
    """
    hits: dict[str, list[int]] = {}
    for path in _package_sources():
        tree = ast.parse(path.read_text())
        lines: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == field_name:
                    lines.append(target.lineno)
        if lines:
            hits[str(path.relative_to(_SRC))] = sorted(lines)
    return hits


def test_corroboration_count_has_exactly_one_writer() -> None:
    """The guard plan §4.4 asks for: a field reused across a principle
    boundary becomes a CI failure at the commit that does it, not a silent
    multi-year drift (the exact history of ``dedupe_evidence.py``'s now-
    removed "Phase B" write, discovered while implementing this fix)."""
    hits = _attribute_write_sites("corroboration_count")
    offenders = {f: lines for f, lines in hits.items() if f != _OWNER}
    assert not offenders, (
        f"corroboration_count is written outside {_OWNER}: {offenders}. "
        "Reichenbach corroboration must have exactly one writer (plan "
        "PRINCIPLE_FIDELITY_PLAN.md §4.4). If this caller needs a "
        "redundancy/volume signal instead, give it an honestly-named new "
        "field (see retrieval_cluster_size) rather than reusing this one."
    )
    assert hits.get(_OWNER), (
        f"{_OWNER} does not write corroboration_count at all — either the "
        "owning module moved (update _OWNER above to match) or the write "
        "was accidentally deleted."
    )


def test_retrieval_cluster_size_has_exactly_one_writer() -> None:
    """Same guard for the field that replaced the old redundancy-reusing
    corroboration_count write — it must not sprout a second writer either."""
    hits = _attribute_write_sites("retrieval_cluster_size")
    offenders = {f: lines for f, lines in hits.items() if f != _OWNER}
    assert not offenders, (
        f"retrieval_cluster_size is written outside {_OWNER}: {offenders}."
    )
    assert hits.get(_OWNER)


# ── Tier-1: _independence_group (pure, no model, no I/O) ───────────────


class TestIndependenceGroup:
    def test_registered_provider_uses_its_independence_group(self) -> None:
        ev = Evidence(objective_id="obj-1", source_type="pubmed")
        assert _independence_group(ev) == "biomedical_literature"

    def test_biorxiv_and_arxiv_share_one_group(self) -> None:
        """The FINDINGS.md §4.5 worked example: two hits from bioRxiv and
        arXiv are one independence group (preprint_archive), not two."""
        a = Evidence(objective_id="obj-1", source_type="biorxiv")
        b = Evidence(objective_id="obj-1", source_type="arxiv")
        assert _independence_group(a) == _independence_group(b) == "preprint_archive"

    def test_unregistered_source_type_falls_back_to_itself(self) -> None:
        """An offline/benchmark corpus provider (e.g. the principle-
        fidelity canary's ``scifact_corpus``) is not in PROVIDER_REGISTRY.
        It must be treated as its own singleton group, never silently
        merged into an unrelated registered one."""
        ev = Evidence(objective_id="obj-1", source_type="scifact_corpus")
        assert _independence_group(ev) == "scifact_corpus"

    def test_default_source_type_is_its_own_group(self) -> None:
        ev = Evidence(objective_id="obj-1")  # source_type defaults "unknown"
        assert _independence_group(ev) == "unknown"


# ── Tier-1: select_top_k_evidence's corroboration computation ──────────
# deduplicate_evidence is stubbed (see module docstring) so these tests
# exercise only the NEW logic, deterministically and fast.


@pytest.fixture
async def repo(tmp_path: Path) -> EpistemicRepository:
    store = await Store.open(tmp_path / "test_field_ownership.db")
    return EpistemicRepository(store)


def _ev(
    *,
    source_type: str,
    content: str,
    support_judgment: str | None = "supports",
    quality_score: float = 0.5,
) -> Evidence:
    return Evidence(
        objective_id="obj-1",
        source_type=source_type,
        source_ref=f"ref-{content}",
        extracted=True,
        extracted_content=content,
        support_judgment=support_judgment,
        quality_score=quality_score,
    )


def _stub_dedup(clusters: list[EvidenceCluster]):
    async def _inner(texts, min_cluster_size=2, *, embedding_model):
        return clusters

    return _inner


class TestCorroborationCountIsIndependenceNotRedundancy:
    async def test_same_provider_near_duplicates_do_not_corroborate(self, repo) -> None:
        """Two near-duplicate items from the SAME provider, HDBSCAN-merged
        into one cluster (the old bug's exact shape: claims.py:226 used to
        set corroboration_count=2 here). Post-fix: one independence group
        contributing → no genuine corroboration."""
        items = [
            _ev(source_type="pubmed", content="a"),
            _ev(source_type="pubmed", content="b"),
        ]
        cluster = EvidenceCluster(
            medoid_index=0, representative_indices=[0], member_indices=[0, 1], count=2
        )
        with patch(
            "andamentum.epistemic.operations.claims.deduplicate_evidence",
            new=_stub_dedup([cluster]),
        ):
            reps, _total, _deferred = await select_top_k_evidence(
                repo, items, embedding_model="stub-model"
            )

        assert len(reps) == 1
        rep = reps[0]
        assert rep.corroboration_count == 1, (
            "same independence_group, same content cluster: redundancy, "
            "not independent corroboration"
        )
        assert rep.retrieval_cluster_size == 2, (
            "the raw retrieval/redundancy count must still be recorded, "
            "just under its own honest field"
        )

    async def test_biorxiv_and_arxiv_near_duplicates_do_not_corroborate(
        self, repo
    ) -> None:
        """Same as above but across TWO providers that share one
        independence_group — the direct FINDINGS.md §4.5 case."""
        items = [
            _ev(source_type="biorxiv", content="a"),
            _ev(source_type="arxiv", content="b"),
        ]
        cluster = EvidenceCluster(
            medoid_index=0, representative_indices=[0], member_indices=[0, 1], count=2
        )
        with patch(
            "andamentum.epistemic.operations.claims.deduplicate_evidence",
            new=_stub_dedup([cluster]),
        ):
            reps, _total, _deferred = await select_top_k_evidence(
                repo, items, embedding_model="stub-model"
            )

        assert reps[0].corroboration_count == 1

    async def test_different_independence_groups_in_one_cluster_do_corroborate(
        self, repo
    ) -> None:
        """Two providers from DIFFERENT independence groups whose hits
        happened to content-cluster together: genuine independent
        corroboration, even though it is one HDBSCAN cluster."""
        items = [
            _ev(source_type="pubmed", content="a"),
            _ev(source_type="clinicaltrials", content="b"),
        ]
        cluster = EvidenceCluster(
            medoid_index=0, representative_indices=[0], member_indices=[0, 1], count=2
        )
        with patch(
            "andamentum.epistemic.operations.claims.deduplicate_evidence",
            new=_stub_dedup([cluster]),
        ):
            reps, _total, _deferred = await select_top_k_evidence(
                repo, items, embedding_model="stub-model"
            )

        assert reps[0].corroboration_count == 2

    async def test_same_provider_different_clusters_do_corroborate(self, repo) -> None:
        """Two items from the SAME provider that HDBSCAN did NOT merge
        (genuinely different content, e.g. two distinct study designs) —
        different content clusters make them independent even though the
        provider (and its independence_group) is identical. This is the
        mechanism that lets independence be detected when only one
        provider is in play at all (e.g. an offline single-provider
        benchmark corpus): distinct clusters are still distinct sources."""
        items = [
            _ev(source_type="pubmed", content="a"),
            _ev(source_type="pubmed", content="b"),
        ]
        clusters = [
            EvidenceCluster(
                medoid_index=0, representative_indices=[0], member_indices=[0], count=1
            ),
            EvidenceCluster(
                medoid_index=1, representative_indices=[1], member_indices=[1], count=1
            ),
        ]
        with patch(
            "andamentum.epistemic.operations.claims.deduplicate_evidence",
            new=_stub_dedup(clusters),
        ):
            reps, _total, _deferred = await select_top_k_evidence(
                repo, items, embedding_model="stub-model"
            )

        assert len(reps) == 2
        for rep in reps:
            assert rep.corroboration_count == 2
            assert rep.retrieval_cluster_size == 1

    async def test_contradicting_cluster_does_not_inflate_supporting_count(
        self, repo
    ) -> None:
        """A cluster that CONTRADICTS the claim must never count toward a
        SUPPORTING representative's corroboration — agreement, not mere
        existence, is what Reichenbach corroboration means."""
        items = [
            _ev(source_type="pubmed", content="a", support_judgment="supports"),
            _ev(
                source_type="clinicaltrials",
                content="b",
                support_judgment="contradicts",
            ),
        ]
        clusters = [
            EvidenceCluster(
                medoid_index=0, representative_indices=[0], member_indices=[0], count=1
            ),
            EvidenceCluster(
                medoid_index=1, representative_indices=[1], member_indices=[1], count=1
            ),
        ]
        with patch(
            "andamentum.epistemic.operations.claims.deduplicate_evidence",
            new=_stub_dedup(clusters),
        ):
            reps, _total, _deferred = await select_top_k_evidence(
                repo, items, embedding_model="stub-model"
            )

        by_judgment = {r.support_judgment: r for r in reps}
        assert by_judgment["supports"].corroboration_count == 1
        assert by_judgment["contradicts"].corroboration_count == 1

    async def test_unjudged_evidence_still_gets_a_value_not_a_crash(self, repo) -> None:
        """ProposeClaimsOperation calls select_top_k_evidence BEFORE any
        evidence-claim judgment exists (support_judgment is None on every
        item at that call site). The function must not crash or leave
        corroboration_count unset — it buckets unjudged items together."""
        items = [
            _ev(source_type="pubmed", content="a", support_judgment=None),
            _ev(source_type="clinicaltrials", content="b", support_judgment=None),
        ]
        cluster = EvidenceCluster(
            medoid_index=0, representative_indices=[0], member_indices=[0, 1], count=2
        )
        with patch(
            "andamentum.epistemic.operations.claims.deduplicate_evidence",
            new=_stub_dedup([cluster]),
        ):
            reps, _total, _deferred = await select_top_k_evidence(
                repo, items, embedding_model="stub-model"
            )

        assert reps[0].corroboration_count == 2
