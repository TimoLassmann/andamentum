"""Integration tests for orphan-evidence linking in ExtractNewEvidence.

When a gatherer returns multiple results for one query, ExtractEvidenceOperation
creates the original stub plus extra Evidence entities.  The extras lack a
claim link and a support_judgment.  The ExtractNewEvidence graph node is
responsible for linking them to the originating claim and judging them.

These tests exercise the linking/judging logic introduced to fix that gap.
"""

from ..entities import Claim, ClaimStage, Evidence, Objective
from ..operations import GatheredEvidence


class FakeMultiGatherer:
    """Returns three GatheredEvidence items so two extras are created."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        self.calls.append((source_type, query))
        return [
            GatheredEvidence(
                content="Source A content",
                source_ref="https://a.example.com",
                source_type="web_search",
                quality_score=0.7,
            ),
            GatheredEvidence(
                content="Source B content",
                source_ref="https://b.example.com",
                source_type="web_search",
                quality_score=0.6,
            ),
            GatheredEvidence(
                content="Source C content",
                source_ref="https://c.example.com",
                source_type="web_search",
                quality_score=0.8,
            ),
        ]


class TestExtractNewEvidenceLinking:
    """ExtractNewEvidence links orphan extras to the originating claim."""

    async def test_extra_evidence_linked_to_claim(self, repo, fake_runner):
        """After extraction, all 3 Evidence entities appear in claim.evidence_ids."""
        obj = Objective(
            entity_id="obj-link",
            objective_id="obj-link",
            description="Does spaced repetition improve retention?",
            phase="planned",
        )
        await repo.save(obj)

        claim = Claim(
            entity_id="c-link",
            objective_id="obj-link",
            statement="Spaced repetition improves retention.",
            scope="Educational settings",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["ev-link"],  # the stub is pre-linked
        )
        await repo.save(claim)

        stub = Evidence(
            entity_id="ev-link",
            objective_id="obj-link",
            source_type="web_search",
            source_ref="spaced repetition effectiveness",
            extracted=False,
            depends_on_claim_id="c-link",
        )
        await repo.save(stub)

        # Build graph deps with multi-gatherer
        from ..graph.deps import EpistemicDeps
        from ..graph.state import EpistemicGraphState
        from ..graph.nodes import ExtractNewEvidence

        gatherer = FakeMultiGatherer()
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=fake_runner,
            evidence_gatherer=gatherer,
        )
        state = EpistemicGraphState(objective_id="obj-link")

        # Run the node via its run() method with a minimal ctx stub
        node = ExtractNewEvidence()

        class FakeCtx:
            pass

        ctx = FakeCtx()
        ctx.state = state
        ctx.deps = deps

        await node.run(ctx)  # type: ignore[arg-type]

        # All 3 evidence entities must now be in claim.evidence_ids
        updated_claim = await repo.get("claim", "c-link")
        assert isinstance(updated_claim, Claim)
        assert len(updated_claim.evidence_ids) == 3, (
            f"Expected 3 evidence_ids, got {updated_claim.evidence_ids}"
        )
        assert updated_claim.evidence_count == 3

    async def test_extra_evidence_judged(self, repo, fake_runner):
        """After extraction, all extra Evidence entities have support_judgment set."""
        obj = Objective(
            entity_id="obj-judge",
            objective_id="obj-judge",
            description="Does spaced repetition improve retention?",
            phase="planned",
        )
        await repo.save(obj)

        claim = Claim(
            entity_id="c-judge",
            objective_id="obj-judge",
            statement="Spaced repetition improves retention.",
            scope="Educational settings",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["ev-judge"],
        )
        await repo.save(claim)

        stub = Evidence(
            entity_id="ev-judge",
            objective_id="obj-judge",
            source_type="web_search",
            source_ref="spaced repetition effectiveness",
            extracted=False,
            depends_on_claim_id="c-judge",
        )
        await repo.save(stub)

        from ..graph.deps import EpistemicDeps
        from ..graph.state import EpistemicGraphState
        from ..graph.nodes import ExtractNewEvidence

        gatherer = FakeMultiGatherer()
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=fake_runner,
            evidence_gatherer=gatherer,
        )
        state = EpistemicGraphState(objective_id="obj-judge")

        node = ExtractNewEvidence()

        class FakeCtx:
            pass

        ctx = FakeCtx()
        ctx.state = state
        ctx.deps = deps

        await node.run(ctx)  # type: ignore[arg-type]

        # All extracted evidence entities should have support_judgment set
        all_evidence = await repo.query("evidence", objective_id="obj-judge", extracted=True)
        assert len(all_evidence) == 3
        for ev in all_evidence:
            assert ev.support_judgment is not None, (
                f"Evidence {ev.entity_id} ({ev.source_ref}) missing support_judgment"
            )

    async def test_no_extras_when_single_result(self, repo, fake_runner):
        """When gatherer returns exactly one item, no extra linking is attempted."""
        obj = Objective(
            entity_id="obj-single",
            objective_id="obj-single",
            description="Does spaced repetition improve retention?",
            phase="planned",
        )
        await repo.save(obj)

        claim = Claim(
            entity_id="c-single",
            objective_id="obj-single",
            statement="Spaced repetition improves retention.",
            scope="General",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["ev-single"],
        )
        await repo.save(claim)

        stub = Evidence(
            entity_id="ev-single",
            objective_id="obj-single",
            source_type="web_search",
            source_ref="spaced repetition",
            extracted=False,
            depends_on_claim_id="c-single",
        )
        await repo.save(stub)

        from ..operations import GatheredEvidence

        class SingleGatherer:
            async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
                return [
                    GatheredEvidence(
                        content="Only one result",
                        source_ref="https://only.example.com",
                        source_type="web_search",
                        quality_score=0.7,
                    )
                ]

        from ..graph.deps import EpistemicDeps
        from ..graph.state import EpistemicGraphState
        from ..graph.nodes import ExtractNewEvidence

        deps = EpistemicDeps(
            repo=repo,
            agent_runner=fake_runner,
            evidence_gatherer=SingleGatherer(),
        )
        state = EpistemicGraphState(objective_id="obj-single")
        node = ExtractNewEvidence()

        class FakeCtx:
            pass

        ctx = FakeCtx()
        ctx.state = state
        ctx.deps = deps

        await node.run(ctx)  # type: ignore[arg-type]

        updated_claim = await repo.get("claim", "c-single")
        # Stub was pre-linked; no extras, so evidence_ids stays length 1
        assert len(updated_claim.evidence_ids) == 1

    async def test_no_claim_link_skipped_gracefully(self, repo, fake_runner):
        """Orphan extras whose original stub has no depends_on_claim_id are skipped."""
        obj = Objective(
            entity_id="obj-orphan",
            objective_id="obj-orphan",
            description="Test question",
            phase="planned",
        )
        await repo.save(obj)

        # Stub with NO depends_on_claim_id — simulates plan-phase evidence
        stub = Evidence(
            entity_id="ev-orphan",
            objective_id="obj-orphan",
            source_type="web_search",
            source_ref="some query",
            extracted=False,
            depends_on_claim_id=None,
        )
        await repo.save(stub)

        from ..graph.deps import EpistemicDeps
        from ..graph.state import EpistemicGraphState
        from ..graph.nodes import ExtractNewEvidence

        gatherer = FakeMultiGatherer()
        deps = EpistemicDeps(
            repo=repo,
            agent_runner=fake_runner,
            evidence_gatherer=gatherer,
        )
        state = EpistemicGraphState(objective_id="obj-orphan")
        node = ExtractNewEvidence()

        class FakeCtx:
            pass

        ctx = FakeCtx()
        ctx.state = state
        ctx.deps = deps

        # Should complete without error even though no claim exists
        await node.run(ctx)  # type: ignore[arg-type]

        # All 3 entities created, none linked (no claim to link to)
        all_evidence = await repo.query("evidence", objective_id="obj-orphan", extracted=True)
        assert len(all_evidence) == 3
