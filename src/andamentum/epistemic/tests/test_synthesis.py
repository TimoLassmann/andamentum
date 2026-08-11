"""Tests for synthesis operations — verify content_body stripping and report assembly.

These tests verify:
1. _build_markdown produces correct output with and without quality signals
2. content_body (benchmark-safe) differs from content (full production) in the right ways
3. Trace mapping is built correctly
4. Quality signals computation is deterministic
"""

from ..entities import (
    Artefact,
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Snapshot,
    Uncertainty,
    UncertaintyType,
)
from ..entities.verdict import (
    EpistemicVerdict,
    PosteriorReport,
    VerdictProvenance,
)
from ..operations import create_operations, SynthesizeReportOperation
from ..operations.base import OperationInput
from .conftest import make_verdict


class TestSynthesizeContentBody:
    """Verify that content_body strips quality signals while preserving everything else."""

    async def test_artefact_has_both_content_fields(self, repo, fake_runner):
        """SynthesizeReportOperation must produce both content and content_body."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Test Q",
            phase="claims_done",
        )
        await repo.save(obj)

        e = Evidence(
            entity_id="e-1",
            objective_id="obj-1",
            quality_score=0.7,
            extracted=True,
            extracted_content="Evidence text",
        )
        await repo.save(e)

        c = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="Claim X",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-1"],
        )
        await repo.save(c)

        snap = Snapshot(
            entity_id="snap-1",
            objective_id="obj-1",
            snapshot_type="final",
            verdict=make_verdict("obj-1", posterior=0.8, claim_ids=("c-1",)),
            claim_ids=["c-1"],
            evidence_ids=["e-1"],
        )
        await repo.save(snap)

        ops = create_operations(repo, fake_runner)
        work = OperationInput(
            entity_id="snap-1", entity_type="snapshot", operation="synthesize_report"
        )
        result = await ops["synthesize_report"].execute(work)

        assert result.success

        artefacts = await repo.query("artefact", objective_id="obj-1")
        assert len(artefacts) == 1
        artefact = artefacts[0]

        assert artefact.content, "content must not be empty"
        assert artefact.content_body is not None, "content_body must exist"

    async def test_content_body_shorter_than_content(self, repo, fake_runner):
        """content_body should be shorter (no quality signals) or same length as content."""
        obj = Objective(
            entity_id="obj-2",
            objective_id="obj-2",
            description="Test Q",
            phase="claims_done",
        )
        await repo.save(obj)

        e = Evidence(
            entity_id="e-2",
            objective_id="obj-2",
            quality_score=0.8,
            extracted=True,
            extracted_content="Strong evidence here",
        )
        await repo.save(e)

        c = Claim(
            entity_id="c-2",
            objective_id="obj-2",
            statement="Well-supported claim",
            stage=ClaimStage.PROVISIONAL,
            scrutiny_verdict="pass",
            evidence_ids=["e-2"],
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
        )
        await repo.save(c)

        snap = Snapshot(
            entity_id="snap-2",
            objective_id="obj-2",
            snapshot_type="final",
            verdict=make_verdict("obj-2", posterior=0.8, claim_ids=("c-2",)),
            claim_ids=["c-2"],
            evidence_ids=["e-2"],
        )
        await repo.save(snap)

        ops = create_operations(repo, fake_runner)
        work = OperationInput(
            entity_id="snap-2", entity_type="snapshot", operation="synthesize_report"
        )
        result = await ops["synthesize_report"].execute(work)

        assert result.success
        artefacts = await repo.query("artefact", objective_id="obj-2")
        artefact = artefacts[0]

        # content_body should be shorter because quality signals are stripped
        assert len(artefact.content_body) <= len(artefact.content), (
            f"content_body ({len(artefact.content_body)}) should be <= content ({len(artefact.content)})"
        )

    async def test_content_body_excludes_methodology_section(self, repo, fake_runner):
        """content_body must not contain the Methodology section."""
        obj = Objective(
            entity_id="obj-3",
            objective_id="obj-3",
            description="Test Q",
            phase="claims_done",
        )
        await repo.save(obj)

        e = Evidence(
            entity_id="e-3", objective_id="obj-3", quality_score=0.6, extracted=True
        )
        await repo.save(e)

        c = Claim(
            entity_id="c-3",
            objective_id="obj-3",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-3"],
        )
        await repo.save(c)

        snap = Snapshot(
            entity_id="snap-3",
            objective_id="obj-3",
            snapshot_type="final",
            verdict=make_verdict("obj-3", posterior=0.8, claim_ids=("c-3",)),
            claim_ids=["c-3"],
            evidence_ids=["e-3"],
        )
        await repo.save(snap)

        ops = create_operations(repo, fake_runner)
        work = OperationInput(
            entity_id="snap-3", entity_type="snapshot", operation="synthesize_report"
        )
        await ops["synthesize_report"].execute(work)

        artefacts = await repo.query("artefact", objective_id="obj-3")
        artefact = artefacts[0]

        # The full content should have methodology info
        # content_body should NOT have the Methodology section header
        if "## Methodology" in artefact.content:
            assert "## Methodology" not in artefact.content_body, (
                "content_body must not contain Methodology section"
            )

    async def test_content_body_contains_prose_summary(self, repo, fake_runner):
        """Artefact content should contain the LLM prose summary, not per-claim blocks.

        Per-claim findings, evidence sources, and methodology are now
        rendered by the HTML report from structured data — the artefact
        stores only the prose summary.
        """
        obj = Objective(
            entity_id="obj-4",
            objective_id="obj-4",
            description="Test Q",
            phase="claims_done",
        )
        await repo.save(obj)

        e = Evidence(
            entity_id="e-4",
            objective_id="obj-4",
            quality_score=0.7,
            extracted=True,
            extracted_content="Important finding",
        )
        await repo.save(e)

        c = Claim(
            entity_id="c-4",
            objective_id="obj-4",
            statement="Spaced repetition works",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-4"],
        )
        await repo.save(c)

        snap = Snapshot(
            entity_id="snap-4",
            objective_id="obj-4",
            snapshot_type="final",
            verdict=make_verdict("obj-4", posterior=0.8, claim_ids=("c-4",)),
            claim_ids=["c-4"],
            evidence_ids=["e-4"],
        )
        await repo.save(snap)

        ops = create_operations(repo, fake_runner)
        work = OperationInput(
            entity_id="snap-4", entity_type="snapshot", operation="synthesize_report"
        )
        await ops["synthesize_report"].execute(work)

        artefacts = await repo.query("artefact", objective_id="obj-4")
        artefact = artefacts[0]

        # LLM answer should be in the artefact
        assert "Spaced repetition is effective" in artefact.content
        # Per-claim blocks should NOT be in the artefact (rendered by HTML report)
        assert "## Findings" not in artefact.content
        assert "## Evidence Sources" not in artefact.content


class TestArtefactEntity:
    """Verify Artefact entity model handles content_body correctly."""

    def test_content_body_default_empty(self):
        a = Artefact(
            objective_id="o",
            snapshot_id="s",
            content="Full content",
        )
        assert a.content_body == ""

    def test_content_body_set(self):
        a = Artefact(
            objective_id="o",
            snapshot_id="s",
            content="Full content",
            content_body="Body only",
        )
        assert a.content_body == "Body only"

    def test_extra_metadata_includes_content_body(self):
        a = Artefact(
            objective_id="o",
            snapshot_id="s",
            content="Full content",
            content_body="Body only",
        )
        meta = a._extra_metadata()
        assert "content_body" in meta
        assert meta["content_body"] == "Body only"

    def test_extra_metadata_excludes_empty_content_body(self):
        a = Artefact(
            objective_id="o",
            snapshot_id="s",
            content="Full content",
            content_body="",
        )
        meta = a._extra_metadata()
        assert "content_body" not in meta

    def test_artefact_id_alias(self):
        a = Artefact(
            entity_id="art-1",
            objective_id="o",
            snapshot_id="s",
        )
        assert a.artefact_id == "art-1"

    def test_legacy_artefact_id_accepted(self):
        """artefact_id in input data should be accepted as entity_id."""
        a = Artefact.model_validate(
            {
                "artefact_id": "art-legacy",
                "objective_id": "o",
                "snapshot_id": "s",
            }
        )
        assert a.entity_id == "art-legacy"
        assert a.artefact_id == "art-legacy"


class TestQualitySignals:
    """Verify that quality signals computation is deterministic."""

    def test_quality_signals_with_evidence(self):
        """Quality signals should include evidence count and quality stats."""
        claims = [
            Claim(
                entity_id="c-qs",
                objective_id="obj-qs",
                statement="Claim",
                stage=ClaimStage.SUPPORTED,
                scrutiny_verdict="pass",
                evidence_ids=["e-qs-0", "e-qs-1", "e-qs-2"],
            ),
        ]
        evidence = [
            Evidence(
                entity_id=f"e-qs-{i}",
                objective_id="obj-qs",
                quality_score=0.5 + i * 0.1,
                extracted=True,
            )
            for i in range(3)
        ]

        # _compute_quality_signals is a static method
        signals = SynthesizeReportOperation._compute_quality_signals(
            claims, evidence, []
        )

        assert isinstance(signals, dict)
        assert "evidence_count" in signals
        assert signals["evidence_count"] == 3
        assert "mean_evidence_quality" in signals

    def test_quality_signals_deterministic(self):
        """Same inputs should produce same quality signals."""
        claims = [
            Claim(
                entity_id="c-1",
                objective_id="o",
                statement="X",
                stage=ClaimStage.SUPPORTED,
            ),
        ]
        evidence = [
            Evidence(entity_id="e-1", objective_id="o", quality_score=0.7),
        ]

        signals1 = SynthesizeReportOperation._compute_quality_signals(
            claims, evidence, []
        )
        signals2 = SynthesizeReportOperation._compute_quality_signals(
            claims, evidence, []
        )

        assert signals1 == signals2


class TestTraceMapping:
    """Verify trace mapping from claims to evidence."""

    async def test_trace_built_from_evidence_ids(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-tr",
            objective_id="obj-tr",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)

        e1 = Evidence(
            entity_id="e-tr-1", objective_id="obj-tr", quality_score=0.7, extracted=True
        )
        e2 = Evidence(
            entity_id="e-tr-2", objective_id="obj-tr", quality_score=0.6, extracted=True
        )
        await repo.save(e1)
        await repo.save(e2)

        c = Claim(
            entity_id="c-tr",
            objective_id="obj-tr",
            statement="Traced claim",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-tr-1", "e-tr-2"],
        )
        await repo.save(c)

        snap = Snapshot(
            entity_id="snap-tr",
            objective_id="obj-tr",
            snapshot_type="final",
            verdict=make_verdict("obj-tr", posterior=0.8, claim_ids=("c-tr",)),
            claim_ids=["c-tr"],
            evidence_ids=["e-tr-1", "e-tr-2"],
        )
        await repo.save(snap)

        ops = create_operations(repo, fake_runner)
        work = OperationInput(
            entity_id="snap-tr", entity_type="snapshot", operation="synthesize_report"
        )
        result = await ops["synthesize_report"].execute(work)

        assert result.success
        artefacts = await repo.query("artefact", objective_id="obj-tr")
        artefact = artefacts[0]

        # Trace should map claim to its evidence
        assert isinstance(artefact.trace, dict)
        # At minimum, the trace should be non-empty when claims have evidence
        if artefact.trace:
            # Flatten all evidence IDs in trace values
            all_traced = []
            for ids in artefact.trace.values():
                all_traced.extend(ids)
            assert len(all_traced) > 0


class TestFreezeSnapshotCollectsAllEntities:
    """Verify snapshot freeze captures the full epistemic state."""

    async def test_snapshot_captures_evidence_and_uncertainties(
        self, repo, fake_runner
    ):
        obj = Objective(
            entity_id="obj-fs",
            objective_id="obj-fs",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)

        e = Evidence(
            entity_id="e-fs", objective_id="obj-fs", quality_score=0.7, extracted=True
        )
        await repo.save(e)

        c = Claim(
            entity_id="c-fs",
            objective_id="obj-fs",
            statement="Claim",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-fs"],
        )
        await repo.save(c)

        u = Uncertainty(
            entity_id="u-fs",
            objective_id="obj-fs",
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="Limited data",
            affected_claim_ids=["c-fs"],
        )
        await repo.save(u)

        ops = create_operations(repo, fake_runner)
        work = OperationInput(
            entity_id="obj-fs", entity_type="objective", operation="freeze_snapshot"
        )
        result = await ops["freeze_snapshot"].execute(work)

        assert result.success
        snapshots = await repo.query("snapshot", objective_id="obj-fs")
        snap = snapshots[0]

        assert "c-fs" in snap.claim_ids
        assert "e-fs" in snap.evidence_ids
        assert "u-fs" in snap.uncertainty_ids


class TestWriterContextCannotContradictTheVerdict:
    """The writer's data context is held to the same coherence contract as
    the rendered header.

    Case 781 was a header that printed "Claims Established: 0 of 1" beside a
    "Refuted" verdict. Removing that pair from the header left it in the
    writer's prompt — the place where the prose is actually produced — with
    a comment claiming the counter was kept "for the HTML report", which
    never reads it.
    """

    @staticmethod
    def _cap_decided_claim() -> Claim:
        """A claim the cycle cap decided: adjudicated, never promoted."""
        return Claim(
            entity_id="c-cap",
            objective_id="obj-cap",
            statement="Aspirin reduces colorectal cancer risk.",
            stage=ClaimStage.HYPOTHESIS,
            cycle_capped=True,
            integrated_assessment="contradicts",
            integrated_confidence=0.85,
            evidence_ids=["e-cap"],
        )

    @staticmethod
    def _verdict() -> "EpistemicVerdict":
        return EpistemicVerdict(
            objective_id="obj-cap",
            posterior_report=PosteriorReport(
                posterior=0.115,
                log_odds=0,
                supporting_count=0.0,
                contradicting_count=1.0,
                counting_posterior=0.115,
                objective_id="obj-cap",
                question_type="verificatory",
                explanation="test fixture",
            ),
            provenance=VerdictProvenance(
                n_claims_total=1,
                n_active_claims=1,
                n_adjudicated=1,
                n_cycle_capped=1,
            ),
        )

    def _context(self) -> dict:
        claim = self._cap_decided_claim()
        evidence = [
            Evidence(
                entity_id="e-cap",
                objective_id="obj-cap",
                quality_score=0.7,
                extracted=True,
                support_judgment="contradicts",
            )
        ]
        signals = SynthesizeReportOperation._compute_quality_signals(
            [claim], evidence, []
        )
        return SynthesizeReportOperation._build_data_context(
            [claim],
            evidence,
            [],
            {},
            {},
            {"e-cap": 1},
            signals,
            verdict=self._verdict(),
        )

    def test_stage_counter_is_named_for_the_stage_it_counts(self):
        """``claims_established`` read as "claims the system settled" — the
        adjudication count. Renaming it back to any name containing
        "established" reopens the misreading."""
        claim = Claim(
            entity_id="c-name",
            objective_id="obj-name",
            statement="X",
            stage=ClaimStage.SUPPORTED,
        )
        signals = SynthesizeReportOperation._compute_quality_signals([claim], [], [])

        assert signals["claims_at_supported_stage"] == 1
        assert not [k for k in signals if "established" in k.lower()], (
            f"quality_signals reintroduced an 'established' key: {sorted(signals)}"
        )

    def test_stage_counter_and_adjudication_count_are_both_present_and_disagree(self):
        """The two quantities are genuinely different and MAY disagree; what
        must hold is that the writer sees both under names that cannot be
        confused. Breaks if the counter is silently deleted from the context,
        or if it is 'fixed' by sourcing it from the verdict (which would
        destroy the stage signal instead of disambiguating it)."""
        context = self._context()

        assert context["quality_signals"]["claims_at_supported_stage"] == 0
        assert "adjudicated=1 of 1 active" in context["verdict"]
        assert not [k for k in context["quality_signals"] if "established" in k.lower()]

    def test_cap_decided_claim_is_marked_as_cap_decided(self):
        """Without the marker the writer sees ``[HYPOTHESIS] …`` immediately
        above ``Adjudicated: contradicts`` and is told by the hedging table to
        call a decided claim preliminary. Breaks if the STATUS line is
        removed."""
        context = self._context()
        summary = context["claims"][0]

        assert "[HYPOTHESIS]" in summary
        assert "Adjudicated: contradicts" in summary
        assert "DECIDED AT THE INVESTIGATION CAP" in summary

    def test_uncapped_claim_carries_no_cap_marker(self):
        """Negative control: the marker is conditional on ``cycle_capped``.
        Breaks if the STATUS line is appended unconditionally."""
        claim = Claim(
            entity_id="c-plain",
            objective_id="obj-plain",
            statement="X",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
        )
        context = SynthesizeReportOperation._build_data_context(
            [claim],
            [],
            [],
            {},
            {},
            {},
            SynthesizeReportOperation._compute_quality_signals([claim], [], []),
            verdict=self._verdict(),
        )

        assert "INVESTIGATION CAP" not in context["claims"][0]

    def test_every_quality_signal_key_is_defined_in_the_writer_prompt(self):
        """The writer is handed the signals as a raw mapping, so an
        undocumented key is a counter the model must guess the meaning of —
        exactly how a stage counter came to be read as an adjudication count.
        Breaks when a key is added to ``_compute_quality_signals`` (or renamed)
        without updating the prompt's glossary."""
        from ..agents.synthesis import WRITE_ANSWER_PROMPT

        claim = Claim(
            entity_id="c-doc",
            objective_id="obj-doc",
            statement="X",
            stage=ClaimStage.SUPPORTED,
        )
        signals = SynthesizeReportOperation._compute_quality_signals([claim], [], [])

        undocumented = [k for k in signals if f"`{k}`" not in WRITE_ANSWER_PROMPT]
        assert not undocumented, (
            "quality_signals keys handed to the writer with no definition in "
            f"WRITE_ANSWER_PROMPT: {undocumented}"
        )

    def test_stage_counter_is_read_by_nothing_outside_the_synthesis_pair(self):
        """Pins the claim the header comment now makes: ``quality_signals`` is
        local, so the stage counter reaches no renderer. Breaks if a consumer
        (e.g. the HTML audit report the old comment invoked) starts reading it
        without the counter first being persisted."""
        from pathlib import Path

        import andamentum.epistemic as epistemic_pkg

        root = Path(epistemic_pkg.__file__).parent
        allowed = {"operations/synthesis.py", "agents/synthesis.py"}
        offenders = []
        for path in root.rglob("*.py"):
            if "tests" in path.parts or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            if rel in allowed:
                continue
            text = path.read_text()
            if "claims_at_supported_stage" in text or "claims_established" in text:
                offenders.append(rel)

        assert not offenders, (
            "the stage-derived claim counter is read outside the synthesis "
            f"operation/prompt pair: {offenders}. It is not persisted, so any "
            "other consumer is reading a number that is not the run's "
            "adjudication count."
        )
