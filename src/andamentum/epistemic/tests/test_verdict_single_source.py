"""Anti-fragmentation guards for the verdict.

Five independent verdict derivations accumulated in this package because
nothing made adding a sixth fail. These are source-level scans, so they
catch the accumulation itself rather than one instance of its symptom.
"""

from __future__ import annotations

from pathlib import Path

import andamentum.epistemic.audit_report as audit_report
import andamentum.epistemic.report_generator as report_generator
from andamentum.epistemic.adapters import WriteAnswerResult
from andamentum.epistemic.agents.output_models import WriteAnswerOutput

_SRC = Path(audit_report.__file__).parent


def _package_sources() -> list[Path]:
    """Every shipped module in ``andamentum.epistemic`` (tests excluded)."""
    return [
        p
        for p in _SRC.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    ]


def test_only_the_verdict_module_reads_the_directional_breakpoint() -> None:
    """ONE posterior→label comparison in the whole package.

    ``POSTERIOR_DIRECTIONAL_BREAKPOINT`` may be imported only by
    ``thresholds.py`` (where it is defined) and ``entities/verdict.py``
    (which owns ``verdict_label``). Every other consumer maps the
    resulting *direction* — that is what removed the ``>`` vs ``>=``
    boundary disagreement between the combiner and the renderer.

    This is the guard that would have prevented five derivations
    accumulating; adding a sixth fails here.
    """
    offenders = []
    for path in _package_sources():
        if path.name in ("thresholds.py", "verdict.py"):
            continue
        if "POSTERIOR_DIRECTIONAL_BREAKPOINT" in path.read_text():
            offenders.append(str(path.relative_to(_SRC)))
    assert not offenders, (
        "POSTERIOR_DIRECTIONAL_BREAKPOINT is read outside "
        f"entities/verdict.py: {offenders}. Map the direction that "
        "``verdict_label`` already decided instead of comparing the "
        "posterior a second time."
    )


def test_the_writer_agent_cannot_emit_a_verdict() -> None:
    """The LLM has no channel to author a verdict. Re-adding the field
    reopens the path where the prose and the structured decision
    disagree."""
    assert "verdict" not in WriteAnswerOutput.model_fields
    assert not hasattr(WriteAnswerResult(), "verdict")


def test_report_generator_does_not_parse_the_verdict_out_of_prose() -> None:
    """The prose re-parse and the offline recompute are DELETED, not
    demoted to fallbacks — leaving either as a fallback preserves
    exactly the disagreement they caused."""
    raw = Path(report_generator.__file__).read_text()
    # Comment lines are excluded so the rationale can name the thing it
    # forbids; the scan is about executable references.
    source = "\n".join(
        ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "compute_posterior" not in source, (
        "report_generator re-computes the posterior offline. It must "
        "READ the persisted snapshot verdict; recomputing cannot thread "
        "retrieval_failed and therefore always disagrees with the live run."
    )
    # The line itself is still recognised — but only to STRIP the
    # machine-owned headline out of the prose body. Extracting its value
    # (the ``stripped[len("> **Verdict:**"):]`` slice) is the scrape, and
    # that is what must stay gone.
    assert '[len("> **Verdict:**")' not in source, (
        "report_generator scrapes the verdict back out of the artefact "
        "prose. Read the persisted snapshot verdict instead."
    )
    assert 'startswith("> **Verdict:**")' in source, (
        "the machine-owned verdict line must still be stripped from the "
        "prose body, or it renders twice in the HTML report."
    )
    # The line is WRITTEN in exactly one place: the synthesis operation.
    synthesis = (_SRC / "operations" / "synthesis.py").read_text()
    assert "> **Verdict:**" in synthesis


def test_audit_report_has_no_independent_verdict_derivation() -> None:
    """The renderer reads the object's labels instead of re-deriving
    them, so the Q&A headline and the per-claim badges in one document
    can no longer disagree."""
    assert not hasattr(audit_report, "_normalised_verdict")


def test_snapshot_carries_a_typed_verdict_not_an_untyped_blob() -> None:
    """``Snapshot.combined_verdict: dict[str, Any]`` was the untyped
    cross-module boundary CLAUDE.md forbids, and it was only ever
    written for decomposed runs."""
    from andamentum.epistemic.entities import Snapshot
    from andamentum.epistemic.entities.verdict import EpistemicVerdict

    assert "combined_verdict" not in Snapshot.model_fields
    field = Snapshot.model_fields["verdict"]
    assert EpistemicVerdict in getattr(field.annotation, "__args__", (field.annotation,))
