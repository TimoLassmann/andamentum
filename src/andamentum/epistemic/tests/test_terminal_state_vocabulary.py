"""Terminal-state vocabulary guard (S7, certify-before-contract G5).

``"uncertified"`` and ``"oscillation_detected"`` are DISTINCT terminals
(docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md D8):

* ``uncertified`` — no active claim reached IBE certification
  (``compute_posterior``'s no-certified-verdict gate).
* ``oscillation_detected`` — the narrower case: every active claim is
  cycle-capped, none certified, AND the counting signal is balanced.

Before D8, both cases were reported as ``oscillation_detected``, hiding
the (much more common) "nothing was certified" case behind a label that
implied balanced oscillation. No source-scan test pinned the split, so
this is that test — modeled on ``test_verdict_single_source.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import andamentum.epistemic.confidence as confidence
import andamentum.epistemic.entities.verdict as verdict

_SRC = Path(confidence.__file__).parent

#: Every file allowed to spell "oscillation_detected" (as code or as
#: documentation of it), and why each is on the list:
#: - confidence.py: the all-capped-and-balanced branch that DECIDES it
#: - entities/verdict.py: the Literal type that OWNS the vocabulary
#: - audit_report.py: the renderer's label mapping
#: - entities/claim.py: the ``cycle_capped`` field docstring explaining
#:   the two terminals a non-directional cap reading can surface as
_ALLOWED_OSCILLATION_FILES = {
    "confidence.py",
    "entities/verdict.py",
    "audit_report.py",
    "entities/claim.py",
}


def _package_sources() -> list[Path]:
    return [
        p
        for p in _SRC.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    ]


def test_oscillation_detected_confined_to_the_allow_listed_files() -> None:
    """``oscillation_detected`` appears only where it means "all active
    claims capped, none certified, counting balanced" — not as a stand-in
    for "no certified verdict" in general (S7)."""
    offenders = []
    for path in _package_sources():
        rel = str(path.relative_to(_SRC))
        if "oscillation_detected" in path.read_text() and rel not in _ALLOWED_OSCILLATION_FILES:
            offenders.append(rel)
    assert not offenders, (
        f"oscillation_detected appears outside the allow-listed files: {offenders}. "
        "It is reserved for the all-capped-and-balanced terminal; the "
        "general no-certified-verdict case is 'uncertified'."
    )


def test_uncertified_is_a_declared_terminal_state() -> None:
    """The vocabulary itself: ``uncertified`` is one of the closed set of
    terminal states, alongside (not merged with) ``oscillation_detected``."""
    raw = Path(verdict.__file__).read_text()
    assert '"uncertified"' in raw
    assert '"oscillation_detected"' in raw
    # Both must appear in the SAME Literal declaration, not as separate
    # ad hoc strings — i.e. within a few lines of each other.
    idx_u = raw.index('"uncertified"')
    idx_o = raw.index('"oscillation_detected"')
    between = raw[min(idx_u, idx_o) : max(idx_u, idx_o)]
    assert between.count("\n") < 5, (
        "'uncertified' and 'oscillation_detected' are not declared in the "
        "same Literal — the closed terminal-state vocabulary has split."
    )


def test_no_certified_verdict_gate_sets_uncertified() -> None:
    """``compute_posterior``'s no-certified-verdict early return sets
    ``terminal_state="uncertified"`` — not ``oscillation_detected`` and
    not left to whatever the counting path would have produced."""
    raw = Path(confidence.__file__).read_text()
    match = re.search(
        r'reason_prefix = "No certified verdict: "', raw
    )
    assert match, (
        "confidence.py's no-certified-verdict reason string is gone or "
        "reworded — this test needs updating alongside that change, not "
        "silently passing on a moved gate."
    )
    # The assignment `ts = "uncertified"` must appear in the same
    # if/else block as the reason prefix (within a handful of lines).
    window = raw[max(0, match.start() - 400) : match.start() + 100]
    assert 'ts = "uncertified"' in window, (
        "the no-certified-verdict branch no longer sets terminal_state to "
        "'uncertified' — S7/D8 regression."
    )


def test_all_capped_no_signal_branch_sets_oscillation_detected() -> None:
    """The narrower, later gate — every active claim capped, none
    certified, counting balanced — is the ONLY place that sets
    ``oscillation_detected``."""
    raw = Path(confidence.__file__).read_text()
    match = re.search(r"all_capped_no_signal\s*=", raw)
    assert match, "the all_capped_no_signal gate is gone or renamed."
    window = raw[match.start() : match.start() + 600]
    assert 'terminal_state = "oscillation_detected"' in window, (
        "all_capped_no_signal no longer sets terminal_state to "
        "'oscillation_detected' — S7/D8 regression."
    )
