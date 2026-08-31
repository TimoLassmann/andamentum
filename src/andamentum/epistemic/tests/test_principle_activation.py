"""Tier-2 principle-activation canary — PRINCIPLE_FIDELITY_PLAN.md §4.3, §5 (Stage 0).

The permanent, real-model regression test for the principle-fidelity plan. It
invokes the canary runner (``paper/scifact/scripts/run_canary.py``) as a
subprocess over the fixture corpus
(``src/andamentum/epistemic/tests/fixtures/canary/corpus.jsonl`` — six
abstracts in the exact SciFact corpus schema, where doc 9003 genuinely
contradicts docs 9001/9002) with the real model, then asserts the six
principle-activation counts against ``MIN_ACTIVATIONS``.

No fakes anywhere in this path (plan §4.2b): no FakeAgentRunner, no scripted
agent responses, no stub embeddings. Determinism comes from the DATA — the
fixture corpus holds a real contradiction — and the assertion is a RATE across
runs, never a single outcome, because labels flip on 10–27% of identical runs.

Marker: ``ollama`` — the run resolves its LLM through the local Ollama endpoint
(``ollama:gemma4:31b-cloud``) and its embeddings through
``embeddinggemma:latest`` on the same daemon, so the test is deselected by the
default ``uv run pytest`` and selected explicitly with ``-m ollama``. Runs are
STRICTLY SEQUENTIAL inside the runner (never two model calls at once).

Ratchet semantics of ``MIN_ACTIVATIONS`` — read before touching it:

* Thresholds count the minimum number of RUNS (of those executed) in which the
  principle's activation signature fired.
* Every entry is initialised to the Stage-0 reality of the CURRENT,
  unmodified system, so this test is GREEN at Stage 0 by construction and goes
  RED only when a later stage claims a fix it did not deliver.
* "0" below does NOT mean "firing is undesirable" — it means one of:
  (a) the audit measured that the principle never fires today, or
  (b) the audit measured a nonzero baseline on dev30 but no baseline has yet
      been measured on THIS fixture, so the only guaranteed-green floor is 0.
  The Stage-0 5-run baseline block in ``docs/epistemic/FIDELITY_LEDGER.md`` is
  that measurement; once recorded, each entry ratchets to its observed floor
  (never below it). A guard this file has not armed yet is a TODO in the
  ledger, not a licence to regress.
* Each later stage ratchets ITS OWN principle's entry as part of the change
  that claims the fix (Stage 2b: agm; Stage 1: popper and a REDEFINED
  reichenbach signature; Stage 3: ibe_real_candidates) — see the per-entry
  comments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

#: Real-model test: deselected by the default ``uv run pytest`` (addopts
#: excludes the gated markers), run explicitly with ``-m ollama``.
pytestmark = pytest.mark.ollama

#: src/andamentum/epistemic/tests/test_principle_activation.py → repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "paper" / "scifact" / "scripts" / "run_canary.py"
#: Under paper/scifact/results/ — gitignored benchmark scratch, per the
#: project rule that run outputs never live in the working tree.
TEST_OUT_DIR = (
    REPO_ROOT / "paper" / "scifact" / "results" / "canary" / "test_activation"
)

#: The user's chosen canary model (plan §4.2b, 2026-08-31): the cloud-served
#: variant, because the local 31b-nvfp4 on a Mac is too slow for iteration.
MODEL = "ollama:gemma4:31b-cloud"

#: Small on purpose: this is the pre-merge gate, not the 5-run ledger baseline
#: (which the runner's default --runs 5 produces).
RUNS = 2

#: Two sequential corpus-mode runs cost minutes, not hours (the live-provider
#: exp1 runs measured median 141s/case; corpus mode is strictly cheaper), but
#: the cap must never silently cut a run short.
TIMEOUT_SECONDS = 3600

#: All six activation signatures, in the runner's counts.json vocabulary.
SIGNATURES = (
    "agm",
    "peirce_resolution",
    "peirce_cap_exhaustion",
    "reichenbach",
    "ibe_real_candidates",
    "popper",
)

#: Minimum number of runs (of the executed RUNS) in which each principle's
#: activation signature must fire. STAGE-0 INITIALISATION — every entry is the
#: current, unmodified system's reality; see the ratchet semantics above.
MIN_ACTIVATIONS = {
    # AGM belief contraction. Audit (FINDINGS.md §9): 0 contractions in 150
    # runs — starved by construction (Investigate only runs on HYPOTHESIS
    # claims, revalidation requires stage != HYPOTHESIS). Target: >= 3 of 5
    # after Stage 2b's reachability fix (plan §5, Stage 2b).
    "agm": 0,
    # Peirce resolution: the inquiry terminated by verdict, not by the cap.
    # Audit: 52/150 runs adjudicated-with-no-cap on dev30 — nonzero but
    # unmeasured on this fixture; ratchet to the Stage-0 baseline floor once
    # the 5-run ledger block exists. Definition-of-done target: >= 3 of 5
    # (plan §4.6).
    "peirce_resolution": 0,
    # Peirce cap exhaustion: counted SEPARATELY from resolution so the ratio
    # is visible (audit: 45/150 runs capped; the audit's 88%-exhaustion
    # figure used claims-with-an-active-inquiry as its denominator — state
    # the denominator when ratcheting). No stage targets MORE capping; this
    # entry exists so a stage that makes convergence WORSE is caught.
    "peirce_cap_exhaustion": 0,
    # Reichenbach corroboration. Fires TODAY via HDBSCAN cluster size — the
    # one principle whose baseline is nonzero — but what it computes is wrong
    # on the merits (rewards redundancy; plan §4.4). Stage 1 REDEFINES the
    # signature (corroboration from independence_group membership), and in
    # this single-group fixture the CORRECT post-fix count is 0: one corpus
    # is one source agreeing with itself. Stage 1 therefore ratchets this
    # entry by redefining what "fired" means (and extending the fixture with
    # a second independence group), not by raising this number.
    "reichenbach": 0,
    # IBE genuine competing candidates. Audit: 81% of claims that reached
    # enumeration scored placeholder candidates; only ~19% had >= 2 genuine.
    # Target: >= 3 of 5 after Stage 3's candidate-quality gate.
    "ibe_real_candidates": 0,
    # Popper refutation-seeking search. Audit: 0 under corpus mode — the op
    # is hard-wired to a provider literally named "web_search" and cannot run
    # when that provider is absent (plan §4.5). An honest zero, not a never-
    # fires: target >= 3 of 5 after Stage 1's capability query, on a fixture
    # whose corpus holds the refuter (doc 9003).
    "popper": 0,
}


def test_every_principle_meets_its_activation_floor() -> None:
    """Run the canary over the fixture corpus and check every floor.

    Fail-loud end to end: a nonzero runner exit (any run raised, produced no
    verdict, hit retrieval_failed, gathered nothing, or leaked a non-corpus
    evidence row) fails this test rather than being read as a green baseline.
    """
    # sys.executable, not "uv run python": this test already runs INSIDE the
    # project venv (uv run pytest), so the interpreter is the same one the
    # runner needs — and the invocation does not silently depend on uv being
    # on the spawned PATH.
    cmd = [
        sys.executable, str(RUNNER),
        "--runs", str(RUNS),
        "--model", MODEL,
        "--out-dir", str(TEST_OUT_DIR),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        timeout=TIMEOUT_SECONDS,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "canary runner exited nonzero — a run errored, was invalid, or "
        "violated the offline guarantee.\n"
        f"stderr tail:\n{proc.stderr[-4000:]}"
    )

    counts_path = TEST_OUT_DIR / "counts.json"
    assert counts_path.exists(), f"runner produced no counts.json at {counts_path}"
    counts = json.loads(counts_path.read_text())

    totals = counts["totals"]
    assert totals["n_valid_runs"] == RUNS, (
        f"expected {RUNS} valid runs, got {totals['n_valid_runs']} "
        f"(errors: {counts['errors']})"
    )
    assert not counts["errors"], f"error runs recorded: {counts['errors']}"

    # The offline guarantee is re-checked by the CONSUMER, not just trusted
    # from the runner's own gate: under corpus mode every evidence row of
    # every run must read source_type == "scifact_corpus".
    for run in counts["per_run"]:
        offending = set(run["evidence_source_types"]) - {"scifact_corpus"}
        assert not offending, (
            f"run {run['run']} leaked live evidence ({offending}) — the "
            "offline wiring in run_canary.py is broken"
        )

    # Every signature must be present and counted — schema drift between the
    # runner and this test must fail loud, not silently pass.
    assert set(totals["signatures"]) == set(SIGNATURES), (
        f"counts.json signature set drifted: {sorted(totals['signatures'])} "
        f"vs expected {sorted(SIGNATURES)}"
    )

    for name in SIGNATURES:
        fired_runs = totals["signatures"][name]["fired_runs"]
        minimum = MIN_ACTIVATIONS[name]
        assert fired_runs >= minimum, (
            f"{name} fired in {fired_runs} of {RUNS} runs, below its floor "
            f"of {minimum}. If a stage claimed this fix, the fix did not "
            "deliver; if no stage has ratcheted this entry yet, see "
            "MIN_ACTIVATIONS' ratchet semantics and the Stage-0 baseline "
            "block in docs/epistemic/FIDELITY_LEDGER.md."
        )