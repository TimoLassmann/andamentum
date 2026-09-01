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
    # STAGE 2B MEASURED (2026-09-01, ollama:gemma4:31b-cloud, both canary
    # claims, 5 runs each, fresh dirs — docs/epistemic/FIDELITY_LEDGER.md
    # "Stage 2" block, finding S2-2): claim 2 (vitamin D/mortality) 5/5 —
    # agm fires EVERY run, exceeding the >= 3/5 target on first
    # measurement. This is the first genuine AGM contraction observed
    # anywhere in this project (audit: 0/150; every prior canary
    # measurement on both claims: 0/5). Mechanism, traced end to end: the
    # claim promotes to SUPPORTED, RunVerification's adversarial track
    # finds real counter-evidence (adversarial_balance ~0.26-0.27, below
    # ADVERSARIAL_REFUTED_THRESHOLD=0.3), Stage 2b's fix routes that into
    # the TMS sweep, RevalidateClaimOperation correctly demotes.
    # promotion_history is identical in shape across all 5 runs — exactly
    # one promote, one demote, no further cycling — so the new demote
    # path terminates cleanly (L5 exercised for real, not just
    # unit-tested). claim 1 (Parkinson's/trehalose) stays 0/5 —
    # UNCHANGED from every prior measurement (2a, 2b-pre) — because it
    # never reaches SUPPORTED at all: it cycle-caps directly out of
    # HYPOTHESIS via genuine scrutiny/uncertainty exhaustion, a defect
    # ORTHOGONAL to the Stage 2b reachability fix and not diagnosed in
    # this session (ledger finding S2-1). A 30-case real-dev30 diagnostic
    # (same date/model) additionally found the reachability fix wired
    # correctly in every one of 12/30 real cases that reach it (verified
    # 1:1 correlation between adversarial_search and revalidate_claim
    # dispatch), but agm fired 0/30 there — real evidence pools rarely
    # push adversarial_balance under the threshold the way the
    # purpose-built canary fixture does (ledger finding S2-4; a rarity
    # finding, not a reachability failure).
    # FLOOR STAYS 0, DELIBERATELY, DESPITE THE PROVEN FIX: this file's
    # own RUNNER invocation below passes no --claim-id, so
    # load_claims(...)[0] (claim 1) is what THIS test actually exercises
    # — never claim 2. A floor above 0 here would assert a rate claim 1's
    # own canary run can never produce, i.e. a permanently-red gate, not
    # an honest measured floor. Widening this test to also cover claim 2
    # (the fixture purpose-built to exercise this signature) is a real,
    # open follow-up — flagged, not silently done here (out of this
    # task's stated scope; see the ledger's "MIN_ACTIVATIONS decision"
    # note under Stage 2).
    "agm": 0,
    # Peirce resolution: the inquiry terminated by verdict, not by the cap.
    # Audit: 52/150 runs adjudicated-with-no-cap on dev30. STAGE-0 MEASURED
    # (2026-08-31): 0/5 on this fixture — the claim never leaves the
    # Scrutinize->ResolveUncertainties loop, so the floor stays 0 (finding
    # S0-2 in the ledger). Definition-of-done target: >= 3 of 5 (plan §4.6).
    "peirce_resolution": 0,
    # Peirce cap exhaustion: counted SEPARATELY from resolution so the ratio
    # is visible (audit: 45/150 dev30 runs capped; the audit's 88%-exhaustion
    # figure used claims-with-an-active-inquiry as its denominator — state
    # the denominator when ratcheting). STAGE-0 RATCHET (2026-08-31): the
    # 5-run ledger baseline measured 5/5, and the archived corroborating runs
    # 4/4 — the cap fired in every genuine run of this fixture, so the armed
    # floor is RUNS (2 of 2). This is a machinery-presence guard: if the cap
    # STOPS firing here, the termination path this fixture exercises has
    # changed, and the stage that changed it must re-register this entry in
    # the same commit that ratchets its own principle's (Stage 2b will
    # legitimately move this number if claims start resolving or demoting
    # before the cap — that is a success, recorded, never a silent pass).
    "peirce_cap_exhaustion": 2,
    # Reichenbach corroboration. The audit saw it firing via HDBSCAN cluster
    # size on decomposed dev30 runs, but what it computes is wrong on the
    # merits (rewards redundancy; plan §4.4). STAGE-0 MEASURED (2026-08-31):
    # 0/5 on this fixture — seed-claim verify mode never reaches
    # select_top_k_evidence, so every evidence row stays "unclustered" and
    # the signature's denominator is EMPTY here (finding S0-1 in the ledger).
    # STAGE 1 FIX LANDED AND VERIFIED (2026-08-31): corroboration_count is now
    # computed from (content cluster, independence_group) pairs agreeing per
    # direction, not raw HDBSCAN cluster size (operations/claims.py); the old
    # signal survives, honestly renamed, as Evidence.retrieval_cluster_size.
    # STAGE 1 MEASURED, claim 1, 5/5 runs, ollama:gemma4:31b-cloud,
    # 2026-08-31: still 0/5 — NOT a computed 0, the SAME empty-denominator
    # artifact as Stage 0: select_top_k_evidence never executes because
    # ClusterEvidence only processes stage==SUPPORTED claims, and this claim
    # is cycle-capped straight out of "hypothesis" without ever promoting
    # (finding S1-1 in the ledger — also confirmed unchanged on claim 2, the
    # fixture extension purpose-built to exercise this signature). The fix's
    # correctness is confirmed separately, on a real-corpus dev30-shaped
    # diagnostic (5 cases, same date/model, see the ledger): 4 of 5 cases DID
    # promote and reach select_top_k_evidence, and corroboration_count
    # demonstrably decoupled from retrieval_cluster_size (e.g.
    # corroboration_count=7 vs retrieval_cluster_size=1 on several
    # representatives in one case) — direct proof the old bug is gone. The
    # floor stays 0 here because the CANARY claim never reaches the code
    # path, not because the fix doesn't work; Stage 2b's reachability fix
    # (the same one AGM needs) is the dependency for raising this floor, not
    # a further Reichenbach change.
    # STAGE 2B MEASURED (2026-09-01, ollama:gemma4:31b-cloud): claim 2,
    # 5/5 runs — reichenbach fires EVERY run once Stage 2b's reachability
    # fix is in place (docs/epistemic/FIDELITY_LEDGER.md "Stage 2" block),
    # exceeding the >= 3/5 target. claim 1 stays 0/5 (still the same
    # empty-denominator artifact — see the agm entry's Stage 2b note for
    # why claim 1 never reaches SUPPORTED). A 30-case real-dev30
    # diagnostic (same date/model) independently corroborates the fix on
    # real data: reichenbach fires in 21/30 cases — strong and
    # consistent with Stage 1's own real-corpus proof. FLOOR STAYS 0 for
    # the same reason as agm's: this file's RUNNER invocation exercises
    # only claim 1 by default (no --claim-id), and claim 1 never reaches
    # select_top_k_evidence, so a floor above 0 would be unmeetable by
    # this specific test as written — see the agm entry and the ledger's
    # "MIN_ACTIVATIONS decision" note under Stage 2 for the full picture.
    "reichenbach": 0,
    # IBE genuine competing candidates. Audit: 81% of claims that reached
    # enumeration scored placeholder candidates; only ~19% had >= 2 genuine.
    # STAGE-0 MEASURED (2026-08-31): 0/5, with claims_reached_enumeration=0
    # in every run — the audit's ~1-in-5 is conditioned on a denominator this
    # fixture never produces (finding S0-3 in the ledger), so the floor stays 0.
    # Target: >= 3 of 5 after Stage 3's candidate-quality gate.
    # STAGE 3 FIX LANDED AND VERIFIED (2026-09-01):
    # EnumerateCandidatesOperation now routes a claim with < 2 genuine
    # candidates to an explicit "insufficient explanatory diversity" decline
    # (docs/epistemic/FIDELITY_LEDGER.md "Stage 3" block) instead of seeding
    # 3 synthetic defaults and scoring them as rivals.
    # STAGE 3 MEASURED, both canary claims, 5 runs each, ollama:gemma4:31b-
    # cloud, 2026-09-01: claim 1 1/5 (0.2), claim 2 0/5 (0.0) — **MISS vs
    # the >= 3/5 target on BOTH claims** (finding S3-1/S3-2). Root cause is
    # reachability, not the fix: claim 1 escapes the cycle cap into
    # enumeration in only 1/5 runs (the same S2-1 pre-promotion defect that
    # already gates agm/reichenbach/popper on this claim) and fires
    # correctly in that one run; claim 2 now never reaches enumeration at
    # all (0/5) because Stage 2b's OWN AGM fix demotes it via TMS before the
    # run reaches the integration stage, consuming its only promotion
    # window (S2-3's oscillation_detected consequence, unchanged from Stage
    # 2 — not a new Stage-3 regression). The fix's own correctness is proven
    # directly on real data instead: a 30-case dev30 diagnostic (same
    # date/model) measured the PLACEHOLDER-SELECTION rate — the actual
    # defect the audit found — at 0/11 (0.0%) among cases that reached
    # enumeration (3 declined correctly with exactly 1 genuine candidate; 8
    # reached full IBE selection and every one chose a genuine candidate
    # over any seeded default sitting alongside it), against the audit's
    # measured 81%. `ibe_real_candidates` itself fired 8/30 (vs Stage 2's
    # 10/30 — within this project's documented 10-27% run-to-run
    # stochasticity, not a regression; Stage 3 does not change how a
    # >=2-genuine claim is counted, only what happens to the <2-genuine
    # ones).
    # FLOOR STAYS 0, DELIBERATELY: unlike agm/reichenbach/popper (0/5 on
    # claim 1, an empty-denominator artifact), claim 1's true rate here is a
    # real but infrequent ~1-in-5 (1/5 measured). This file's RUNNER
    # invocation exercises only claim 1, RUNS=2 — at a true rate of ~20%, a
    # 2-run gate has roughly a 64% chance of observing zero fires by chance
    # alone even when the fix works exactly as measured. Raising the floor
    # to even 1 would make the gate flaky against a correct system, not an
    # honest measured floor. Same open follow-up as agm/reichenbach/popper:
    # widening the pre-merge canary to cover claim 2, or raising RUNS, is
    # available and unaddressed — see the ledger's Stage 3 "MIN_ACTIVATIONS
    # decision" note for the full picture.
    "ibe_real_candidates": 0,
    # Popper refutation-seeking search. Audit: 0 under corpus mode — the op
    # is hard-wired to a provider literally named "web_search" and cannot run
    # when that provider is absent (plan §4.5).
    # STAGE 1 FIX LANDED AND VERIFIED (2026-08-31): AdversarialSearchOperation
    # now asks the gatherer a capability question (capable_of("adversarial"))
    # instead of hard-coding "web_search" (operations/verification.py,
    # operations/base.py's new CapabilityQueryable protocol).
    # STAGE 1 MEASURED, claim 1, 5/5 runs, ollama:gemma4:31b-cloud,
    # 2026-08-31: still 0/5 — adversarial_search appears ZERO times in
    # operations.jsonl across all 5 runs: the op is never INVOKED, not
    # invoked-and-declined (capability_decline_reason is never even reached).
    # Root cause is identical to reichenbach's, above: RunVerification (which
    # dispatches this op) is reached only via ClusterEvidence, gated on
    # stage==SUPPORTED, and this claim never promotes (finding S1-2 in the
    # ledger — also confirmed unchanged on claim 2). The fix's correctness is
    # confirmed separately, on the same real-corpus dev30-shaped diagnostic:
    # AdversarialSearchOperation fired successfully in 3 of 5 real cases,
    # correctly dispatching to the registered corpus provider (not the old
    # hard-coded "web_search"), gathering and evaluating real
    # counterarguments with correct provenance. The floor stays 0 here for
    # the same reason as reichenbach's: Stage 2b's reachability fix is the
    # dependency, not a further Popper change.
    # STAGE 2B MEASURED (2026-09-01, ollama:gemma4:31b-cloud): claim 2,
    # 5/5 runs — popper fires EVERY run once Stage 2b's reachability fix
    # is in place (docs/epistemic/FIDELITY_LEDGER.md "Stage 2" block),
    # exceeding the >= 3/5 target; claim 1 stays 0/5 (same root cause as
    # agm/reichenbach — see the agm entry's note). A 30-case real-dev30
    # diagnostic (same date/model) corroborates on real data: popper
    # fires in 12/30 cases, and raw operations.jsonl shows the
    # reachability wiring itself is correct in EVERY case that reaches it
    # (adversarial_search and revalidate_claim dispatch together, 1:1,
    # in all 12) — the other 18/30 cases never reach RunVerification at
    # all (claim didn't promote), not a capability-query decline.
    # FLOOR STAYS 0 for the same structural reason as agm/reichenbach:
    # this file's RUNNER invocation exercises only claim 1 by default —
    # see the agm entry and the ledger's "MIN_ACTIVATIONS decision" note
    # under Stage 2 for the full picture and the open follow-up.
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