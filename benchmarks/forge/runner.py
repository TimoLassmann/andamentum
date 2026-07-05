"""Drive forge over a benchmark case and score the runs.

``run_case`` runs forge ``runs`` times over one brief and aggregates the outcomes into a
:class:`CaseScore`.

- **Tier 1** (``full=False``, default) — design-only: ``run_forge(brief, model=...,
  dest=None)`` runs understand→frame→decompose→compile→review and returns a ``ForgeResult``
  whose ``.spec`` is inspected for control-flow features. Scored on design *shape*.
- **Tier 2** (``full=True``) — end-to-end: ``run_forge(brief, model=..., dest=<tmp>,
  stop_after="audit")`` renders, agent-authors every node body, and sandbox-audits the
  package. Scored on whether the generated system actually **works** (``audit.works`` —
  holes filled, tests pass, dialect-clean). This is the reliability signal Tier 1 cannot see.

A ``ValueError`` is forge refusing at the fitness gate (or failing a coherence loop) — a
refusal either tier. An optional ``sink`` injects a stub ``AgentSink`` so the whole path
runs with no model (the offline self-tests).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from andamentum.forge import run_forge
from andamentum.forge.agents import AgentSink
from andamentum.forge.sandbox import SandboxPort

from .shape import detect_features, outcome_matches, outcome_matches_tier2
from .types import Case, CaseScore, RunOutcome


async def run_case(
    case: Case,
    *,
    model: str,
    runs: int = 3,
    full: bool = False,
    sandbox_backend: str = "subprocess",
    sink: AgentSink | None = None,
    sandbox: SandboxPort | None = None,
) -> CaseScore:
    """Run forge ``runs`` times over ``case`` and score the outcomes.

    ``full=False`` (Tier 1) scores design shape; ``full=True`` (Tier 2) renders + builds +
    sandbox-audits and scores on whether the system works. ``sandbox_backend`` selects the
    Tier-2 execution seam (``"podman"`` for real isolation incl. network briefs,
    ``"subprocess"`` for the no-container fallback — refuses network nodes). Passing
    ``sink`` / ``sandbox`` drives forge with stubs (the offline self-test path).
    """
    outcomes: list[RunOutcome] = []
    for _ in range(runs):
        outcomes.append(
            await _run_once(
                case,
                model=model,
                full=full,
                sandbox_backend=sandbox_backend,
                sink=sink,
                sandbox=sandbox,
            )
        )

    matches = outcome_matches_tier2 if full else outcome_matches
    passes = sum(1 for o in outcomes if matches(case, o))
    total = len(outcomes)
    return CaseScore(
        case=case,
        runs=outcomes,
        passes=passes,
        total=total,
        pass_rate=(passes / total if total else 0.0),
    )


async def _run_once(
    case: Case,
    *,
    model: str,
    full: bool,
    sandbox_backend: str,
    sink: AgentSink | None,
    sandbox: SandboxPort | None,
) -> RunOutcome:
    """One forge run over a brief, mapped to a :class:`RunOutcome`."""
    if not full:
        return await _run_once_tier1(case, model=model, sink=sink)
    return await _run_once_tier2(
        case, model=model, sandbox_backend=sandbox_backend, sink=sink, sandbox=sandbox
    )


async def _run_once_tier1(
    case: Case, *, model: str, sink: AgentSink | None
) -> RunOutcome:
    """Design-only run: score the spec's control-flow shape."""
    try:
        result = await run_forge(case.brief, model=model, dest=None, sink=sink)
    except ValueError as exc:
        return RunOutcome(kind="refused", error=str(exc))
    except Exception as exc:  # genuine design failure (a crash, not a refusal)
        return RunOutcome(kind="design_failed", error=str(exc))
    return RunOutcome(kind="built", features=detect_features(result.spec))


async def _run_once_tier2(
    case: Case,
    *,
    model: str,
    sandbox_backend: str,
    sink: AgentSink | None,
    sandbox: SandboxPort | None,
) -> RunOutcome:
    """End-to-end run: render + agent-author + sandbox-audit, score on ``audit.works``.

    A fresh temp destination per run keeps each build isolated. ``ValueError`` is a
    refusal; any other exception is a build-stage crash. On success the outcome carries
    the reliability signals (holes filled/total, test counts, remaining holes).
    """
    with tempfile.TemporaryDirectory(prefix="forge-bench-") as tmp:
        try:
            result = await run_forge(
                case.brief,
                model=model,
                dest=Path(tmp),
                stop_after="audit",
                sandbox_backend=sandbox_backend,
                sink=sink,
                sandbox=sandbox,
            )
        except ValueError as exc:
            return RunOutcome(kind="refused", error=str(exc))
        except Exception as exc:  # a crash in render/build/audit, not a refusal
            return RunOutcome(kind="build_failed", error=str(exc))
        return _tier2_outcome(result)


def _tier2_outcome(result: object) -> RunOutcome:
    """Map a completed Tier-2 ``ForgeResult`` to a scored :class:`RunOutcome`."""
    audit = getattr(result, "audit", None)
    build = getattr(result, "build", None)
    stage = getattr(result, "stage_reached", "")

    holes_total = 0
    holes_filled = 0
    remaining: list[str] = []
    if build is not None:
        holes_filled = len(getattr(build, "filled", []))
        remaining = list(getattr(build, "remaining_holes", []))
        holes_total = holes_filled + len(remaining)

    tests_passed = tests_failed = 0
    if audit is not None:
        for check in getattr(audit, "checks", []):
            if getattr(check, "name", "") == "tests":
                tests_passed = getattr(check, "tests_passed", 0)
                tests_failed = getattr(check, "tests_failed", 0)

    works = bool(getattr(audit, "works", False)) if audit is not None else False
    kind = "works" if works else "incomplete"
    return RunOutcome(
        kind=kind,
        works=works,
        stage_reached=stage,
        holes_filled=holes_filled,
        holes_total=holes_total,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        remaining_holes=remaining,
    )


async def run_all(
    cases: list[Case],
    *,
    model: str,
    runs: int,
    full: bool,
    sandbox_backend: str = "subprocess",
    sink: AgentSink | None = None,
    sandbox: SandboxPort | None = None,
) -> list[CaseScore]:
    """Score every case in ``cases`` (sequentially, to keep model load bounded)."""
    scores: list[CaseScore] = []
    for case in cases:
        scores.append(
            await run_case(
                case,
                model=model,
                runs=runs,
                full=full,
                sandbox_backend=sandbox_backend,
                sink=sink,
                sandbox=sandbox,
            )
        )
    return scores
