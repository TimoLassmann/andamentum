"""Stage 4 — the whole-system audit: does the built system actually work?

This is where generated code finally *executes* — behind the sandbox, never in the
forge process. Four checks at four altitudes:

  tests    [sandbox] — run the system's own shipped test suite (assembles + smoke-runs
                       the graph end-to-end with stub agents) in the container.
  dialect  [det]     — the dialect's own ``check_code`` over the built package (forge
                       verifying its output against the canon).
  requirements [agent] — does the built system serve the brief?
  critic   [agent]   — adversarial pass: what is missing, wrong, or faked?

``works`` is true only when every hole is filled, the tests pass, and the package is
dialect-clean. Requirements/critic are advisory quality signals recorded as issues.
Engine-free leaf worker: it calls agents through a Port and runs code through the
sandbox Port, but imports no graph engine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from andamentum.agentic_dialect import check_code

from .agents import CRITIC, MODEL_OUTPUT_ERRORS, REQUIREMENTS, AgentSink
from .diagnose import _NEAR_MISS_THRESHOLD, _nearest
from .extract import discover_holes
from .reporter import ForgeReporter, NoopReporter
from .sandbox import SandboxPort, SandboxUnavailableError
from .schemas import (
    AuditIssue,
    AuditReport,
    BuildReport,
    CheckResult,
    CriticVerdict,
    RequirementsVerdict,
)
from .spec import SystemSpec

_TEST_TIMEOUT = 180


def _system_summary(spec: SystemSpec) -> str:
    lines = [f"PURPOSE: {spec.description}", "NODES:"]
    for n in spec.nodes:
        kind = "head(LLM)" if n.kind.value == "head" else "spine"
        net = " [network]" if n.network else ""
        lines.append(f"- {n.name} ({kind}){net}: {n.job or n.purpose}")
    return "\n".join(lines)


def _node_bodies(pkg: Path) -> str:
    f = pkg / "nodes.py"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def _pytest_summary(output: str) -> str:
    """The meaningful lines of a pytest run — the FAILED/ERROR lines and the final
    count — not the trailing warnings block (which buries the real verdict)."""
    lines = output.splitlines()
    failed = [ln.strip() for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    summary = next(
        (
            ln.strip()
            for ln in reversed(lines)
            if (" passed" in ln or " failed" in ln or " error" in ln) and "===" in ln
        ),
        "",
    )
    picked = "; ".join([*failed, summary]).strip("; ")
    return picked or output.strip()[-400:] or "no output"


def _parse_counts(output: str) -> tuple[int, int, int]:
    """The ``(tests_passed, tests_failed, tests_errored)`` off the pytest summary line
    (``=== N passed, M failed, K error(s) ===``). ``error`` is pytest's distinct marker
    for a collection/import/setup failure (the package would not import) — as opposed to a
    ``failed`` assertion (a test ran and a node produced the wrong value). Attribution uses
    the error count to route an import failure to a loud terminal (re-authoring a body's
    logic cannot fix a package that will not import), while a ``failed`` assertion still
    flows to the normal rebuild path."""
    # Parse ONLY pytest's summary line, never the whole output — an assertion message or
    # captured stdout containing prose like "got 2 errors" must not be miscounted (a
    # behavioural failure would then be misread as an import error). The summary line is
    # identified by pytest's trailing duration (``... in 0.12s``, optionally ``===``-wrapped)
    # alongside a count word; assertion prose has a count word but no duration.
    summary = next(
        (
            ln
            for ln in reversed(output.splitlines())
            if re.search(r"\d+ (?:passed|failed|error)", ln)
            and (re.search(r" in \d+(?:\.\d+)?s", ln) or "===" in ln)
        ),
        "",
    )
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0
    errored = int(m.group(1)) if (m := re.search(r"(\d+) error", summary)) else 0
    return passed, failed, errored


def _named_check(audit: AuditReport, name: str) -> CheckResult | None:
    return next((c for c in audit.checks if c.name == name), None)


def audit_rank(audit: AuditReport) -> tuple[bool, int, int, bool]:
    """The regression guard's total order (§4.5): rank an audit by
    ``(works, tests_passed, -tests_failed, dialect_clean)``. A strictly greater tuple is
    a strictly better audit (a tie is non-improving). Pure — reads the retained counts
    off the ``tests``/``dialect`` checks; a missing check ranks as 0 / not-clean."""
    tests = _named_check(audit, "tests")
    dialect = _named_check(audit, "dialect")
    passed = tests.tests_passed if tests is not None else 0
    failed = tests.tests_failed if tests is not None else 0
    clean = dialect.passed if dialect is not None else False
    return (audit.works, passed, -failed, clean)


def failing_checks_summary(audit: AuditReport) -> str:
    """A short one-line summary of the checks that failed this pass (empty when every
    check passed) — the ``AuditRound.failing_checks`` record and the per-target rebuild
    feedback fall back to it."""
    return "; ".join(f"{c.name}: {c.detail}" for c in audit.checks if not c.passed)


def feedback_for_targets(audit: AuditReport, targets: list[str]) -> dict[str, str]:
    """Map each re-authored target to the audit-failure text it should see when
    re-authored (§4.3 step 3): the audit issues that name it, else the overall
    failing-check summary — so every target sees a concrete failure to address."""
    summary = failing_checks_summary(audit)
    out: dict[str, str] = {}
    for t in targets:
        mentions = [i.detail for i in audit.issues if t in i.detail]
        out[t] = "\n".join(mentions) if mentions else summary
    return out


def _run_tests(sandbox: SandboxPort, spec: SystemSpec, dest: Path) -> CheckResult:
    # The container mounts the RESOLVED dest (e.g. /tmp → /private/tmp on macOS), so any
    # path handed to pytest in argv must be resolved too — an unresolved path is not
    # mounted inside the container and pytest would collect nothing.
    dest = dest.resolve()
    pkg = dest / spec.name
    try:
        res = sandbox.run(
            # `-p no:cacheprovider`: the package is mounted read-only in the container,
            # so pytest's on-disk cache can't be written — disable it (else a noisy,
            # harmless warning, never a failure).
            # `--tb=short`: pin the traceback style so `nodes.py:line` frames are
            # emitted for every failure (attribution signal 2), never left to the
            # default `--tb=auto`.
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                str(pkg / "tests"),
            ],
            cwd=dest,
            extra_path=dest,
            timeout=_TEST_TIMEOUT,
            allow_network=spec.has_network,
        )
    except SandboxUnavailableError as e:
        return CheckResult(name="tests", passed=False, detail=str(e))
    if res.timed_out:
        return CheckResult(
            name="tests",
            passed=False,
            detail="the test run timed out (possible infinite loop)",
        )
    raw = res.stdout or res.stderr or ""
    tests_passed, tests_failed, tests_errored = _parse_counts(raw)
    detail = "passed" if res.ok else _pytest_summary(raw or f"exit {res.exit_code}")
    return CheckResult(
        name="tests",
        passed=res.ok,
        detail=detail,
        raw_output=raw,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errored=tests_errored,
    )


def _reconcile_node(name: str, node_names: list[str]) -> str:
    """Map a critic-named node string to a REAL spec node name (§4.4 signal 4).

    Node names are per-spec, so the critic emits a free string; reconcile it to an
    actual node via the same tiered ``partial_ratio`` match ``diagnose`` uses for
    near-miss variable names. An exact hit wins outright; otherwise the closest name
    above ``_NEAR_MISS_THRESHOLD`` wins. Returns ``""`` when nothing reconciles — an
    unreconcilable name is dropped (never a rebuild target)."""
    if name in node_names:
        return name
    nearest, score = _nearest(name, node_names)
    return nearest if nearest and score >= _NEAR_MISS_THRESHOLD else ""


def _run_dialect(spec: SystemSpec, dest: Path) -> CheckResult:
    violations = check_code(dest / spec.name)
    if not violations:
        return CheckResult(
            name="dialect", passed=True, detail="package is dialect-clean"
        )
    detail = "; ".join(
        f"{v.file.split('/')[-1]}:{v.line} [{v.law}] {v.code}" for v in violations[:8]
    )
    # Retain the untruncated structured list (attribution signal 1) alongside the
    # human-readable, capped `detail` — never re-parsed from the string.
    return CheckResult(
        name="dialect", passed=False, detail=detail, violations=list(violations)
    )


async def audit_system(
    spec: SystemSpec,
    brief: str,
    dest: Path,
    *,
    sink: AgentSink,
    sandbox: SandboxPort,
    build: BuildReport | None,
    audit_rounds: int = 0,
    reporter: ForgeReporter | None = None,
) -> AuditReport:
    """Audit the assembled system at ``dest/<spec.name>``.

    ``audit_rounds`` is the number of rebuilds performed so far (§4.2); the report's
    ``rounds`` is set to ``audit_rounds + 1`` — the audit-pass count. Passed as a plain
    int so this worker stays engine-free (no ForgeState access)."""
    rep: ForgeReporter = reporter if reporter is not None else NoopReporter()
    pkg = dest / spec.name
    checks: list[CheckResult] = []
    issues: list[AuditIssue] = []

    remaining = [h.node for h in discover_holes(pkg)]
    if build is not None:
        remaining = sorted(set(remaining) | set(build.remaining_holes))

    rep.audit_check(name="tests", status="running", detail="shipped tests in sandbox")
    tests = _run_tests(sandbox, spec, dest)
    rep.audit_check(
        name="tests", status="passed" if tests.passed else "failed", detail=tests.detail
    )
    checks.append(tests)
    if not tests.passed:
        issues.append(AuditIssue(source="tests", detail=tests.detail))

    rep.audit_check(name="dialect", status="running", detail="check_code over package")
    dialect = _run_dialect(spec, dest)
    rep.audit_check(
        name="dialect",
        status="passed" if dialect.passed else "failed",
        detail=dialect.detail,
    )
    checks.append(dialect)
    if not dialect.passed:
        issues.append(AuditIssue(source="dialect", detail=dialect.detail))

    # Requirements + critic are ADVISORY heads (they inform issues, never the `works`
    # verdict). If the model call itself fails, skip the head rather than crash the audit —
    # a hard LLM failure here must not sink an otherwise-complete run.
    rep.audit_check(
        name="requirements", status="running", detail="does it serve the brief?"
    )
    requirements: RequirementsVerdict | None
    try:
        out = await sink.run(REQUIREMENTS, brief=brief, system=_system_summary(spec))
        assert isinstance(out, RequirementsVerdict)
        requirements = out
    except MODEL_OUTPUT_ERRORS:
        requirements = None
    if requirements is not None:
        rep.audit_check(
            name="requirements",
            status="passed" if requirements.meets_brief else "failed",
            detail="",
        )
        for gap in requirements.gaps:
            issues.append(AuditIssue(source="requirements", detail=gap))
    else:
        rep.audit_check(
            name="requirements", status="skipped", detail="model unavailable"
        )

    rep.audit_check(name="critic", status="running", detail="adversarial review")
    node_names = [n.name for n in spec.nodes]
    critic: CriticVerdict | None
    try:
        out = await sink.run(
            CRITIC, bodies=_node_bodies(pkg), nodes="\n".join(node_names)
        )
        assert isinstance(out, CriticVerdict)
        critic = out
    except MODEL_OUTPUT_ERRORS:
        critic = None
    if critic is not None:
        rep.audit_check(
            name="critic",
            status="passed" if not critic.issues else "failed",
            detail="",
        )
        for problem in critic.issues:
            # Reconcile the critic's free node name to a real spec node; when it reconciles,
            # surface the attribution in the issue, else record the bare problem.
            node = _reconcile_node(problem.node, node_names)
            detail = f"{node}: {problem.issue}" if node else problem.issue
            issues.append(AuditIssue(source="critic", detail=detail))
    else:
        rep.audit_check(name="critic", status="skipped", detail="model unavailable")

    works = tests.passed and dialect.passed and not remaining
    return AuditReport(
        works=works,
        rounds=audit_rounds + 1,
        checks=checks,
        requirements=requirements,
        critic=critic,
        remaining_holes=remaining,
        issues=issues,
    )
