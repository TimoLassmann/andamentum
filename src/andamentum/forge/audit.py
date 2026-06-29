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

import sys
from pathlib import Path

from andamentum.agentic_dialect import check_code

from .agents import CRITIC, REQUIREMENTS, AgentSink
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
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
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
    detail = (
        "passed"
        if res.ok
        else _pytest_summary(res.stdout or res.stderr or f"exit {res.exit_code}")
    )
    return CheckResult(name="tests", passed=res.ok, detail=detail)


def _run_dialect(spec: SystemSpec, dest: Path) -> CheckResult:
    violations = check_code(dest / spec.name)
    if not violations:
        return CheckResult(
            name="dialect", passed=True, detail="package is dialect-clean"
        )
    detail = "; ".join(
        f"{v.file.split('/')[-1]}:{v.line} [{v.law}] {v.code}" for v in violations[:8]
    )
    return CheckResult(name="dialect", passed=False, detail=detail)


async def audit_system(
    spec: SystemSpec,
    brief: str,
    dest: Path,
    *,
    sink: AgentSink,
    sandbox: SandboxPort,
    build: BuildReport | None,
    reporter: ForgeReporter | None = None,
) -> AuditReport:
    """Audit the assembled system at ``dest/<spec.name>``."""
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

    rep.audit_check(
        name="requirements", status="running", detail="does it serve the brief?"
    )
    requirements = await sink.run(
        REQUIREMENTS, brief=brief, system=_system_summary(spec)
    )
    assert isinstance(requirements, RequirementsVerdict)
    rep.audit_check(
        name="requirements",
        status="passed" if requirements.meets_brief else "failed",
        detail="",
    )
    for gap in requirements.gaps:
        issues.append(AuditIssue(source="requirements", detail=gap))

    rep.audit_check(name="critic", status="running", detail="adversarial review")
    critic = await sink.run(CRITIC, bodies=_node_bodies(pkg))
    assert isinstance(critic, CriticVerdict)
    rep.audit_check(
        name="critic",
        status="passed" if not critic.issues else "failed",
        detail="",
    )
    for problem in critic.issues:
        issues.append(AuditIssue(source="critic", detail=problem))

    works = tests.passed and dialect.passed and not remaining
    return AuditReport(
        works=works,
        rounds=1,
        checks=checks,
        requirements=requirements,
        critic=critic,
        remaining_holes=remaining,
        issues=issues,
    )
