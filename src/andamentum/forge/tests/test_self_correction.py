"""The self-correction loop (§4 / §9) as OFFLINE integration tests.

Every case runs the whole ``run_forge`` graph with **no live model and no container**: a
scripted :class:`ScriptedSink` answers the design + authoring + audit heads, and a stateful
fake ``SandboxPort`` scripts the *dynamic* verdict (a crash, a degradation, a clean pass).
The loop must therefore be fully exercisable from scripted stubs — which is the acceptance
bar of §9. The six cases here mirror §9.1–§9.6:

  1. a crashing body is attributed (traceback → node, signal 2), re-authored, converges;
  2. a node-body dialect violation is attributed (nodes.py-scoped check_code, signal 1) and
     repaired — while a *render-owned* violation settles as a loud terminal, never a rebuild;
  3. a critic-caught stand-in is attributed (reconciled NodeFinding, signal 4) and repaired;
  4. a genuinely unfixable failure settles loud (works=False, best build, full history) at
     the cap — no hang, no raise;
  5. a regression is caught (the count-based total order) and the kept best is re-materialised
     onto disk (§4.6);
  6. a clean first build does not fire the loop, and a rebuild does not duplicate audit rows.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

from andamentum.forge import run_forge
from andamentum.forge.graph import (
    MAX_AUDIT_ROUNDS,
    MAX_PLAN_REVIEW_ROUNDS,
    ForgeDeps,
)
from andamentum.forge.reporter import RichReporter
from andamentum.forge.schemas import (
    CriticVerdict,
    ForgeWhy,
    NodeFinding,
    NodeTyping,
    PieceOut,
    SandboxResult,
)
from andamentum.forge.spec import NodeKind

from .conftest import FakeSandbox, ScriptedSink, _draft_body


# --- scripted plans (the design front-end, canned) -------------------------------


def _reading_list_kwargs() -> dict[str, object]:
    """A one-spine plan: parse the request (spine) → answer it (head)."""
    return dict(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
        typings={
            "n1": NodeTyping(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeTyping(
                kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
            ),
        },
    )


def _two_spine_kwargs() -> dict[str, object]:
    """A two-spine plan (parse → normalise → answer). Two spine holes let a rebuild target
    *one* node while the spine fallback would target *both* — so an attribution test can
    prove which signal fired by the size of ``rebuild_targets``."""
    return dict(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={
            "core": ["Parse the request.", "Normalise the request.", "Answer the request."]
        },
        typings={
            "n1": NodeTyping(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeTyping(
                kind=NodeKind.SPINE,
                consumes=["parsed_request"],
                produces=["normalised"],
            ),
            "n3": NodeTyping(
                kind=NodeKind.HEAD, consumes=["normalised"], produces=["answer"]
            ),
        },
    )


def _first_class_line(source: str) -> int:
    """A line strictly inside the first generated node class — the frame a scripted crash
    points at, so signal-2 attribution maps it to that node."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.body[-1].lineno
    raise AssertionError("no node class in nodes.py")


def _authored_node(context: str) -> str:
    """The node name a build-draft/repair context is authoring (its ``NODE:`` header)."""
    first = context.splitlines()[0] if context else ""
    return first.split("NODE:", 1)[1].split("(")[0].strip() if "NODE:" in first else ""


# --- §9.1 — a crashing body is repaired ------------------------------------------


class _CrashOnceSandbox:
    """A stateful ``SandboxPort`` stub: the FIRST test run fails with a ``--tb=short``
    traceback frame inside a real node class (the scripted crash), every run after passes."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        self.calls += 1
        if self.calls > 1:
            return SandboxResult(exit_code=0, stdout="1 passed in 0.10s\n")
        # argv[-1] is <pkg>/tests; the sibling nodes.py holds the filled node classes.
        pkg = Path(argv[-1]).parent
        line = _first_class_line((pkg / "nodes.py").read_text())
        out = (
            "_________________ test_smoke _________________\n"
            f"{pkg.name}/nodes.py:{line}: in run\n"
            "    raise RuntimeError('boom')\n"
            "E   RuntimeError: boom\n"
            "1 failed in 0.10s\n"
        )
        return SandboxResult(exit_code=1, stdout=out)


async def test_self_correction_loop_repairs_a_crashing_body(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    sandbox = _CrashOnceSandbox()
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=reading_list_sink,
        sandbox=sandbox,
    )

    # The first audit crashed → the loop re-authored the attributed node and re-audited.
    assert sandbox.calls == 2, sandbox.calls
    assert result.works
    assert result.audit is not None and result.audit.works
    # audit_rounds incremented: two audit passes (rounds = audit_rounds + 1 = 2).
    assert result.audit.rounds == 2

    # One history entry per audit pass; the first names the re-authored target.
    assert len(result.audit_history) == 2
    first, second = result.audit_history
    assert first.index == 1
    assert "ParseTheRequest" in first.rebuild_targets
    assert first.failing_checks  # a loud account of what failed
    assert second.index == 2
    assert second.rebuild_targets == [] and second.failing_checks == ""

    # The converged package is on disk and importable — the reported best is the last round.
    assert (tmp_path / result.spec.name / "nodes.py").exists()


# --- §9.2 — a node-body dialect violation is repaired; render-owned is a loud terminal --


class _DialectViolationOnceSink(ScriptedSink):
    """Authors a node body with a *node-body* dialect violation on its FIRST authoring: a
    literal-bound ``range`` loop, which passes forge's static build gates (contract, purity,
    coverage) yet trips ``check_code``'s L5 ``literal-loop-bound`` inside ``nodes.py``. Every
    re-author is clean — so the loop attributes the violation (signal 1) and repairs it."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._emitted_bad = False

    async def run(self, defn, **kwargs):  # type: ignore[no-untyped-def]
        if defn.name in ("build_draft", "build_repair"):
            valid = _draft_body(str(kwargs.get("context", "")))
            if not self._emitted_bad:
                self._emitted_bad = True
                return PieceOut(body="for _i in range(2):\n    pass\n" + valid)
            return PieceOut(body=valid)
        return await super().run(defn, **kwargs)


async def test_self_correction_repairs_a_node_body_dialect_violation(
    tmp_path: Path,
) -> None:
    sink = _DialectViolationOnceSink(**_reading_list_kwargs())  # type: ignore[arg-type]
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=sink,
        sandbox=FakeSandbox(),  # tests always pass; only the dialect check is red at first
    )

    assert result.works
    assert result.audit is not None and result.audit.rounds == 2
    assert len(result.audit_history) == 2
    # Attributed to the offending node via the nodes.py-scoped violation, then re-authored.
    assert result.audit_history[0].rebuild_targets == ["ParseTheRequest"]
    assert result.audit_history[1].rebuild_targets == []


class _PlantsRenderBugSandbox:
    """Simulates a forge *render* defect: before returning a passing test run it plants a
    render-owned dialect violation — an engine import in ``deps.py`` (a worker file that may
    not import the engine). The audit's dialect check then fails on a file NO node-body
    re-authoring can fix, so attribution yields nothing and the loop settles as a loud
    terminal (§6.1/§12.4) — never a rebuild."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        self.calls += 1
        deps = Path(argv[-1]).parent / "deps.py"
        src = deps.read_text(encoding="utf-8")
        if "pydantic_graph" not in src:
            deps.write_text(
                "import pydantic_graph  # planted render defect\n" + src,
                encoding="utf-8",
            )
        return SandboxResult(exit_code=0, stdout="1 passed in 0.10s\n")


async def test_render_owned_violation_settles_as_a_loud_terminal(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    sandbox = _PlantsRenderBugSandbox()
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=reading_list_sink,
        sandbox=sandbox,
    )

    # A render-owned violation is a forge bug → loud terminal, NOT a rebuild: audited once.
    assert not result.works
    assert sandbox.calls == 1, sandbox.calls
    assert len(result.audit_history) == 1
    assert result.audit_history[0].rebuild_targets == []

    # The violation is surfaced honestly, attributed to the render-owned file (not a node).
    assert result.audit is not None
    dialect = next(c for c in result.audit.checks if c.name == "dialect")
    assert not dialect.passed
    assert any(Path(v.file).name == "deps.py" for v in dialect.violations)


# --- §9.3 — a critic-caught stand-in is repaired (signal 4, reconciled) -----------


class _CriticNamesOnceSink(ScriptedSink):
    """The critic names a node (with a *slightly wrong* name, to exercise rapidfuzz
    reconciliation) on its FIRST audit, then passes clean. The other three signals are
    empty (see the sandbox below), so signal 4 is the only attributor."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._critic_calls = 0

    async def run(self, defn, **kwargs):  # type: ignore[no-untyped-def]
        if defn.name == "critic":
            self._critic_calls += 1
            if self._critic_calls == 1:
                return CriticVerdict(
                    issues=[NodeFinding(node="ParseRequest", issue="hardcoded stand-in")]
                )
            return CriticVerdict(issues=[])
        return await super().run(defn, **kwargs)


class _CrashNoNodesFrameOnceSandbox:
    """First run FAILS with a traceback frame in the TEST file only (never ``nodes.py``), so
    signal 2 finds nothing and the failure is attributable *only* via the critic; every run
    after passes."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        self.calls += 1
        if self.calls > 1:
            return SandboxResult(exit_code=0, stdout="1 passed in 0.10s\n")
        out = (
            "_____ test_smoke _____\n"
            "tests/test_graph.py:9: in test_smoke\n"
            "    assert out is not None\n"
            "E   AssertionError\n"
            "0 passed, 1 failed in 0.10s\n"
        )
        return SandboxResult(exit_code=1, stdout=out)


async def test_self_correction_repairs_a_critic_caught_stand_in(tmp_path: Path) -> None:
    sink = _CriticNamesOnceSink(**_two_spine_kwargs())  # type: ignore[arg-type]
    sandbox = _CrashNoNodesFrameOnceSandbox()
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=sink,
        sandbox=sandbox,
    )

    assert result.works
    assert sandbox.calls == 2, sandbox.calls
    # ONLY the critic-named node was re-authored — the spine fallback would have named BOTH
    # spine nodes — proving signal-4 attribution, reconciled from "ParseRequest" via rapidfuzz.
    assert result.audit_history[0].rebuild_targets == ["ParseTheRequest"]
    assert result.audit_history[1].rebuild_targets == []


# --- §9.4 — a genuinely unfixable failure settles loud, within the cap ------------


class _MonotonicCrashSandbox:
    """Fails every run with an attributable ``nodes.py`` frame but a strictly DECREASING
    failure count (3 → 2 → 1). Each rebuild strictly improves on the count-based order yet
    never reaches green, so the loop runs to the round cap and settles ``works=False``."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        self.calls += 1
        pkg = Path(argv[-1]).parent
        line = _first_class_line((pkg / "nodes.py").read_text())
        failed = {1: 3, 2: 2}.get(self.calls, 1)
        out = (
            f"{pkg.name}/nodes.py:{line}: in run\n"
            "    raise RuntimeError('x')\n"
            "E   RuntimeError: x\n"
            f"0 passed, {failed} failed in 0.10s\n"
        )
        return SandboxResult(exit_code=1, stdout=out)


async def test_unfixable_failure_settles_loud_within_the_cap(tmp_path: Path) -> None:
    sink = ScriptedSink(**_two_spine_kwargs())  # type: ignore[arg-type]
    sandbox = _MonotonicCrashSandbox()
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=sink,
        sandbox=sandbox,
    )

    # Loud, honest result — never a hang or a raise.
    assert not result.works
    # Initial build + max_audit_rounds rebuilds = max_audit_rounds + 1 audit passes.
    assert result.audit is not None
    assert result.audit.rounds == MAX_AUDIT_ROUNDS + 1
    assert sandbox.calls == MAX_AUDIT_ROUNDS + 1
    assert len(result.audit_history) == MAX_AUDIT_ROUNDS + 1
    # The cap stops the rebuilding: the final pass attributes no further targets.
    assert result.audit_history[-1].rebuild_targets == []
    # The BEST build is still carried (capped at inquiry, not at output).
    assert result.build is not None and result.build.all_filled


# --- §9.5 — a regression is caught, and best is re-materialised on disk ------------


class _DegradeSink(ScriptedSink):
    """Marks the ONE spine target's body distinctly per authoring: ``marker_round0`` on the
    first build, ``marker_round1`` on the rebuild — so the on-disk body is identifiable."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._target_calls = 0

    async def run(self, defn, **kwargs):  # type: ignore[no-untyped-def]
        if defn.name in ("build_draft", "build_repair"):
            context = str(kwargs.get("context", ""))
            body = _draft_body(context)
            if _authored_node(context) == "ParseTheRequest":
                self._target_calls += 1
                mark = "marker_round0 = 1" if self._target_calls == 1 else "marker_round1 = 1"
                body = mark + "\n" + body
            return PieceOut(body=body)
        return await super().run(defn, **kwargs)


class _DegradeSandbox:
    """Round 0 is the BETTER audit (2 passed, 1 failed); the rebuild DEGRADES (0 passed, 3
    failed). The count-based total order catches the regression and keeps round 0."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        self.calls += 1
        pkg = Path(argv[-1]).parent
        line = _first_class_line((pkg / "nodes.py").read_text())
        frame = (
            f"{pkg.name}/nodes.py:{line}: in run\n"
            "    raise RuntimeError('x')\n"
            "E   RuntimeError: x\n"
        )
        tail = "2 passed, 1 failed in 0.10s\n" if self.calls == 1 else "0 passed, 3 failed in 0.10s\n"
        return SandboxResult(exit_code=1, stdout=frame + tail)


async def test_regression_keeps_best_and_re_materialises_it_on_disk(
    tmp_path: Path,
) -> None:
    sink = _DegradeSink(**_two_spine_kwargs())  # type: ignore[arg-type]
    sandbox = _DegradeSandbox()
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=sink,
        sandbox=sandbox,
    )

    assert not result.works
    # One rebuild, then the regression guard (the round-1 audit did not improve) stops.
    assert sandbox.calls == 2, sandbox.calls
    assert len(result.audit_history) == 2

    # The kept best is round 0 (the better audit), not the worse round-1 rebuild.
    assert result.build is not None
    assert any("marker_round0" in f.body for f in result.build.filled)
    assert all("marker_round1" not in f.body for f in result.build.filled)

    # §4.6: dest/ is re-materialised to best — the worse round-1 body is NOT what sits on disk.
    disk = (tmp_path / result.spec.name / "nodes.py").read_text()
    assert "marker_round0" in disk
    assert "marker_round1" not in disk


# --- §9.6 — the loop fires rarely: clean build no-fire; rebuild no duplicate rows --


class _CountingPassSandbox:
    """Always passes; counts its runs so a test can prove the loop never fired."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        self.calls += 1
        return SandboxResult(exit_code=0, stdout="1 passed in 0.10s\n")


async def test_clean_first_build_does_not_fire_the_loop(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    sandbox = _CountingPassSandbox()
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=reading_list_sink,
        sandbox=sandbox,
    )

    assert result.works
    # A single audit pass — the loop never fired (audit_rounds stayed 0 ⇒ rounds == 1).
    assert sandbox.calls == 1, sandbox.calls
    assert result.audit is not None and result.audit.rounds == 1
    assert len(result.audit_history) == 1
    assert result.audit_history[0].rebuild_targets == []


async def test_rebuild_does_not_duplicate_reporter_audit_rows(tmp_path: Path) -> None:
    from rich.console import Console

    rep = RichReporter(
        Console(file=io.StringIO(), force_terminal=True, width=80),
        brief="Manage my reading list",
        model="test",
        dest=str(tmp_path),
    )
    sink = ScriptedSink(**_reading_list_kwargs())  # type: ignore[arg-type]
    sandbox = _CrashOnceSandbox()  # one rebuild → two audit passes
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=sink,
        sandbox=sandbox,
        reporter=rep,
    )

    assert result.works
    assert sandbox.calls == 2
    # Two audit passes ran, but the reporter shows only the LAST pass's four checks — a
    # rebuild replaces the audit rows, it does not stack them.
    assert rep._audit.results == [
        ("tests", "passed"),
        ("dialect", "passed"),
        ("requirements", "passed"),
        ("critic", "passed"),
    ]


# --- the loop caps are named constants (§8.3/§10.6 — caps live here, not in topology) --


def test_self_correction_caps_are_named_constants() -> None:
    # The two sequential, independently-capped cycles: Review→Frame (redesign) and
    # Audit→Render (rebuild). Both bounds are module constants, both wired into ForgeDeps.
    assert MAX_AUDIT_ROUNDS == 2
    assert MAX_PLAN_REVIEW_ROUNDS == 2
    assert ForgeDeps.__dataclass_fields__["max_audit_rounds"].default == MAX_AUDIT_ROUNDS
