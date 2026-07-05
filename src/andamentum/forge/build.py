"""Stage 3 — the per-node builder: agents author every node body, statically gated.

``build_system(spec, pkg_dir, ...)`` fills every ``NotImplementedError`` hole in the
rendered package, one node at a time, against that node's typed contract:

  draft  [agent]   — write the body from the exact contract + the dialect laws
  compile [det]    — py_compile (catches syntax)
  contract [det]   — reads/writes/returns must be declared (catches hallucinated names)
  purity  [det]    — no process control / raw IO / clock / random; network only if declared
  repair  [agent]  — on any gate failure, fed the exact violated obligation
  cap              — restore the NotImplementedError, mark unfillable (honest failure)

Every gate is **in-process and static** — no LLM-written code executes here. Execution
happens later, behind the sandbox, in the audit (stage 4). Bounded by ``attempt_cap``
(draft + N repairs). Engine-free leaf worker (dialect Law 2): it calls agents through
the ``AgentSink`` Port and rewrites files, but imports no graph engine.
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

from .agents import DRAFT, MODEL_OUTPUT_ERRORS, REPAIR, AgentSink
from .astcheck import (
    check_deps_access,
    check_fail_loud,
    check_node_body,
    check_purity,
)
from .contract import Hole, NodeContract, node_contract
from .extract import discover_holes
from .patch import apply_body
from .reporter import ForgeReporter, NoopReporter
from .schemas import (
    BuildReport,
    FilledNode,
    PieceOut,
    UnfillableNode,
)
from .spec import NodeSpec, SystemSpec


def _fields(label: str, items: list[tuple[str, str]]) -> list[str]:
    if not items:
        return []
    return [label] + [f"  ctx.state.{name}: {ann}" for name, ann in items]


def _draft_context(node: NodeSpec, contract: NodeContract, hole: Hole) -> str:
    reads = [(f.name, f.annotation) for f in contract.reads]
    writes = [(f.name, f.annotation) for f in contract.writes]
    lines = [
        f"NODE: {node.name}  ({hole.kind.value})",
        f"SIGNATURE: {hole.signature.strip()}",
        f"JOB: {node.job or node.purpose or 'implement this node'}",
    ]
    if hole.context.strip():
        lines += [
            "",
            "PREAMBLE — keep these lines first, unchanged:",
            hole.context.strip(),
        ]
    lines += _fields(
        "YOU MAY READ these state fields (and nothing else on ctx.state):", reads
    )
    lines += _fields("YOU MUST SET these state fields:", writes)
    if contract.agent_output is not None:
        lines.append(
            f"The head result `out` ({contract.agent_output.name}) has fields:"
        )
        lines += [
            f"  out.{f.name}: {f.annotation}" for f in contract.agent_output.fields
        ]
    if node.network:
        lines.append(
            "NETWORK NODE: you MAY import and use an HTTP client (httpx or requests) to reach the endpoint this "
            "node needs, and parse the response. Still forbidden: os, subprocess, socket, open(), eval, clock/random. "
            "Handle an empty or error response without crashing."
        )
    succ = ", ".join(contract.successors)
    lines.append(
        f"RETURN exactly one of these successors: {succ} — e.g. `return X()`; for End use `return End(<str>)`."
    )
    return "\n".join(lines)


def _repair_context(
    node: NodeSpec,
    contract: NodeContract,
    hole: Hole,
    body: str,
    error: str,
) -> str:
    return "\n".join(
        [
            _draft_context(node, contract, hole),
            "",
            "YOUR PREVIOUS BODY (rejected):",
            body.strip(),
            "",
            f"WHY IT WAS REJECTED: {error.strip()}",
            "Fix exactly that and return the corrected body.",
        ]
    )


def _compiles(file: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(file), doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e.msg or e)


async def _body_from(sink: AgentSink, defn, context: str) -> str:
    out = await sink.run(defn, context=context)
    assert isinstance(out, PieceOut)
    return out.body


def _declared_deps(pkg_dir: Path) -> set[str]:
    """The dependency attribute names the rendered ``Deps`` actually provides — the single
    source of truth the deps gate checks bodies against (so gate and renderer cannot drift).
    Reads the annotated fields off the generated ``deps.py`` dataclass."""
    deps_file = pkg_dir / "deps.py"
    if not deps_file.exists():
        return set()
    tree = ast.parse(deps_file.read_text())
    names: set[str] = set()
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    names.add(stmt.target.id)
    return names


async def build_system(
    spec: SystemSpec,
    pkg_dir: Path,
    *,
    sink: AgentSink,
    attempt_cap: int,
    reporter: ForgeReporter | None = None,
) -> BuildReport:
    """Fill every node hole in the rendered package under ``pkg_dir`` against the spec.

    Every hole is authored by the draft/repair agents, statically gated (contract, purity,
    deps, fail-loud). Bounded by ``attempt_cap`` (draft + N repairs).
    """
    rep: ForgeReporter = reporter if reporter is not None else NoopReporter()
    by_name: dict[str, NodeSpec] = {n.name: n for n in spec.nodes}
    allowed_deps = _declared_deps(pkg_dir)
    holes = list(discover_holes(pkg_dir))
    rep.build_starting(total=len(holes))
    filled: list[FilledNode] = []
    unfillable: list[UnfillableNode] = []
    for index, hole in enumerate(holes, start=1):
        node = by_name.get(hole.node)
        if node is None:
            unfillable.append(
                UnfillableNode(
                    node=hole.node,
                    last_error="no matching NodeSpec — spec/package out of sync",
                    attempts=0,
                )
            )
            rep.node_built(
                node=hole.node, status="unfillable", attempts=0, detail="no NodeSpec"
            )
            continue
        result = await _build_one(
            spec,
            node,
            hole,
            sink=sink,
            attempt_cap=attempt_cap,
            reporter=rep,
            index=index,
            total=len(holes),
            allowed_deps=allowed_deps,
        )
        if isinstance(result, FilledNode):
            filled.append(result)
            status = "filled"
        else:
            unfillable.append(result)
            status = "unfillable"
        rep.node_built(
            node=node.name, status=status, attempts=result.attempts, detail=""
        )
    return BuildReport(filled=filled, unfillable=unfillable)


async def _build_one(
    spec: SystemSpec,
    node: NodeSpec,
    hole: Hole,
    *,
    sink: AgentSink,
    attempt_cap: int,
    reporter: ForgeReporter,
    index: int,
    total: int,
    allowed_deps: set[str],
) -> FilledNode | UnfillableNode:
    file = hole.file
    assert file is not None  # discovered holes always carry their file
    original = file.read_text()
    contract = node_contract(spec, node.name)
    reads = {f.name for f in contract.reads}
    writes = {f.name for f in contract.writes}
    successors = set(contract.successors)

    last_error = "(no attempt made)"
    body = ""
    for attempt in range(1, attempt_cap + 1):
        reporter.node_building(
            node=node.name,
            kind=node.kind.value,
            index=index,
            total=total,
            attempt=attempt,
            phase="draft" if attempt == 1 else "repair",
        )
        try:
            if attempt == 1:
                body = await _body_from(
                    sink, DRAFT, _draft_context(node, contract, hole)
                )
            else:
                body = await _body_from(
                    sink,
                    REPAIR,
                    _repair_context(node, contract, hole, body, last_error),
                )
        except MODEL_OUTPUT_ERRORS as e:
            # The model could not produce a valid body (e.g. a small model returning the
            # schema envelope). Treat it as a failed attempt — the loop retries, and at
            # budget exhaustion the node settles to UnfillableNode. A single node's
            # authoring failure never crashes the whole build (fail loud, not fatal).
            last_error = f"the model failed to produce a valid body: {e}"
            continue

        # Always patch from the pristine original so the source always parses.
        file.write_text(original)
        try:
            apply_body(file, hole.node, hole.method, body)
        except ValueError as e:
            last_error = f"patch failed: {e}"
            continue

        ok, err = _compiles(file)
        if not ok:
            last_error = f"compile error: {err}"
            continue

        violations = check_node_body(
            file,
            hole.node,
            hole.method,
            reads=reads,
            writes=writes,
            successors=successors,
        )
        violations += check_purity(
            file, hole.node, hole.method, allow_network=node.network
        )
        violations += check_deps_access(
            file, hole.node, hole.method, allowed=allowed_deps
        )
        violations += check_fail_loud(file, hole.node, hole.method)
        if violations:
            last_error = "; ".join(violations)
            continue

        # Static gates passed — the deterministic gates decide fillability, so the body
        # is accepted and the node is filled.
        return FilledNode(node=node.name, attempts=attempt, body=body)

    # Never produced a gate-valid body → honest unfillable; restore the hole.
    file.write_text(original)
    return UnfillableNode(node=node.name, last_error=last_error, attempts=attempt_cap)
