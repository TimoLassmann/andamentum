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

from .agents import COMPONENT_MANAGER, DRAFT, MODEL_OUTPUT_ERRORS, REPAIR, AgentSink
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
    BodyVerdict,
    BuildConcern,
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


def _audit_feedback_note(text: str) -> str:
    """Rebuild-only addendum: this node is being re-authored after the assembled system
    failed its audit even though the previous body passed every static gate."""
    return "\n".join(
        [
            "",
            "THIS NODE IS BEING RE-AUTHORED. The previous body passed every static gate, but the "
            "assembled system then FAILED its audit:",
            text.strip(),
            "Address that failure in your implementation.",
        ]
    )


def _repair_context(
    node: NodeSpec,
    contract: NodeContract,
    hole: Hole,
    body: str,
    error: str,
    audit_feedback: str | None = None,
) -> str:
    lines = [
        _draft_context(node, contract, hole),
        "",
        "YOUR PREVIOUS BODY (rejected):",
        body.strip(),
        "",
        f"WHY IT WAS REJECTED: {error.strip()}",
        "Fix exactly that and return the corrected body.",
    ]
    if audit_feedback and audit_feedback.strip():
        lines.append(_audit_feedback_note(audit_feedback))
    return "\n".join(lines)


def _manager_context(
    node: NodeSpec, contract: NodeContract, hole: Hole, body: str
) -> str:
    return "\n".join(
        [
            _draft_context(node, contract, hole),
            "",
            "AUTHORED BODY (passed every static gate — judge only whether it does the job):",
            body.strip(),
        ]
    )


async def _manager_verdict(sink: AgentSink, context: str) -> BodyVerdict:
    out = await sink.run(COMPONENT_MANAGER, context=context)
    assert isinstance(out, BodyVerdict)
    return out


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
    prior_bodies: dict[str, str] | None = None,
    targets: set[str] | None = None,
    rebuild_feedback: dict[str, str] | None = None,
) -> BuildReport:
    """Fill every node hole in the rendered package under ``pkg_dir`` against the spec.

    Tri-state on ``targets`` (the rebuild contract, §4.3):
      - ``None``       — author EVERY hole (the first build; byte-identical to before).
      - a set          — author ONLY those holes; re-apply every OTHER hole verbatim from
                         ``prior_bodies`` via ``apply_body`` (no sink / LLM call).
      - empty set      — author NOTHING; re-apply ALL holes from ``prior_bodies``.

    ``prior_bodies`` maps a node name to its known-good source (``FilledNode.body`` from an
    earlier build). ``rebuild_feedback`` maps a re-authored target's name to the audit-failure
    text, threaded into that target's repair context so it sees why it failed. Defaults keep the
    first-build path unchanged.
    """
    rep: ForgeReporter = reporter if reporter is not None else NoopReporter()
    bodies: dict[str, str] = prior_bodies or {}
    feedback: dict[str, str] = rebuild_feedback or {}
    by_name: dict[str, NodeSpec] = {n.name: n for n in spec.nodes}
    allowed_deps = _declared_deps(pkg_dir)
    holes = list(discover_holes(pkg_dir))
    rep.build_starting(total=len(holes))
    filled: list[FilledNode] = []
    unfillable: list[UnfillableNode] = []
    concerns: list[BuildConcern] = []
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
        # Rebuild: a non-target hole re-applies its known-good body verbatim, no LLM call.
        if targets is not None and hole.node not in targets:
            prior = bodies.get(hole.node)
            if prior is None:
                unfillable.append(
                    UnfillableNode(
                        node=hole.node,
                        last_error="no prior body to re-apply (non-target with no known-good source)",
                        attempts=0,
                    )
                )
                rep.node_built(
                    node=hole.node,
                    status="unfillable",
                    attempts=0,
                    detail="no prior body",
                )
                continue
            file = hole.file
            assert file is not None  # discovered holes always carry their file
            apply_body(file, hole.node, hole.method, prior)
            filled.append(FilledNode(node=hole.node, attempts=0, body=prior))
            rep.node_built(
                node=hole.node, status="filled", attempts=0, detail="reapplied"
            )
            continue
        result, concern = await _build_one(
            spec,
            node,
            hole,
            sink=sink,
            attempt_cap=attempt_cap,
            reporter=rep,
            index=index,
            total=len(holes),
            allowed_deps=allowed_deps,
            audit_feedback=feedback.get(node.name),
        )
        if isinstance(result, FilledNode):
            filled.append(result)
            status = "kept" if concern is not None else "filled"
        else:
            unfillable.append(result)
            status = "unfillable"
        rep.node_built(
            node=node.name,
            status=status,
            attempts=result.attempts,
            detail=concern.issue if concern is not None else "",
        )
        if concern is not None:
            concerns.append(concern)
    return BuildReport(filled=filled, unfillable=unfillable, concerns=concerns)


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
    audit_feedback: str | None = None,
) -> tuple[FilledNode | UnfillableNode, BuildConcern | None]:
    file = hole.file
    assert file is not None  # discovered holes always carry their file
    original = file.read_text()
    contract = node_contract(spec, node.name)
    reads = {f.name for f in contract.reads}
    writes = {f.name for f in contract.writes}
    successors = set(contract.successors)

    last_error = "(no attempt made)"
    body = ""
    last_valid_body: str | None = None
    last_valid_attempt = 0
    last_concern = ""
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
                draft_context = _draft_context(node, contract, hole)
                if audit_feedback and audit_feedback.strip():
                    draft_context += _audit_feedback_note(audit_feedback)
                body = await _body_from(sink, DRAFT, draft_context)
            else:
                body = await _body_from(
                    sink,
                    REPAIR,
                    _repair_context(
                        node, contract, hole, body, last_error, audit_feedback
                    ),
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

        # Static gates passed — remember this as a keepable, contract-valid body.
        last_valid_body = body
        last_valid_attempt = attempt

        # The component manager judges whether the body does the JOB. On objection, feed
        # the issue into THIS draft/repair loop (bounded by attempt_cap). It is ADVISORY —
        # if the model call itself fails, skip it (the gate-valid body stands); an advisory
        # head must never block a body the deterministic gates already accepted.
        try:
            verdict = await _manager_verdict(
                sink, _manager_context(node, contract, hole, body)
            )
        except MODEL_OUTPUT_ERRORS:
            verdict = None
        if verdict is not None and not verdict.implements_job and verdict.issue.strip():
            last_error = verdict.issue.strip()
            last_concern = verdict.issue.strip()
            continue

        return FilledNode(node=node.name, attempts=attempt, body=body), None

    # Budget spent. If we ever produced a gate-valid body, KEEP it (deterministic gates
    # decide fillability; the manager only tries to improve). Record the residual concern.
    if last_valid_body is not None:
        file.write_text(original)
        apply_body(file, hole.node, hole.method, last_valid_body)
        concern = (
            BuildConcern(node=node.name, issue=last_concern) if last_concern else None
        )
        return (
            FilledNode(
                node=node.name, attempts=last_valid_attempt, body=last_valid_body
            ),
            concern,
        )

    # Never produced a gate-valid body → honest unfillable; restore the hole.
    file.write_text(original)
    return (
        UnfillableNode(node=node.name, last_error=last_error, attempts=attempt_cap),
        None,
    )
