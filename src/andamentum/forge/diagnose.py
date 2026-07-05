"""Worker: diagnose a declared node board for structural problems (the rich engine).

Pure and engine-free — stdlib + ``rapidfuzz`` + the sibling schemas only. Given the
node board and the :class:`DataGraph` :mod:`assemble` matched from it, it gathers EVERY
structural problem with a concrete, agent-readable suggested fix. The determinism does
the heavy lifting (finding + suggesting); the repair agent only applies targeted
corrections, which is what makes the loop converge on small local models.

The full catalogue (one :class:`FindingKind` each):

  - ``dangling_read``    a consumed name no node produces and not an input token, with no
                         near-miss producer — surface it plainly.
  - ``near_miss``        a consumed name no node produces, but a PRODUCED name is highly
                         similar — almost certainly the same variable misspelled. Reported
                         INSTEAD of ``dangling_read`` for that read (de-dupe: one problem,
                         one finding) with the rename as the suggestion.
  - ``orphan_output``    a produced name nobody consumes and not the single system output.
  - ``duplicate_producer``  a name produced by more than one node.
  - ``multiple_sinks``   more than one terminal output node (the extras overlap with
                         ``orphan_output`` — reported here once, not also as orphans).
  - ``no_output``        no terminal output node at all.
  - ``unreachable``      a node not reachable from the input via produce→consume edges.
  - ``dead_end``         a node whose work reaches no sink/output.
  - ``disconnected``     a component with no path to the input→output flow (reported once,
                         per node, beyond plain unreachable/dead-end).
  - ``unintended_cycle`` a cycle among nodes none of which is a bounded-loop checkpoint.
                         A checkpoint back-edge is legitimate and is NOT a finding.
  - ``each_needs_collection``  an 'each' (map-over-items) node whose inputs are not exactly
                         ONE collection datum. Collection-ness is COMPUTED (input flag +
                         each-node produces, via :func:`assemble.collection_data`), never
                         declared per-datum. A WHOLE node consuming a collection is fine —
                         that is the reduce/synthesis case.

All deterministic and pure; thresholds are named module constants.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from .assemble import DataGraph, collection_data
from .schemas import DataKind, DesignFinding, DesignReport, FindingKind, NodeDraft
from .spec import NodeControl, NodeMode

# rapidfuzz `partial_ratio` (0–100) above which two variable names are "probably the
# same". `partial_ratio` (not plain `ratio`) catches the common near-miss where one name
# is a prefix/substring of the other — e.g. `bullets` vs `bullet_statements` (92), or
# `summary` vs `brief_summary` (100) — while unrelated pairs stay well below (e.g.
# `widget` vs `answer` = 44).
_NEAR_MISS_THRESHOLD = 80.0


def _nearest(name: str, candidates: list[str]) -> tuple[str, float]:
    """Return the highest-similarity candidate name and its score (0 if none)."""
    best_name = ""
    best_score = 0.0
    for candidate in candidates:
        score = fuzz.partial_ratio(name, candidate)
        if score > best_score:
            best_score = score
            best_name = candidate
    return best_name, best_score


def diagnose(
    nodes: list[NodeDraft],
    graph: DataGraph,
    *,
    input_is_collection: bool = False,
) -> DesignReport:
    """Inspect the assembled DAG and gather every structural problem with a fix.

    ``input_is_collection`` is the understand head's flag (``ForgeWhy``): it seeds the
    computed collection-ness of the graph input for the ``each_needs_collection`` check.
    """
    findings: list[DesignFinding] = []
    produced_names = sorted(graph.writers)
    consumed_names = sorted(graph.readers)

    findings.extend(_dangling_and_near_miss(nodes, graph, produced_names))
    findings.extend(_duplicate_producers(graph))
    sink_findings, system_output = _sinks(nodes, graph, consumed_names)
    findings.extend(sink_findings)
    findings.extend(_orphans(nodes, graph, consumed_names))
    findings.extend(_reachability(nodes, graph, system_output))
    findings.extend(_cycles(nodes, graph))
    findings.extend(
        _each_streams(
            nodes, collection_data(nodes, input_is_collection=input_is_collection)
        )
    )

    return DesignReport(findings=findings)


def _dangling_and_near_miss(
    nodes: list[NodeDraft], graph: DataGraph, produced_names: list[str]
) -> list[DesignFinding]:
    """A consumed name no node produces: a near-miss rename if a producer is close, else dangling."""
    findings: list[DesignFinding] = []
    for node in nodes:
        for name in node.consumes:
            if name in graph.inputs or name in graph.writers:
                continue  # matched to an input token or a real producer
            nearest, score = _nearest(name, produced_names)
            if nearest and score >= _NEAR_MISS_THRESHOLD:
                findings.append(
                    DesignFinding(
                        kind=FindingKind.NEAR_MISS,
                        node=node.id,
                        variable=name,
                        detail=(
                            f"reads {name!r}, which no step produces; the produced name "
                            f"{nearest!r} is {score:.0f}% similar — probably the same variable."
                        ),
                        suggestion=(
                            f"rename {name!r} to {nearest!r} (or rename the producer) so the "
                            "two refer to one variable"
                        ),
                    )
                )
            else:
                findings.append(
                    DesignFinding(
                        kind=FindingKind.DANGLING_READ,
                        node=node.id,
                        variable=name,
                        detail=f"reads {name!r}, which no step produces and is not the graph input",
                        suggestion="have an upstream step produce it, or read an existing variable",
                    )
                )
    return findings


def _duplicate_producers(graph: DataGraph) -> list[DesignFinding]:
    """A variable produced by more than one node — only one writer is allowed."""
    findings: list[DesignFinding] = []
    for name in sorted(graph.writers):
        producers = graph.writers[name]
        if len(producers) > 1:
            findings.append(
                DesignFinding(
                    kind=FindingKind.DUPLICATE_PRODUCER,
                    variable=name,
                    detail=f"{name!r} is produced by {len(producers)} nodes ({', '.join(producers)})",
                    suggestion=(
                        "give each step a distinct produced name; if a later step refines the "
                        "value, have it READ the original and produce a new name"
                    ),
                )
            )
    return findings


def _signal_produces(node: NodeDraft) -> list[str]:
    """The signal (run-scoped) names a node produces — entities are not signal terminals."""
    return list(node.produces) if node.produces_kind is DataKind.SIGNAL else []


def _sinks(
    nodes: list[NodeDraft], graph: DataGraph, consumed_names: list[str]
) -> tuple[list[DesignFinding], str]:
    """Find the terminal signal(s) (produced, read by nobody).

    The system output is the signal the last board node produces (matching the compile
    backstop's topologically-last rule). Returns the findings and that output variable
    (``""`` when it cannot be determined). 0 signal terminals → ``no_output``; ≥2 →
    ``multiple_sinks`` (reported once; the extras are not also reported as orphans).
    """
    consumed = set(consumed_names)
    terminals = sorted(
        name
        for node in nodes
        for name in _signal_produces(node)
        if name not in consumed
    )
    if not terminals:
        return [
            DesignFinding(
                kind=FindingKind.NO_OUTPUT,
                detail="no step produces a final signal output — every produced signal is read by another step",
                suggestion="have one terminal step produce the system's answer and read nothing further",
            )
        ], ""

    last = nodes[-1]
    system_output = next(
        (name for name in _signal_produces(last) if name not in consumed), ""
    )
    if len(terminals) == 1:
        return [], terminals[0]
    return [
        DesignFinding(
            kind=FindingKind.MULTIPLE_SINKS,
            detail=(
                f"{len(terminals)} steps produce a final signal ({', '.join(terminals)}); "
                "exactly one is the system output"
            ),
            suggestion=(
                "chain the extra signals forward (have a later step read them) or merge them, "
                "so a single variable is the system output"
            ),
        )
    ], system_output


def _orphans(
    nodes: list[NodeDraft], graph: DataGraph, consumed_names: list[str]
) -> list[DesignFinding]:
    """A produced ENTITY no step reads — dead stored work.

    Signal terminals are handled by :func:`_sinks` (one is the system output, extras are
    ``multiple_sinks``); an entity is never a signal terminal, so an unread produced
    entity is a distinct orphan — work written to a store that nothing ever consumes.
    """
    consumed = set(consumed_names)
    findings: list[DesignFinding] = []
    for node in nodes:
        if node.produces_kind is not DataKind.ENTITY:
            continue
        for name in node.produces:
            if name in consumed:
                continue
            nearest, score = _nearest(name, consumed_names)
            if nearest and score >= _NEAR_MISS_THRESHOLD:
                suggestion = (
                    f"a later step reads {nearest!r} ({score:.0f}% similar) — unify the names so "
                    "this entity feeds it"
                )
            else:
                suggestion = "have a later step read it, or remove this step"
            findings.append(
                DesignFinding(
                    kind=FindingKind.ORPHAN_OUTPUT,
                    node=node.id,
                    variable=name,
                    detail=f"produces entity {name!r}, which no step reads",
                    suggestion=suggestion,
                )
            )
    return findings


def _reachability(
    nodes: list[NodeDraft], graph: DataGraph, system_output: str
) -> list[DesignFinding]:
    """Reachable-from-input, reaches-an-output, and fully-disconnected checks."""
    if not system_output:
        return []  # an undefined output makes downstream reachability meaningless

    succ: dict[str, list[str]] = {node.id: [] for node in nodes}
    pred: dict[str, list[str]] = {node.id: [] for node in nodes}
    for producer, consumer in graph.edges:
        succ[producer].append(consumer)
        pred[consumer].append(producer)

    # Sources: nodes that read an input token. Sinks: nodes that produce the system output.
    sources = [n.id for n in nodes if any(c in graph.inputs for c in n.consumes)]
    sinks = [n.id for n in nodes if system_output in n.produces]

    reachable = _walk(sources, succ)
    reaches_output = _walk(sinks, pred)

    findings: list[DesignFinding] = []
    for node in nodes:
        in_flow = node.id in reachable and node.id in reaches_output
        if in_flow:
            continue
        if node.id not in reachable and node.id not in reaches_output:
            findings.append(
                DesignFinding(
                    kind=FindingKind.DISCONNECTED,
                    node=node.id,
                    detail=(
                        f"node {node.id} is on no path between the input and the system output — "
                        "its work neither follows from the input nor reaches the output"
                    ),
                    suggestion="wire it into the flow (read an upstream value and feed a downstream one) or remove it",
                )
            )
        elif node.id not in reachable:
            findings.append(
                DesignFinding(
                    kind=FindingKind.UNREACHABLE,
                    node=node.id,
                    detail=f"node {node.id} is not reachable from the input",
                    suggestion="have it read a value an upstream step produces (transitively from the input)",
                )
            )
        else:
            findings.append(
                DesignFinding(
                    kind=FindingKind.DEAD_END,
                    node=node.id,
                    detail=f"node {node.id} reaches no output — its work goes nowhere",
                    suggestion="have a downstream step read what it produces, leading to the system output",
                )
            )
    return findings


def _walk(starts: list[str], adjacency: dict[str, list[str]]) -> set[str]:
    """Deterministic depth-first reachable set from ``starts`` over ``adjacency``."""
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return seen


def _cycles(nodes: list[NodeDraft], graph: DataGraph) -> list[DesignFinding]:
    """A cycle whose nodes include no bounded-loop checkpoint is unintended."""
    succ: dict[str, list[str]] = {node.id: [] for node in nodes}
    for producer, consumer in graph.edges:
        succ[producer].append(consumer)
    is_checkpoint = {node.id: node.control is NodeControl.CHECKPOINT for node in nodes}
    # A node that read-modify-writes an ENTITY is a legitimate round-trip, not a loop: the
    # durable value is loaded at the start of the run and saved at the end (dialect L1), so
    # the same step reading and rewriting it is read-modify-write, not data flowing forever.
    # Its producer→consumer self-edge would otherwise read as a cycle — exempt it.
    rmw_entity = {
        node.id
        for node in nodes
        if node.produces_kind is DataKind.ENTITY
        and set(node.consumes) & set(node.produces)
    }

    findings: list[DesignFinding] = []
    reported: set[frozenset[str]] = set()
    for cycle in _find_cycles(nodes, succ):
        if any(is_checkpoint.get(member, False) for member in cycle):
            continue  # a checkpoint back-edge is a legitimate bounded loop
        if set(cycle) <= rmw_entity:
            continue  # an entity read-modify-write round-trip, not an unintended loop
        key = frozenset(cycle)
        if key in reported:
            continue
        reported.add(key)
        findings.append(
            DesignFinding(
                kind=FindingKind.UNINTENDED_CYCLE,
                detail=(
                    f"steps {', '.join(cycle)} form a cycle with no bounded-loop checkpoint — "
                    "the data flow loops forever"
                ),
                suggestion=(
                    "break the back edge, or mark the step that closes the loop as a checkpoint "
                    "(a bounded loop) if the repetition is intended"
                ),
            )
        )
    return findings


def _each_streams(nodes: list[NodeDraft], collections: set[str]) -> list[DesignFinding]:
    """An 'each' node must consume EXACTLY ONE collection datum (its stream) and nothing
    else (the v1 map shape; scalar context can come later). A WHOLE node consuming a
    collection is fine — that is the reduce/synthesis case, never a finding.

    The repair loop can only re-run the SELECT pass, so the suggestion names a concrete
    collection to read where one exists; when none exists at all, the mode itself is
    wrong and the suggestion says to make the step 'whole' (the loop then fails loud at
    the cap if the wiring cannot converge — surfaced, never silently degraded).
    """
    findings: list[DesignFinding] = []
    for index, node in enumerate(nodes):
        if node.mode is not NodeMode.EACH:
            continue
        streams = [name for name in node.consumes if name in collections]
        scalars = [name for name in node.consumes if name not in collections]
        if len(streams) == 1 and not scalars:
            continue
        # The concrete fix: the nearest available collection — the graph input (when it
        # is a list) or the latest EARLIER each-node's produced list.
        earlier = [
            name
            for other in nodes[:index]
            if other.mode is NodeMode.EACH
            for name in other.produces
        ]
        candidates = (["input"] if "input" in collections else []) + earlier
        if candidates:
            suggestion = (
                f"have it read exactly one collection ({candidates[-1]!r} is available) "
                "and nothing else — or mark the step 'whole' if it sees the data all at once"
            )
        else:
            suggestion = (
                "no collection exists for it to map over — mark the step 'whole' "
                "(it sees the data all at once)"
            )
        findings.append(
            DesignFinding(
                kind=FindingKind.EACH_NEEDS_COLLECTION,
                node=node.id,
                variable=scalars[0] if scalars else "",
                detail=(
                    f"runs once PER ITEM ('each') but reads {node.consumes or ['nothing']}; "
                    f"an 'each' step must read exactly ONE collection and nothing else "
                    f"(collections here: {sorted(streams) or 'none'}, "
                    f"scalars here: {sorted(scalars) or 'none'})"
                ),
                suggestion=suggestion,
            )
        )
    return findings


def _find_cycles(nodes: list[NodeDraft], succ: dict[str, list[str]]) -> list[list[str]]:
    """All elementary cycles, found deterministically via DFS over a fixed node order.

    Returns each cycle as the node ids in the order first encountered along the path.
    """
    cycles: list[list[str]] = []
    seen_keys: set[frozenset[str]] = set()
    order = [node.id for node in nodes]

    def visit(start: str, current: str, path: list[str], on_path: set[str]) -> None:
        for nxt in succ.get(current, ()):
            if nxt == start and len(path) >= 1:
                key = frozenset(path)
                if key not in seen_keys:
                    seen_keys.add(key)
                    cycles.append(list(path))
            elif nxt not in on_path and order.index(nxt) > order.index(start):
                on_path.add(nxt)
                path.append(nxt)
                visit(start, nxt, path, on_path)
                path.pop()
                on_path.discard(nxt)

    for node_id in order:
        visit(node_id, node_id, [node_id], {node_id})
    return cycles


__all__ = ["diagnose"]
