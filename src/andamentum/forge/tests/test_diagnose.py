"""One unit test per ``FindingKind`` for the engine-free ``diagnose`` worker — no model.

Each test builds a tiny ``NodeDraft`` board exhibiting exactly one flaw, runs
``assemble`` + ``diagnose``, and asserts the right finding (and, where the plan calls
for it, the concrete suggestion — e.g. the ``bullets`` / ``bullet_statements`` rename).
A clean board produces no findings; the per-flaw boards isolate their kind.
"""

from __future__ import annotations

from andamentum.forge.assemble import assemble
from andamentum.forge.diagnose import diagnose
from andamentum.forge.schemas import DataKind, FindingKind, NodeDraft
from andamentum.forge.spec import NodeControl


def _node(
    node_id: str,
    consumes: list[str],
    produces: list[str],
    *,
    produces_kind: DataKind = DataKind.SIGNAL,
    control: NodeControl = NodeControl.NONE,
) -> NodeDraft:
    return NodeDraft(
        id=node_id,
        consumes=consumes,
        produces=produces,
        produces_kind=produces_kind,
        control=control,
    )


def _kinds(report) -> set[FindingKind]:
    return {f.kind for f in report.findings}


def _run(nodes: list[NodeDraft]):
    return diagnose(nodes, assemble(nodes))


def test_clean_board_has_no_findings() -> None:
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed"], ["answer"]),
    ]
    report = _run(nodes)
    assert report.clean
    assert report.findings == []


def test_dangling_read() -> None:
    # n2 reads 'sources' — no producer, and nothing close to it.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["sources"], ["answer"]),
    ]
    report = _run(nodes)
    dangling = [f for f in report.findings if f.kind is FindingKind.DANGLING_READ]
    assert len(dangling) == 1
    assert dangling[0].variable == "sources"
    assert dangling[0].node == "n2"
    # No near producer, so it is a plain dangling read, not a near-miss.
    assert FindingKind.NEAR_MISS not in _kinds(report)


def test_near_miss() -> None:
    # n2 reads 'bullets'; n1 produces 'bullet_statements' — the plan's named example.
    nodes = [
        _node("n1", ["input"], ["bullet_statements"]),
        _node("n2", ["bullets"], ["answer"]),
    ]
    report = _run(nodes)
    near = [f for f in report.findings if f.kind is FindingKind.NEAR_MISS]
    assert len(near) == 1
    assert near[0].variable == "bullets"
    assert "bullet_statements" in near[0].suggestion
    # De-duped: a near-miss read is NOT also reported as a plain dangling read.
    assert FindingKind.DANGLING_READ not in _kinds(report)


def test_orphan_output() -> None:
    # n2 writes an ENTITY 'audit_log' nobody reads; n3 carries the real signal output.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed"], ["audit_log"], produces_kind=DataKind.ENTITY),
        _node("n3", ["parsed"], ["answer"]),
    ]
    report = _run(nodes)
    orphans = [f for f in report.findings if f.kind is FindingKind.ORPHAN_OUTPUT]
    assert len(orphans) == 1
    assert orphans[0].variable == "audit_log"
    assert orphans[0].node == "n2"


def test_duplicate_producer() -> None:
    # Both n1 and n2 produce 'parsed'.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["input"], ["parsed"]),
        _node("n3", ["parsed"], ["answer"]),
    ]
    report = _run(nodes)
    dups = [f for f in report.findings if f.kind is FindingKind.DUPLICATE_PRODUCER]
    assert len(dups) == 1
    assert dups[0].variable == "parsed"
    assert "n1" in dups[0].detail and "n2" in dups[0].detail


def test_multiple_sinks() -> None:
    # Two unconsumed signal terminals: 'answer' and 'report'.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed"], ["answer"]),
        _node("n3", ["parsed"], ["report"]),
    ]
    report = _run(nodes)
    sinks = [f for f in report.findings if f.kind is FindingKind.MULTIPLE_SINKS]
    assert len(sinks) == 1
    assert "answer" in sinks[0].detail and "report" in sinks[0].detail


def test_no_output() -> None:
    # n2 reads n1's output and n1 reads n2's output: every signal is consumed → no terminal.
    nodes = [
        _node("n1", ["input", "answer"], ["draft"], control=NodeControl.CHECKPOINT),
        _node("n2", ["draft"], ["answer"]),
    ]
    report = _run(nodes)
    assert FindingKind.NO_OUTPUT in _kinds(report)


def test_unreachable() -> None:
    # n3 and n4 feed the output (extra -> n2 -> answer) so they reach the output, but they
    # are fed only by each other (floating <-> extra), never from the input → unreachable.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed", "extra"], ["answer"]),
        _node("n3", ["floating"], ["extra"]),
        _node("n4", ["extra"], ["floating"]),
    ]
    report = _run(nodes)
    unreachable = [f for f in report.findings if f.kind is FindingKind.UNREACHABLE]
    assert any(f.node in {"n3", "n4"} for f in unreachable)


def test_dead_end() -> None:
    # n2 reads 'parsed' and produces 'side' that nothing consumes; the last node n3 carries
    # the real output, so n2 is reachable from the input but reaches no output → dead_end.
    # ('side' is also a signal terminal, so multiple_sinks fires too; assert dead_end present.)
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed"], ["side"]),
        _node("n3", ["parsed"], ["answer"]),
    ]
    report = _run(nodes)
    dead = [f for f in report.findings if f.kind is FindingKind.DEAD_END]
    assert any(f.node == "n2" for f in dead)


def test_disconnected() -> None:
    # n3<->n4 form a component reading/producing only each other's names: neither reachable
    # from input nor reaching the output. n4 produces 'answer' is avoided.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed"], ["answer"]),
        _node("n3", ["loop_b"], ["loop_a"], control=NodeControl.CHECKPOINT),
        _node("n4", ["loop_a"], ["loop_b"]),
    ]
    report = _run(nodes)
    disconnected = [f for f in report.findings if f.kind is FindingKind.DISCONNECTED]
    assert any(f.node in {"n3", "n4"} for f in disconnected)


def test_unintended_cycle() -> None:
    # n2 and n3 cycle (n2 -> n3 -> n2) with NO checkpoint among them → unintended.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed", "back"], ["forward"]),
        _node("n3", ["forward"], ["back", "answer"]),
    ]
    report = _run(nodes)
    cycles = [f for f in report.findings if f.kind is FindingKind.UNINTENDED_CYCLE]
    assert len(cycles) >= 1
    assert "n2" in cycles[0].detail and "n3" in cycles[0].detail


def test_checkpoint_cycle_is_not_a_finding() -> None:
    # Same cycle, but n2 is a bounded-loop checkpoint → legitimate, no unintended_cycle.
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed", "back"], ["forward"], control=NodeControl.CHECKPOINT),
        _node("n3", ["forward"], ["back", "answer"]),
    ]
    report = _run(nodes)
    assert FindingKind.UNINTENDED_CYCLE not in _kinds(report)
