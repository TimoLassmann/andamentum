"""Static topology assertions over the epistemic graph.

Phase 0 of the Move-3 plan. Uses ``graph.topology.topology()`` to
inspect the graph as a data structure (built from each node's
``run()`` return type annotation) and asserts properties that the
recurring bug class would violate:

* ``CheckCompletion`` must NOT be reachable in one step from
  ``AbandonOrDemote``. The recurring routing bug had AbandonOrDemote
  returning CheckCompletion directly, stranding soft-promoted and
  pass-verdict claims. Tightening AbandonOrDemote's return annotation
  makes this a static check; future widenings of the annotation will
  fail this test in CI.

* Every node must be reachable from ``PrepareObjective``. A node
  registered but unreachable is a dormancy bug (the same shape as the
  original DecomposeQuestionOperation un-wiring).

* No accidental cycles (intentional cycles like the Scrutinize ↔
  Investigate inquiry loop are allowlisted).

These tests run against the LIVE topology (whatever is currently
encoded in ``graph/nodes.py``) and survive the Move 3 refactor — they
read return type annotations, not file structure.
"""

from __future__ import annotations

from andamentum.epistemic.graph.nodes import (
    AbandonOrDemote,
    CheckCompletion,
    CheckSynthesisDemand,
    PrepareObjective,
    Synthesize,
    SynthesizeInsufficient,
)
from andamentum.epistemic.graph.topology import (
    all_nodes,
    reachable_from,
    topology,
)


def test_check_completion_not_in_abandon_or_demote_successors() -> None:
    """The recurring routing-bug invariant. AbandonOrDemote must NOT
    return CheckCompletion directly — that path strands soft-promoted
    and pass-verdict claims by skipping PromoteToSupported, the
    dispatcher that routes them to IBE.

    If this test fires, someone widened AbandonOrDemote's return
    annotation back to ``Union[..., CheckCompletion]``. Don't. The
    correct routing is AbandonOrDemote → PromoteToSupported →
    (ClusterEvidence | CheckCompletion). PromoteToSupported is
    idempotent — when there's no work, it itself returns
    CheckCompletion.
    """
    topo = topology()
    successors = topo[AbandonOrDemote]
    assert CheckCompletion not in successors, (
        f"AbandonOrDemote.run() return annotation includes "
        f"CheckCompletion. Successors: "
        f"{sorted(s.__name__ for s in successors)}. "
        "This was the recurring routing bug — AbandonOrDemote "
        "shouldn't terminate directly; route through "
        "PromoteToSupported instead."
    )


def test_every_node_reachable_from_prepare_objective() -> None:
    """Dormancy check at the topology level. Every BaseNode subclass
    discovered in graph/nodes.py must be reachable from PrepareObjective
    (the entry point). A registered-but-unreachable node is a dormancy
    bug — the same class of issue the structural-wiring test catches
    for operations.
    """
    topo = topology()
    everyone = all_nodes(topo)
    reachable = reachable_from(PrepareObjective, topo)
    unreachable = everyone - reachable
    assert not unreachable, (
        "Nodes registered in graph/nodes.py but not reachable from "
        f"PrepareObjective: {sorted(n.__name__ for n in unreachable)}. "
        "Either wire them into a successor or delete them."
    )


def test_synthesize_terminates_at_end() -> None:
    """Synthesize is the only non-CheckCompletion path to End[...]
    in the current topology. Its successor set should not include any
    BaseNode subclasses — only End[...] (which the topology helper
    filters out, so the set is empty).
    """
    topo = topology()
    successors = topo[Synthesize]
    assert successors == frozenset(), (
        f"Synthesize should terminate at End[...] only; got successors "
        f"{sorted(s.__name__ for s in successors)}"
    )


def test_synthesize_insufficient_terminates_at_end() -> None:
    """``SynthesizeInsufficient`` is the structural fallibilism terminal:
    reached when ``CheckSynthesisDemand`` finds the demand unsatisfied
    AND no claim is eligible for further work. Like ``Synthesize``, it
    must terminate at ``End[...]`` only — not loop back into the graph
    (the per-claim cap already drained eligibility, so a second
    loop-back would be unbounded)."""
    topo = topology()
    successors = topo[SynthesizeInsufficient]
    assert successors == frozenset(), (
        f"SynthesizeInsufficient should terminate at End[...] only; "
        f"got successors {sorted(s.__name__ for s in successors)}"
    )


def test_check_synthesis_demand_routes_to_insufficient() -> None:
    """Maximal-B topology contract. ``CheckSynthesisDemand`` must be
    able to route to ``SynthesizeInsufficient`` — that's the
    architectural "system suspends judgment" path. Without this edge
    the writer agent gets called on no-data state and invents a
    directional verdict (the K3 failure)."""
    topo = topology()
    successors = topo[CheckSynthesisDemand]
    assert SynthesizeInsufficient in successors, (
        f"CheckSynthesisDemand must include SynthesizeInsufficient as a "
        f"successor; got {sorted(s.__name__ for s in successors)}. If "
        "this edge is missing, the gate's no-eligible-claims branch "
        "falls through to Synthesize and the writer fabricates a "
        "verdict from absent evidence."
    )


def test_topology_size_matches_node_count() -> None:
    """Sanity: the topology helper found roughly the right number
    of nodes. Currently 21 in graph/nodes.py. This test isn't a hard
    contract (the count will change during Move 3); it's an early
    warning if the reflection helper accidentally drops nodes.
    """
    topo = topology()
    assert 15 <= len(topo) <= 30, (
        f"Topology contains {len(topo)} nodes — expected ~21. "
        "If Move 3 has progressed, update the bounds; otherwise the "
        "topology() helper may be missing or double-counting nodes."
    )
