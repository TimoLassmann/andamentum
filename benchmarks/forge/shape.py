"""Deterministic shape inspection of a designed :class:`SystemSpec`.

``detect_features`` reports which control-flow features a compiled spec exhibits, by
pure structural inspection (no model, no heuristics): a loop is a declared loop cap, a
branch is a decision node or a multi-way successor, a fan-out is one written State
field consumed by two or more nodes, an entity is any declared durable record.

``outcome_matches`` then scores one run against the case's expectation: a refuse case
passes when forge refused; a build case passes when forge built *and* the design shows
the grammar feature the case calls for.
"""

from __future__ import annotations

from andamentum.forge.spec import NodeControl, SystemSpec

from .types import Case, RunOutcome

#: The structural features a buildable design may exhibit.
ALL_FEATURES = frozenset({"loop", "branch", "fanout", "entity"})

#: Map a case grammar to the feature its design must show. ``None`` means the design
#: must show *no* structural feature (a plain sequence).
_GRAMMAR_FEATURE: dict[str, str | None] = {
    "sequence": None,
    "none": None,
    "branch": "branch",
    "loop": "loop",
    "fanout": "fanout",
    "stateful": "entity",
}


def detect_features(spec: SystemSpec) -> set[str]:
    """Return the subset of {loop, branch, fanout, entity} the spec exhibits."""
    features: set[str] = set()

    # loop — a declared bounded cycle.
    if spec.loop_caps:
        features.add("loop")

    # branch — a decision node, or a node with two or more real (non-End) successors.
    branch = any(n.control is NodeControl.DECISION for n in spec.nodes) or any(
        len([s for s in n.successors if s != "End"]) >= 2 for n in spec.nodes
    )
    if branch:
        features.add("branch")

    # fanout — some written State field is read by two or more nodes.
    written: set[str] = {w for n in spec.nodes for w in n.writes}
    read_counts: dict[str, int] = {}
    for n in spec.nodes:
        for r in set(n.reads):
            read_counts[r] = read_counts.get(r, 0) + 1
    if any(read_counts.get(w, 0) >= 2 for w in written):
        features.add("fanout")

    # entity — any declared durable record.
    if spec.entities:
        features.add("entity")

    return features


def outcome_matches(case: Case, outcome: RunOutcome) -> bool:
    """True if one run's outcome satisfies the case's expectation."""
    if case.expected == "refuse":
        return outcome.kind == "refused"
    if case.expected == "build":
        if outcome.kind != "built":
            return False
        feature = _GRAMMAR_FEATURE.get(case.grammar)
        if feature is None:
            # A plain sequence must show no structural feature.
            return not (outcome.features & ALL_FEATURES)
        return feature in outcome.features
    return False
