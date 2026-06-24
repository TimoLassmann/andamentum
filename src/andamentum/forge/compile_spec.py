"""Worker: compile a typed design board into a ``SystemSpec`` — deterministic, no LLM.

The design heads produce the nodes; compilation derives everything else (input, state,
order, gates, an agent per head) and assembles the spec — whose construction *is* the
recipe check (every validator in ``spec.py`` runs here). Ported from the ``forge``
dump's ``design/compile.py``, retargeted to consume this module's ``DesignPlan``.

Order is a deterministic topological backbone over *signal* dependencies (entities
create no edge): each node follows the single next node, so shared-datum consumers are
sequenced, not branched. Only ``decision`` nodes branch; ``checkpoint`` adds a bounded
loop-back; ``consequential`` gets a HITL gate. Reachability is enforced by ``SystemSpec``.

Engine-free leaf worker (dialect Law 2).
"""

from __future__ import annotations

from .naming import to_pascal as _pascal
from .naming import to_snake as _snake
from .schemas import INPUT_TOKENS, DataKind, DesignPlan, NodeDraft
from .spec import (
    END,
    AgentRole,
    AgentSpec,
    EntitySpec,
    FieldSpec,
    GateKind,
    HitlGate,
    InputSpec,
    LoopCap,
    ModelSpec,
    NodeControl,
    NodeKind,
    NodeSpec,
    StateSpec,
    SystemSpec,
)

_DEFAULT_CHECKPOINT_CAP = 2


def _unique(base: str, used: set[str]) -> str:
    name, k = base, 2
    while name in used:
        name, k = f"{base}{k}", k + 1
    used.add(name)
    return name


def _split_multi_successor_heads(
    nodes: list[NodeSpec], used_names: set[str]
) -> list[NodeSpec]:
    """Split any multi-successor HEAD into a single-successor head + a router (recipe
    dichotomy: an LLM call never also does the path-picking)."""
    out: list[NodeSpec] = []
    for n in nodes:
        if n.kind is NodeKind.HEAD and len(n.successors) > 1:
            router = _unique(_pascal(f"route {n.name}", f"{n.name}Router"), used_names)
            out.append(n.model_copy(update={"successors": [router]}))
            out.append(
                NodeSpec(
                    name=router,
                    kind=NodeKind.SPINE,
                    purpose=f"Route on the result of {n.name}.",
                    job=f"Route on the result of {n.name}.",
                    successors=list(n.successors),
                    reads=list(n.writes),
                    serves=n.serves,
                )
            )
        else:
            out.append(n)
    return out


def _role_for(job: str) -> AgentRole:
    j = job.lower()
    if any(k in j for k in ("classif", "decide", "route", "triage")):
        return AgentRole.TRIAGE
    if any(k in j for k in ("check", "sufficient", "verif", "valid")):
        return AgentRole.VERIFY
    if any(
        k in j for k in ("write", "answer", "summar", "compose", "draft", "generate")
    ):
        return AgentRole.SYNTHESIZE
    if any(k in j for k in ("plan", "need")):
        return AgentRole.PLAN
    return AgentRole.OTHER


def _topo_order(nodes: list[NodeDraft]) -> list[NodeDraft]:
    """Deterministic topological order over *signal* dependencies (entities create
    none). Tie-break by declaration order. Raises on a signal-dependency cycle."""
    sig = {
        c.id: (set(c.produces) if c.produces_kind is DataKind.SIGNAL else set())
        for c in nodes
    }
    deps = {
        c.id: {o.id for o in nodes if o.id != c.id and (set(c.consumes) & sig[o.id])}
        for c in nodes
    }
    idx = {c.id: i for i, c in enumerate(nodes)}
    placed: set[str] = set()
    order: list[NodeDraft] = []
    remaining = list(nodes)
    while remaining:
        ready = sorted(
            (c for c in remaining if deps[c.id] <= placed), key=lambda c: idx[c.id]
        )
        if not ready:
            raise ValueError("signal-dependency cycle among nodes (no valid order)")
        chosen = ready[0]
        order.append(chosen)
        placed.add(chosen.id)
        remaining.remove(chosen)
    return order


def _entity_names(nodes: list[NodeDraft]) -> set[str]:
    return {p for n in nodes if n.produces_kind is DataKind.ENTITY for p in n.produces}


def _canon_datum(name: str) -> str:
    """Canonical form of a data name: the graph-input tokens kept verbatim, everything
    else snake_cased so casing/spacing variants of one datum unify (not collide)."""
    if name.lower() in INPUT_TOKENS:
        return name.lower()
    return _snake(name, name, max_words=0)


def compile_spec(plan: DesignPlan) -> SystemSpec:
    """Assemble (and thereby validate) a ``SystemSpec`` from a design plan."""
    if not plan.nodes:
        raise ValueError("cannot compile: design has no nodes")

    # A node that reaches the network OR pauses for human approval is deterministic
    # spine code, never an LLM call. Normalize a mislabelled HEAD first.
    for c in plan.nodes:
        if c.kind is NodeKind.HEAD and (
            c.network or c.control is NodeControl.CONSEQUENTIAL
        ):
            c.kind = NodeKind.SPINE

    # Canonicalise data names so casing/spacing variants of one datum unify. A real
    # model routinely emits 'Main ideas' (produces) and 'main_ideas' (consumes) for the
    # SAME thing; both resolve to the State field 'main_ideas'. Without this they look
    # like two distinct names colliding on one field (a hard error); canonicalising first
    # makes the producer and consumer agree, so the wiring connects instead of crashing.
    for c in plan.nodes:
        c.consumes = [_canon_datum(d) for d in c.consumes]
        c.produces = [_canon_datum(d) for d in c.produces]

    # Deterministic wiring safety net: drop any consume that no node produces and that is
    # not the graph input — a hallucinated dependency a small model sometimes invents. This
    # guarantees every State read resolves to a field written by an upstream node (topo
    # order sequences producer before consumer), so a built body never reads a never-set
    # None. Sound wiring by construction; the residual after reconciliation can't crash.
    produced = {p for c in plan.nodes for p in c.produces}
    for c in plan.nodes:
        c.consumes = [
            d for d in c.consumes if d in produced or d.lower() in INPUT_TOKENS
        ]

    sys_name = _snake(plan.why.purpose, "system")
    sys_pascal = "".join(p[:1].upper() + p[1:] for p in sys_name.split("_"))

    # node class names (PascalCase, unique), keyed by draft id
    node_names: dict[str, str] = {}
    used_nodes: set[str] = set()
    for i, c in enumerate(plan.nodes):
        node_names[c.id] = _unique(_pascal(c.job or c.id, f"Node{i + 1}"), used_nodes)

    # one agent per head node
    agents: list[AgentSpec] = []
    agent_for: dict[str, str] = {}
    entity_set = _entity_names(plan.nodes)
    used_agents: set[str] = set()
    used_models: set[str] = {f"{sys_pascal}Input"}
    for c in plan.nodes:
        if c.kind is NodeKind.HEAD:
            an = _unique(_snake(c.job or c.id, "head"), used_agents)
            agent_for[c.id] = an
            out_name = _unique(_pascal(c.job or c.id, "Head") + "Out", used_models)
            agents.append(
                AgentSpec(
                    name=an,
                    role=_role_for(c.job),
                    prompt=c.job or f"Do the job of {node_names[c.id]}.",
                    output=ModelSpec(
                        name=out_name,
                        fields=[FieldSpec(name="result", annotation="str")],
                    ),
                )
            )

    # Order: signal-only deps → a deterministic topological linear backbone.
    order = _topo_order(plan.nodes)
    next_of: dict[str, NodeDraft] = {
        order[i].id: order[i + 1] for i in range(len(order) - 1)
    }

    def branch_targets(card: NodeDraft) -> list[str]:
        signals = card.produces if card.produces_kind is DataKind.SIGNAL else []
        outs: list[str] = []
        for other in plan.nodes:
            if other.id != card.id and (set(other.consumes) & set(signals)):
                if node_names[other.id] not in outs:
                    outs.append(node_names[other.id])
        return sorted(outs)

    def backbone_next(card: NodeDraft) -> list[str]:
        return [node_names[next_of[card.id].id]] if card.id in next_of else [END]

    # Resolve conceptual data names to concrete State field names ONCE.
    field_of: dict[str, str] = {}
    origin: dict[str, str] = {}
    for d in sorted({d for c in plan.nodes for d in (c.consumes + c.produces)}):
        if d in INPUT_TOKENS or d in entity_set:
            continue
        fname = _snake(d, "value", max_words=0)
        if fname in origin and origin[fname] != d:
            raise ValueError(
                f"data names {origin[fname]!r} and {d!r} both resolve to State field {fname!r}; "
                "rename one so the node contract stays unambiguous."
            )
        origin[fname] = d
        field_of[d] = fname

    def contract_fields(names: list[str]) -> list[str]:
        out: list[str] = []
        for d in names:
            if d in INPUT_TOKENS:
                out.append("request")
            elif d in field_of:
                out.append(field_of[d])
        return list(dict.fromkeys(out))

    nodes: list[NodeSpec] = []
    loop_caps: list[LoopCap] = []
    hitl: list[HitlGate] = []
    for c in order:
        succ = (
            (branch_targets(c) or backbone_next(c))
            if c.control is NodeControl.DECISION
            else backbone_next(c)
        )
        checkpoint_counter: str | None = None
        if c.control is NodeControl.CHECKPOINT:
            area_first = next(
                (node_names[x.id] for x in order if x.area == c.area), None
            )
            if area_first and area_first != node_names[c.id] and area_first not in succ:
                succ = succ + [area_first]
            checkpoint_counter = f"{node_names[c.id].lower()}_loops"
            loop_caps.append(
                LoopCap(
                    name=checkpoint_counter,
                    limit=c.checkpoint_cap or _DEFAULT_CHECKPOINT_CAP,
                )
            )
        if c.control is NodeControl.CONSEQUENTIAL:
            hitl.append(
                HitlGate(
                    node=node_names[c.id],
                    kind=GateKind.APPROVAL,
                    purpose=c.job or node_names[c.id],
                )
            )
        node_reads = contract_fields(c.consumes)
        node_writes = contract_fields(c.produces)
        if checkpoint_counter is not None:
            if checkpoint_counter not in node_reads:
                node_reads = node_reads + [checkpoint_counter]
            if checkpoint_counter not in node_writes:
                node_writes = node_writes + [checkpoint_counter]
        nodes.append(
            NodeSpec(
                name=node_names[c.id],
                kind=c.kind,
                purpose=c.job,
                successors=succ,
                agent=agent_for.get(c.id) if c.kind is NodeKind.HEAD else None,
                job=c.job,
                reads=node_reads,
                writes=node_writes,
                consumes=c.consumes,
                produces=c.produces,
                control=c.control,
                checkpoint_cap=c.checkpoint_cap,
                network=c.network,
                serves=c.area,
            )
        )

    nodes = _split_multi_successor_heads(nodes, used_nodes)
    entry = node_names[order[0].id]

    # entities — durable data. v1 models each as id + content (forge used a separate
    # entity-modeler head; here we keep a minimal valid record and leave fields to fill).
    entities: list[EntitySpec] = []
    for ename in sorted(entity_set):
        entities.append(
            EntitySpec(
                record_type=_snake(ename, "record"),
                model=ModelSpec(
                    name=_unique(_pascal(ename, "Entity"), used_models),
                    fields=[
                        FieldSpec(name="id", annotation="str", description="store id"),
                        FieldSpec(
                            name="content",
                            annotation="str",
                            description=f"the {ename} payload",
                        ),
                    ],
                ),
            )
        )

    # state = the resolved SIGNAL fields (entities live in the store) + loop counters
    state_fields: list[FieldSpec] = []
    seen: set[str] = set()
    for fname in sorted(set(field_of.values())):
        seen.add(fname)
        state_fields.append(FieldSpec(name=fname, annotation="str", optional=True))
    for lc in loop_caps:
        if lc.name not in seen:
            seen.add(lc.name)
            state_fields.append(FieldSpec(name=lc.name, annotation="int", default="0"))

    input_spec = InputSpec(
        model=ModelSpec(
            name=f"{sys_pascal}Input",
            fields=[
                FieldSpec(
                    name="request",
                    annotation="str",
                    description=plan.why.boundary_in or "the request",
                )
            ],
        ),
        primary_text_field="request",
        validation_rules=["reject if blank after stripping whitespace"],
    )

    return SystemSpec(
        name=sys_name,
        description=plan.why.purpose,
        input=input_spec,
        entities=entities,
        state=StateSpec(fields=state_fields),
        agents=agents,
        nodes=nodes,
        entry_node=entry,
        loop_caps=loop_caps,
        hitl_gates=hitl,
    )
