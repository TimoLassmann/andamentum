"""Worker: compile a typed design board into a ``SystemSpec`` — deterministic, no LLM.

The design heads produce the nodes; compilation derives everything else (input, state,
order, gates, an agent per head) and assembles the spec — whose construction *is* the
recipe check (every validator in ``spec.py`` runs here). Ported from the ``forge``
dump's ``design/compile.py``, retargeted to consume this module's ``DesignPlan``.

Order is a deterministic topological backbone over *signal* dependencies (entities
create no edge): each node follows the single next node, so shared-datum consumers are
sequenced, not branched. Only ``decision`` nodes branch; ``checkpoint`` adds a bounded
loop-back; ``consequential`` renders as a spine hole. Reachability is enforced by ``SystemSpec``.

Engine-free leaf worker (dialect Law 2).
"""

from __future__ import annotations

from .assemble import collection_data
from .naming import canonical_datum
from .naming import to_pascal as _pascal
from .naming import to_snake as _snake
from .schemas import INPUT_TOKENS, DataKind, DesignPlan, NodeDraft
from .spec import (
    END,
    AgentRole,
    AgentSpec,
    EntitySpec,
    FieldSpec,
    InputSpec,
    LoopCap,
    ModelSpec,
    NodeControl,
    NodeKind,
    NodeMode,
    NodeSpec,
    StateSpec,
    SystemSpec,
)

# The State field the rendered map scaffold records per-item failures into (soft-fail:
# a failed item is skipped and logged; ALL items failing raises). Reserved — a datum
# may not resolve to it.
ITEM_FAILURES_FIELD = "item_failures"

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
    """Canonical form of a data name (shared with the design stage so they agree)."""
    return canonical_datum(name, INPUT_TOKENS)


# Verbs that signal judgment OVER NATURAL-LANGUAGE TEXT (recipe §5: "open" reasoning).
# A job whose answer must be DERIVED from prose — not computed over an existing field —
# is a HEAD, not deterministic code. The prose-input guard below stops this from
# promoting a hard-key sort/filter (those read a computed field, not text).
_JUDGMENT_VERBS = frozenset(
    {
        "summar", "condens", "synthes", "rank", "score", "select", "classif", "categor",
        "extract", "identif", "interpret", "analyz", "assess", "evaluat", "judg", "draft",
        "compose", "rephrase", "rewrit", "translat", "paraphras", "verif", "valid", "critique",
        "describe", "explain", "generat", "distil", "highlight", "prioriti",
    }
)  # fmt: skip


def _is_judgment_over_text(job: str) -> bool:
    j = job.lower()
    return any(v in j for v in _JUDGMENT_VERBS)


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

    # Canonicalise data names so casing/spacing variants of one datum unify (a lossless
    # normalisation, not a fallback): 'Main ideas' and 'main_ideas' both become the field
    # 'main_ideas'. The design stage already canonicalises and diagnose→repair makes the
    # board coherent, so this is idempotent for a forge-designed plan — it only matters for
    # a hand-authored one.
    for c in plan.nodes:
        c.consumes = [_canon_datum(d) for d in c.consumes]
        c.produces = [_canon_datum(d) for d in c.produces]

    # Deterministic SPINE→HEAD promotion (recipe §5), symmetric to the HEAD→SPINE demotion
    # above. A job that reads natural-language text and judges its meaning (rank/select/
    # summarize/classify/extract over prose) is a HEAD — otherwise the code-writer is forced
    # to FAKE the judgment with string slicing. Guarded so it fires only when the node reads
    # PROSE (the raw input or a head's output), so a hard-key sort/filter over a computed
    # field stays spine. Fixpoint: a promoted head's output is prose, which can promote a
    # downstream consumer (the chain promotes from `input` outward).
    prose_vars: set[str] = set(INPUT_TOKENS) | {
        p for c in plan.nodes if c.kind is NodeKind.HEAD for p in c.produces
    }
    promoted = True
    while promoted:
        promoted = False
        for c in plan.nodes:
            if (
                c.kind is NodeKind.SPINE
                and c.control is NodeControl.NONE
                and not c.network
                and c.produces_kind is not DataKind.ENTITY
                and _is_judgment_over_text(c.job)
                and (set(c.consumes) & prose_vars)
            ):
                c.kind = NodeKind.HEAD
                prose_vars |= set(c.produces)
                promoted = True

    # §7 round-trip detector (entity classification, deterministic — never keywords). A datum
    # a single node both READS and WRITES (read-modify-write) is durable state: it carries a
    # value in and a changed value out, which only makes sense if that value persisted from an
    # earlier run. It MUST be declared an entity; a read-modify-write *signal* would be silently
    # forgotten when the run ends (faked persistence). Derive entity-ness from the data-flow
    # shape and fail loud on a mismatch — entity-ness is structural, not the model's say-so.
    for c in plan.nodes:
        rmw = sorted(set(c.consumes) & set(c.produces))
        if rmw and c.produces_kind is not DataKind.ENTITY:
            raise ValueError(
                f"node {c.job or c.id!r} reads and rewrites {rmw} in one step — that is durable "
                "state (a value loaded, changed, saved). Declare it an entity (produces_kind="
                "entity) so it persists across runs, or reshape the brief; a read-modify-write "
                "signal would be silently forgotten when the run ends."
            )

    # Map-over-items ('each') backstops — fail loud, mirroring diagnose's
    # each_needs_collection check (which repairs on the design path; a hand-authored plan
    # lands here directly). Collection-ness is COMPUTED, never declared per-datum: the
    # input is a collection iff the understand head said so, an EACH node's produce is a
    # collection, a WHOLE node's produce is a scalar (assemble.collection_data — the one
    # shared rule, so compile and diagnose cannot disagree).
    collections = collection_data(
        plan.nodes, input_is_collection=plan.why.input_is_collection
    )
    for c in plan.nodes:
        if c.mode is not NodeMode.EACH:
            continue
        if c.control is not NodeControl.NONE:
            raise ValueError(
                f"node {c.job or c.id!r} is 'each' (one run per item) but also declares "
                f"control={c.control.value!r}; a map step is a plain per-item transform (v1) — "
                "make it 'whole', or drop the control role."
            )
        if c.produces_kind is DataKind.ENTITY:
            raise ValueError(
                f"node {c.job or c.id!r} is 'each' (one run per item) but produces an entity; "
                "a map step produces the list of its per-item results (a signal, v1) — make it "
                "'whole', or produce a signal."
            )
        streams = [d for d in c.consumes if d in collections]
        stray = [d for d in c.consumes if d not in collections]
        if len(streams) != 1 or stray:
            raise ValueError(
                f"node {c.job or c.id!r} is 'each' (one run per item) but reads "
                f"{c.consumes or ['nothing']}; an 'each' step must consume exactly ONE "
                f"collection (its stream) and nothing else — collections read: {streams}, "
                f"scalars read: {stray}. Mark it 'whole', or have it read one collection."
            )

    # Fail loud on a dangling read. Every consumed signal must be produced by some node or
    # be the graph input — otherwise it would resolve to a State field nothing ever writes,
    # i.e. a silent None at runtime. We do NOT drop it (that hides an incomplete design and
    # the system would run but not do its job); we reject the spec so the gap is visible.
    produced = {p for c in plan.nodes for p in c.produces}
    for c in plan.nodes:
        dangling = [
            d for d in c.consumes if d not in produced and d.lower() not in INPUT_TOKENS
        ]
        if dangling:
            raise ValueError(
                f"node {c.job or c.id!r} reads {dangling}, which no node produces and is not the graph input. "
                "A read with no writer would be silently None at runtime — the design is incomplete. "
                "Fix the wiring (the gap is reported, never dropped)."
            )

    # Coherence gate (fail loud). Build the signal dependency graph and assert it is clean:
    # (1) single writer per signal — two nodes producing one variable means the last write
    #     silently wins; reject it. (2) no orphan — a signal produced but read by nobody and
    #     not the system output is discarded work. Both surface loudly, never resolved by
    #     execution order or silently pruned. Fan-in (a node reading several upstream signals)
    #     is untouched — it is the desired join, and the cure for an orphan.
    order = _topo_order(plan.nodes)
    writers: dict[str, list[NodeDraft]] = {}
    for c in plan.nodes:
        if c.produces_kind is DataKind.SIGNAL:
            for p in c.produces:
                writers.setdefault(p, []).append(c)
    dup = {v: ws for v, ws in writers.items() if len(ws) > 1}
    if dup:
        detail = "; ".join(
            f"{v!r} written by {[w.job or w.id for w in ws]}" for v, ws in dup.items()
        )
        raise ValueError(
            f"single-writer violation: {detail}. A signal has exactly one writer or the last write silently "
            "wins. If several steps refine one value, chain them (read X → produce a distinct refined name). "
            "Surfaced, never resolved by execution order."
        )
    # The system output is the signal the final (topologically last) node produces.
    out_var = next(
        (p for p in order[-1].produces if order[-1].produces_kind is DataKind.SIGNAL),
        None,
    )
    consumed = {d for c in plan.nodes for d in c.consumes}
    orphans = sorted(v for v in writers if v not in consumed and v != out_var)
    if orphans:
        detail = "; ".join(
            f"{v!r} (by {next(w.job or w.id for w in writers[v])!r})" for v in orphans
        )
        raise ValueError(
            f"orphan output(s): {detail} — produced but read by no step and not the system output. This is "
            "work the system discards: chain it forward (have a later step read it), make it the output, or "
            "remove the step. Surfaced, never silently pruned."
        )

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

    # Resolve conceptual data names to concrete State field names ONCE. An entity datum
    # gets a State field too — its working copy during the run (dialect L1: the run entry
    # seeds it from the store at the start and saves it back at the end). Only the graph
    # input is not a State field.
    field_of: dict[str, str] = {}
    origin: dict[str, str] = {}
    for d in sorted({d for c in plan.nodes for d in (c.consumes + c.produces)}):
        if d in INPUT_TOKENS:
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
                LoopCap(name=checkpoint_counter, limit=_DEFAULT_CHECKPOINT_CAP)
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
                network=c.network,
                mode=c.mode,
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
                state_field=field_of[ename],
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

    # state = the resolved SIGNAL fields (entities live in the store) + loop counters.
    # A collection datum (computed above) renders as list[str]; a scalar as str — the
    # annotations FOLLOW the propagated collection-ness, they are never declared.
    state_fields: list[FieldSpec] = []
    seen: set[str] = set()
    for fname in sorted(set(field_of.values())):
        seen.add(fname)
        annotation = "list[str]" if origin[fname] in collections else "str"
        state_fields.append(FieldSpec(name=fname, annotation=annotation, optional=True))
    for lc in loop_caps:
        if lc.name not in seen:
            seen.add(lc.name)
            state_fields.append(FieldSpec(name=lc.name, annotation="int", default="0"))
    # Any 'each' node → the shared per-item failure log the rendered map scaffold
    # appends to (soft-fail per item, aggregate raise when every item fails).
    if any(c.mode is NodeMode.EACH for c in plan.nodes):
        if ITEM_FAILURES_FIELD in seen:
            raise ValueError(
                f"a datum resolves to the reserved State field {ITEM_FAILURES_FIELD!r} "
                "(the map scaffold's per-item failure log); rename that datum."
            )
        state_fields.append(
            FieldSpec(name=ITEM_FAILURES_FIELD, annotation="list[str]", default="[]")
        )

    # Input law: one primary field — a list of items when the understand head flagged the
    # input as a collection (the run entry splits the CLI text on newlines), else a str.
    input_spec = InputSpec(
        model=ModelSpec(
            name=f"{sys_pascal}Input",
            fields=[
                FieldSpec(
                    name="request",
                    annotation=("list[str]" if plan.why.input_is_collection else "str"),
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
    )
