"""The Agent Graph Recipe, made executable.

A :class:`SystemSpec` is the typed declaration of one agentic system: the five
things the recipe (``docs/AGENT_GRAPH_RECIPE.md`` §9) says every new system must
declare — **Input, Entities, State, Nodes, Agents** — plus its loop caps.

The purpose of this module is to make the recipe *checkable*. A ``SystemSpec``
either validates against the recipe's Invariants and Rules or it raises. The
validators below are the prose turned into code:

- I2  every successor edge points at a real node (or ``"End"``)
- I3  spine nodes call no LLM; head nodes name exactly one agent
- I4  an agent's output model has 1–6 flat fields
- I5  a closed-vocabulary field is an enum, not a free string
- I6  any cycle in the graph is covered by a declared loop cap
- R1  state carries signals, not entities
- R5  every head's agent is a declared registry entry
- Input law  the input model has one primary natural-language field

Nothing here renders code or calls a model. This module only *describes and
validates* a system. Rendering lives in ``render.py``; designing a spec from a
natural-language brief is the job of the ``forge`` graph (``graph.py``).

Dialect note: this is a leaf worker file — pure ``pydantic`` + stdlib, imports no
graph engine. Ported from the ``forge`` exploratory dump, where it was already
dialect-clean (``andamentum-agentic-dialect check`` reported zero violations).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from .naming import IDENT as _IDENT
from .naming import PASCAL as _PASCAL

# Recipe I4 — an LLM head's output model stays tiny and flat.
#
# Two ceilings, by role. A *reasoning* head (triage, plan, verify, synthesize,
# validate) does open generation, where small models stay reliable only when the
# output is tiny: 1–6 fields. A *form-filling* head (compose, fill) does typed
# extraction into an enum-guarded schema — that stays reliable at a higher field
# count *because* there is no open generation, so it gets a larger ceiling.
# This resolves the I4-vs-WorkQuery tension found while writing the work-graph
# example (the real Compose emits an 11-field query). See AGENT_GRAPH_RECIPE.md I4.
MIN_OUTPUT_FIELDS = 1
MAX_OUTPUT_FIELDS = 6  # reasoning heads
MAX_FORM_FIELDS = 12  # form-filling heads (compose / fill)

# Identifier patterns live in `naming` (shared with the design compiler); aliased
# here so the field validators below read unchanged.

#: Base annotations a graph ``State`` field may use. State carries only signals
#: (counters, flags, queues) and IDs into the store — never entities (Rule R1).
#: This is a deliberately small allow-list; anything richer belongs in an entity.
SIGNAL_BASE_TYPES = frozenset(
    {
        "int",
        "bool",
        "str",
        "float",
        "list[str]",
        "set[str]",
        "list[int]",
        "dict[str, int]",
    }
)

#: The end-of-run sentinel a node may name as a successor.
END = "End"


class NodeKind(str, Enum):
    """Whether a node is part of the deterministic spine or an LLM head (I3)."""

    SPINE = "spine"  # pure code: load, query, count, filter, gate, write, assemble
    HEAD = "head"  # one tiny LLM call


class AgentRole(str, Enum):
    """The legitimate LLM-head roles (recipe §5). 'other' is an escape hatch
    that the generator should treat as a smell to be justified, not a default."""

    TRIAGE = "triage"  # coarse intent, one enum label
    PLAN = "plan"  # state the information need in plain language
    COMPOSE = "compose"  # fill an enum-guarded query/spec
    VERIFY = "verify"  # is this on-topic / sufficient? (the Demand shape)
    SYNTHESIZE = "synthesize"  # the narrative core only — prose
    VALIDATE = "validate"  # does the prose match the data?
    FILL = "fill"  # fill one section of a typed declaration
    OTHER = "other"


#: Roles that do typed extraction into a guarded schema — allowed more output
#: fields than reasoning roles (see MAX_FORM_FIELDS).
FORM_ROLES = frozenset({AgentRole.COMPOSE, AgentRole.FILL})


class NodeControl(str, Enum):
    """How a node participates in control flow, set during decomposition.

    Drives order derivation (see generator-redesign.md §8): a ``decision`` node
    routes among successors, a ``checkpoint`` may loop back (bounded by a cap), and
    a ``consequential`` node (an irreversible/outward action) gets a human gate
    before it. ``none`` is an ordinary step in the data-flow chain.
    """

    NONE = "none"
    DECISION = "decision"
    CHECKPOINT = "checkpoint"
    CONSEQUENTIAL = "consequential"


class FieldConstraints(BaseModel):
    """Typed Field(...) constraints. Kept explicit rather than a free dict so the
    contract is checkable and the renderer can emit them verbatim."""

    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


class FieldSpec(BaseModel):
    """One field of a model (input, entity, state, or agent output)."""

    name: str = Field(description="snake_case field name")
    annotation: str = Field(
        description="Python type as text, e.g. 'str', 'list[str]', 'date'"
    )
    description: str = ""
    optional: bool = False
    default: str | None = Field(
        default=None, description="Python literal as text, e.g. '0', 'None', '[]'"
    )
    enum_values: list[str] | None = Field(
        default=None,
        description="If set, this is a closed vocabulary rendered as an Enum (recipe I5).",
    )
    constraints: FieldConstraints = FieldConstraints()

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"field name {v!r} must be a snake_case identifier")
        return v

    @field_validator("enum_values")
    @classmethod
    def _enum_values_nonempty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("enum_values, when present, must list at least one value")
        return v

    @property
    def is_closed_vocabulary(self) -> bool:
        return self.enum_values is not None


class ModelSpec(BaseModel):
    """A typed model to be generated (input, entity, or an agent's output)."""

    name: str = Field(description="PascalCase model name")
    description: str = ""
    fields: list[FieldSpec]

    @field_validator("name")
    @classmethod
    def _name_is_pascal(cls, v: str) -> str:
        if not _PASCAL.match(v):
            raise ValueError(f"model name {v!r} must be PascalCase")
        return v

    @field_validator("fields")
    @classmethod
    def _fields_present_and_unique(cls, v: list[FieldSpec]) -> list[FieldSpec]:
        if not v:
            raise ValueError("a model must declare at least one field")
        names = [f.name for f in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate field names: {sorted(dupes)}")
        return v


class InputSpec(BaseModel):
    """The system's one validated input (Input law)."""

    model: ModelSpec
    primary_text_field: str = Field(
        description="The single natural-language field that drives the run"
    )
    validation_rules: list[str] = Field(
        default_factory=list,
        description="Plain-language rules enforced at the door, e.g. 'reject if shorter than 10 chars'",
    )

    @model_validator(mode="after")
    def _primary_field_exists(self) -> "InputSpec":
        if self.primary_text_field not in {f.name for f in self.model.fields}:
            raise ValueError(
                f"primary_text_field {self.primary_text_field!r} is not a field of {self.model.name}"
            )
        return self


class EntitySpec(BaseModel):
    """A durable record type. Its truth lives in the store *between* runs; a working copy
    lives in one State field *during* the run (dialect L1: loaded at start, saved at end)."""

    model: ModelSpec
    store: str = Field(default="store", description="Where the entity lives")
    record_type: str = Field(
        description="The store collection (and metadata discriminator)"
    )
    state_field: str = Field(
        default="",
        description=(
            "The State field that holds this entity's working copy during the run. The run "
            "entry loads the durable record into it at the start and saves it back at the end "
            "(single-record, constant key). Empty on a hand-authored spec with no binding."
        ),
    )

    @field_validator("record_type")
    @classmethod
    def _record_type_ident(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"record_type {v!r} must be a snake_case identifier")
        return v


class StateSpec(BaseModel):
    """Run-scoped signals only: counters, flags, queues, and IDs (Rules R1/R2).

    Entity-name checking (a state field must not embed an entity) happens at the
    :class:`SystemSpec` level, where the entity names are known.
    """

    fields: list[FieldSpec] = Field(default_factory=list)

    @field_validator("fields")
    @classmethod
    def _signal_like(cls, v: list[FieldSpec]) -> list[FieldSpec]:
        for f in v:
            if f.is_closed_vocabulary:
                continue  # an enum signal is fine
            # IDs are strings that point into the store; allowed.
            if f.annotation in SIGNAL_BASE_TYPES:
                continue
            if (
                f.annotation.endswith("| None")
                and f.annotation[: -len("| None")].strip() in SIGNAL_BASE_TYPES
            ):
                continue
            raise ValueError(
                f"state field {f.name!r} has type {f.annotation!r}; state may hold only signals "
                f"(one of {sorted(SIGNAL_BASE_TYPES)}), enums, or IDs (str). "
                "Richer data belongs in an entity (Rule R1)."
            )
        return v


class AgentSpec(BaseModel):
    """A registry entry for one LLM head (Rule R5)."""

    name: str = Field(description="snake_case registry name")
    role: AgentRole
    prompt: str = Field(description="The system prompt for this head")
    output: ModelSpec
    retries: int = 3
    output_retries: int = 5

    @field_validator("name")
    @classmethod
    def _name_ident(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"agent name {v!r} must be a snake_case identifier")
        return v

    @model_validator(mode="after")
    def _output_is_tiny(self) -> "AgentSpec":
        n = len(self.output.fields)
        cap = MAX_FORM_FIELDS if self.role in FORM_ROLES else MAX_OUTPUT_FIELDS
        if not (MIN_OUTPUT_FIELDS <= n <= cap):
            kind = "form-filling" if self.role in FORM_ROLES else "reasoning"
            raise ValueError(
                f"agent {self.name!r} ({kind}, role={self.role.value}) output {self.output.name!r} "
                f"has {n} fields; a {kind} head's output must have {MIN_OUTPUT_FIELDS}–{cap} (recipe I4). "
                "Split the call, or move logic into the deterministic spine."
            )
        return self


class NodeSpec(BaseModel):
    """One graph node, typed by its successors (I2)."""

    name: str = Field(description="PascalCase node class name")
    kind: NodeKind
    purpose: str = ""
    successors: list[str] = Field(description="Names of successor nodes, or 'End'")
    agent: str | None = Field(
        default=None, description="Required iff kind == head; names an AgentSpec"
    )
    network: bool = Field(
        default=False,
        description="This spine node reaches the network (a declared external effect). Its body may use an "
        "HTTP client (httpx/requests); it runs only behind the container sandbox, never a bare subprocess.",
    )

    # The node's RESOLVED data contract — concrete State field names (snake_case),
    # populated by `compile`. This is what the renderer prints into the hole and the
    # builder fills against; `consumes`/`produces` below are the conceptual data
    # names from decomposition that these are resolved from. Empty on hand-authored
    # specs (the Wiring Invariant only fires when populated).
    reads: list[str] = Field(
        default_factory=list, description="State field names this node may read"
    )
    writes: list[str] = Field(
        default_factory=list, description="State field names this node must set"
    )

    # Node-card fields, populated by the decompose-first design front-end
    # (generator-redesign.md). Optional so hand-authored specs stay valid.
    job: str = Field(
        default="",
        description="One plain sentence: the node's single job (and runtime narration)",
    )
    consumes: list[str] = Field(
        default_factory=list, description="Data names this node reads"
    )
    produces: list[str] = Field(
        default_factory=list,
        description="Data names this node writes (one, conceptually)",
    )
    control: NodeControl = Field(
        default=NodeControl.NONE, description="Role in control flow (see NodeControl)"
    )
    serves: str = Field(
        default="",
        description="Trace link: the area/responsibility id this node serves",
    )

    @field_validator("name")
    @classmethod
    def _name_pascal(cls, v: str) -> str:
        if not _PASCAL.match(v):
            raise ValueError(f"node name {v!r} must be PascalCase")
        return v

    @field_validator("successors")
    @classmethod
    def _successors_present(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "a node must declare at least one successor (a node name or 'End')"
            )
        return v

    @model_validator(mode="after")
    def _kind_matches_agent(self) -> "NodeSpec":
        # Recipe I3: flow control is the spine's job; only heads call an LLM.
        if self.kind is NodeKind.HEAD and not self.agent:
            raise ValueError(f"head node {self.name!r} must name an agent")
        if self.kind is NodeKind.SPINE and self.agent:
            raise ValueError(
                f"spine node {self.name!r} names an agent {self.agent!r}; spine nodes are pure code (I3). "
                "Make it a head, or drop the agent."
            )
        if self.network and self.kind is NodeKind.HEAD:
            raise ValueError(
                f"head node {self.name!r} declares network=True; a head is one LLM call and never writes "
                "fetch code. A node that reaches the network is a spine node (head reasons, spine fetches)."
            )
        return self


class LoopCap(BaseModel):
    """A hard bound on a cycle, checked by a deterministic node (I6)."""

    name: str = Field(description="The state field holding the counter")
    limit: int = Field(
        ge=1, description="Maximum iterations before the graph forces an exit"
    )


class SystemSpec(BaseModel):
    """The complete, recipe-checked declaration of one agentic system.

    Validating an instance of this model is equivalent to checking the system
    against the Agent Graph Recipe. It is the contract that the generator fills
    and the renderer consumes.
    """

    name: str = Field(
        description="snake_case import name (the package's `import {name}`)"
    )
    pypi_name: str | None = Field(
        default=None, description="Defaults to 'mosaic-{name}'"
    )
    description: str

    input: InputSpec
    entities: list[EntitySpec] = Field(default_factory=list)
    state: StateSpec
    agents: list[AgentSpec] = Field(default_factory=list)
    nodes: list[NodeSpec]
    entry_node: str = Field(description="The node the graph starts at")

    loop_caps: list[LoopCap] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_ident(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"system name {v!r} must be a snake_case identifier")
        return v

    @model_validator(mode="after")
    def _default_pypi_name(self) -> "SystemSpec":
        if self.pypi_name is None:
            object.__setattr__(
                self, "pypi_name", f"mosaic-{self.name.replace('_', '-')}"
            )
        return self

    @model_validator(mode="after")
    def _check_recipe(self) -> "SystemSpec":
        node_names = {n.name for n in self.nodes}
        if len(node_names) != len(self.nodes):
            raise ValueError("duplicate node names")
        agent_names = {a.name for a in self.agents}
        entity_names = {e.model.name for e in self.entities}

        # Entry node exists.
        if self.entry_node not in node_names:
            raise ValueError(f"entry_node {self.entry_node!r} is not a declared node")

        # I2: every successor points at a real node or End.
        targets = node_names | {END}
        for n in self.nodes:
            for s in n.successors:
                if s not in targets:
                    raise ValueError(
                        f"node {n.name!r} has successor {s!r} which is neither a node nor 'End'"
                    )

        # At least one path can terminate.
        if not any(END in n.successors for n in self.nodes):
            raise ValueError("no node returns 'End'; the graph cannot terminate")

        # Reachability: every node is reachable from the entry, and an End is
        # reachable. A node the entry can never get to is dead weight; a graph whose
        # End is unreachable cannot finish. (Catches stranded nodes after branching.)
        adj = {n.name: n.successors for n in self.nodes}
        reached: set[str] = set()
        end_reachable = False
        stack = [self.entry_node]
        while stack:
            cur = stack.pop()
            if cur in reached:
                continue
            reached.add(cur)
            for s in adj.get(cur, []):
                if s == END:
                    end_reachable = True
                elif s not in reached:
                    stack.append(s)
        if node_names - reached:
            raise ValueError(
                f"nodes unreachable from entry {self.entry_node!r}: {sorted(node_names - reached)}"
            )
        if not end_reachable:
            raise ValueError(f"no End is reachable from entry {self.entry_node!r}")

        # R5: every head's agent is declared.
        for n in self.nodes:
            if n.kind is NodeKind.HEAD and n.agent not in agent_names:
                raise ValueError(
                    f"head node {n.name!r} names undeclared agent {n.agent!r}"
                )

        # R1: a state field must not be an entity.
        for f in self.state.fields:
            base = (
                f.annotation.replace("| None", "")
                .replace("list[", "")
                .replace("set[", "")
                .strip("[] ")
            )
            if base in entity_names:
                raise ValueError(
                    f"state field {f.name!r} embeds entity {base!r}; state holds IDs, not entities (R1). "
                    f"Store the entity and keep its id in state."
                )

        # Wiring Invariant: a node's resolved data contract must name real State
        # fields. `reads`/`writes` hold State field names (resolved by `compile` from
        # the conceptual consumes/produces); every one must be a declared State field
        # or the input's primary field (which the renderer threads into State). This
        # is what makes "a node references a field that does not exist" a design-time
        # error instead of the runtime crash it used to be. Fires only when the
        # contract is populated, so hand-authored specs that omit it stay valid.
        state_field_names = {f.name for f in self.state.fields} | {
            self.input.primary_text_field
        }
        for n in self.nodes:
            for label, names in (("reads", n.reads), ("writes", n.writes)):
                for fname in names:
                    if fname not in state_field_names:
                        raise ValueError(
                            f"node {n.name!r} {label} {fname!r}, which is not a declared State field "
                            f"or the input field {self.input.primary_text_field!r}; a node's data contract "
                            f"must resolve to real state (Wiring Invariant)."
                        )

        # I6: if the graph has a cycle, a loop cap must exist.
        if self._has_cycle() and not self.loop_caps:
            raise ValueError(
                "the graph contains a cycle but declares no loop_caps; "
                "every loop must be bounded by a counter checked in a spine node (I6)."
            )

        # I6 counter integrity: every LoopCap.name must be a declared State field
        # with an int (or int | None) annotation. A counter that is not in State, or
        # has a non-int type, would silently fail at runtime — reject it here.
        state_field_map = {f.name: f.annotation for f in self.state.fields}
        for lc in self.loop_caps:
            if lc.name not in state_field_map:
                raise ValueError(
                    f"LoopCap {lc.name!r} is not a declared State field; "
                    "the loop counter must be a State field so the checkpoint node can read and "
                    "increment it (I6). Add a field like: FieldSpec(name={lc.name!r}, annotation='int', default='0')."
                )
            ann = state_field_map[lc.name].replace(" ", "")
            # Accept bare "int" and the nullable "int|None" form.
            if ann not in ("int", "int|None"):
                raise ValueError(
                    f"LoopCap {lc.name!r} has State annotation {state_field_map[lc.name]!r}; "
                    "a loop counter must be annotated 'int' or 'int | None' so it can be incremented."
                )

        # Dichotomy: a head makes one LLM call and goes to exactly one next node.
        # Routing (more than one successor) belongs in a separate deterministic spine
        # node — so an LLM call never also does the path-picking. `compile` enforces
        # this by splitting decision heads into a head + a router.
        for n in self.nodes:
            if n.kind is NodeKind.HEAD and len(n.successors) != 1:
                raise ValueError(
                    f"head node {n.name!r} has {len(n.successors)} successors; a head goes to exactly one next "
                    "node. Put the routing in a separate spine node (head → router)."
                )

        # Phase 4 — no silent output drop.
        #
        # The renderer wires head outputs into State positionally (zip(writes, out_fields)).
        # Two invariants must hold to prevent silent data loss:
        #
        # (a) Cardinality: len(writes) must not exceed len(agent.output.fields).
        #     If it did, the zip would silently truncate — fields beyond the agent's output
        #     would never be set.  Reject it here as a design-time error.
        #
        # (b) Non-terminal heads with a populated contract must declare at least one write.
        #     A non-terminal head with an empty writes list AND a non-empty reads/produces
        #     contract has done LLM work that goes nowhere — the output is silently dropped.
        #     Gate on (node.reads or node.produces) so hand-authored heads that omit the
        #     contract entirely (work-graph Plan/Compose/Synthesize etc.) stay valid, and
        #     a terminal head (sole successor == End) with empty writes is allowed (it may
        #     return its output directly via out_text).
        # A checkpoint node's loop counter is a CONTROL-plane write — the renderer
        # increments it in the loop guard, never binds it to the agent output — so it
        # must not count against the head's output fields in the cardinality check below.
        counter_names = {lc.name for lc in self.loop_caps}
        agent_map = {a.name: a for a in self.agents}
        for n in self.nodes:
            if n.kind is not NodeKind.HEAD:
                continue
            agent = agent_map.get(n.agent or "")  # type: ignore[arg-type]
            if agent is None:
                continue  # already caught by R5 above

            # (a) Too many DATA writes — zip would silently drop the excess. Control-plane
            # loop counters are excluded (they are incremented, not bound to the output).
            data_writes = [w for w in n.writes if w not in counter_names]
            n_out = len(agent.output.fields)
            if len(data_writes) > n_out:
                raise ValueError(
                    f"head node {n.name!r} declares {len(data_writes)} data writes "
                    f"but its agent {agent.name!r} output has only {n_out} field(s); "
                    "positional binding would silently drop the excess writes. "
                    "Either reduce writes to match the output fields, or add output fields."
                )

            # (b) Non-terminal head with a populated contract and no writes.
            is_terminal = len(n.successors) == 1 and n.successors[0] == END
            contract_populated = bool(n.reads or n.produces)
            if not is_terminal and contract_populated and not n.writes:
                raise ValueError(
                    f"head node {n.name!r} has a populated contract (reads/produces) "
                    "but declares no writes; its LLM output would be silently discarded. "
                    "Declare at least one write, or clear reads/produces if this node has "
                    "no data contract (leaving the output in the generated out_text path)."
                )

        return self

    def _has_cycle(self) -> bool:
        """Directed-cycle detection over the successor graph (ignores 'End')."""
        adj = {n.name: [s for s in n.successors if s != END] for n in self.nodes}
        WHITE, GREY, BLACK = 0, 1, 2
        color = {name: WHITE for name in adj}

        def visit(u: str) -> bool:
            color[u] = GREY
            for w in adj.get(u, []):
                if color.get(w) == GREY:
                    return True
                if color.get(w) == WHITE and visit(w):
                    return True
            color[u] = BLACK
            return False

        return any(color[name] == WHITE and visit(name) for name in adj)

    # --- convenience accessors -------------------------------------------------

    @property
    def head_nodes(self) -> list[NodeSpec]:
        return [n for n in self.nodes if n.kind is NodeKind.HEAD]

    @property
    def spine_nodes(self) -> list[NodeSpec]:
        return [n for n in self.nodes if n.kind is NodeKind.SPINE]

    @property
    def network_nodes(self) -> list[NodeSpec]:
        """Spine nodes that declare a network effect (run only behind the container sandbox)."""
        return [n for n in self.nodes if n.network]

    @property
    def has_network(self) -> bool:
        """True if any node reaches the network — the whole system must run behind the container."""
        return any(n.network for n in self.nodes)
