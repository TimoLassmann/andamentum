"""Deterministic rendering of a `SystemSpec` into a runnable package.

This enacts the recipe's Assembly law at the meta level: an agent (or a human)
fills the typed `SystemSpec`; *this code* — no LLM — assembles the package. What
the renderer can derive mechanically it writes in full; what is genuine business
logic it leaves as a clearly-marked `NotImplementedError` hole (which the
constitution sanctions as the way to mark unimplemented work).

Fully generated:
  - all models (input, entities, agent outputs, State) as Pydantic `BaseModel`s,
    with closed vocabularies rendered as `Literal[...]`
  - prompt constants and per-head user-prompt builders
  - `Deps`, the graph assembly, the `run` entry with input validation
  - a `__main__.py` CLI launcher (`python -m <name> "<text>" --model <id>`) — an
    adapter over the `run` entry, so the system is runnable without a hand-written driver
  - single-successor head nodes (call `run_head`, return the one successor)
  - a stub test that imports the package, checks the graph assembles, and drives
    each single-successor head through the `agent_overrides` seam

Left as holes (NotImplementedError, with guidance):
  - spine-node bodies (real logic: load, query, count, write)
  - multi-successor routing (head or spine)
  - HumanGate `decide` (route on the human's answer)

The output is RunEndT = `str` for the MVP (most recipe systems return a text
answer); a future spec field can make this explicit.
"""

from __future__ import annotations

from pathlib import Path

from .spec import (
    END,
    AgentSpec,
    FieldSpec,
    LoopCap,
    ModelSpec,
    NodeControl,
    NodeKind,
    NodeSpec,
    StateSpec,
    SystemSpec,
)

# The recipe's load-bearing laws, stamped into each generated package's docstring so
# the constitution travels with the code (forge's "librarian", distilled to the three
# that matter at read-time). They line up with the agentic-dialect (L6 / L2 / L3 + R1).
_DICHOTOMY = "Flow control lives in the graph, never in the model (deterministic spine, small heads)."
_PURITY = "A node body does one job; real business logic is a NotImplementedError hole, never a fake stub."
_STATE_RULE = "State carries only signals (counters, flags, IDs); durable entities live in a store (Rule R1)."

# --- field / model rendering ----------------------------------------------------


def _num(v: float) -> str:
    """Render a constraint number without a trailing .0 when it is whole."""
    return str(int(v)) if float(v).is_integer() else str(v)


def _default_lit(f: FieldSpec) -> str | None:
    """The field's default as source text, or None if the field is required."""
    if f.default is not None:
        return f.default
    if f.optional:
        return "None"
    return None


def _annotation(f: FieldSpec) -> str:
    """The Python annotation for a field — closed vocabularies become Literal,
    and any field whose default is None is made nullable."""
    if f.enum_values:
        lit = "Literal[" + ", ".join(repr(v) for v in f.enum_values) + "]"
        ann = f"list[{lit}]" if f.annotation.startswith("list[") else lit
    else:
        ann = f.annotation
    if _default_lit(f) == "None" and "None" not in ann:
        ann = f"{ann} | None"
    return ann


def _field_rhs(f: FieldSpec) -> str | None:
    """The right-hand side of a model field: a Field(...) call, a bare default, or
    None when the field is required with no metadata."""
    kwargs: list[str] = []
    if f.description:
        kwargs.append(f"description={f.description!r}")
    c = f.constraints
    for key in ("ge", "le", "gt", "lt"):
        v = getattr(c, key)
        if v is not None:
            kwargs.append(f"{key}={_num(v)}")
    for key in ("min_length", "max_length"):
        v = getattr(c, key)
        if v is not None:
            kwargs.append(f"{key}={v}")
    if c.pattern is not None:
        kwargs.append(f"pattern={c.pattern!r}")

    default_lit = _default_lit(f)

    if kwargs:
        if default_lit is not None:
            return "Field(default=" + default_lit + ", " + ", ".join(kwargs) + ")"
        return "Field(" + ", ".join(kwargs) + ")"
    return default_lit


def _field_line(f: FieldSpec) -> str:
    rhs = _field_rhs(f)
    line = f"    {f.name}: {_annotation(f)}"
    return f"{line} = {rhs}" if rhs is not None else line


def _render_model(m: ModelSpec) -> str:
    doc = f'    """{m.description}"""\n' if m.description else ""
    body = "\n".join(_field_line(f) for f in m.fields)
    return f"class {m.name}(BaseModel):\n{doc}{body}\n"


def _render_state(name: str, state: StateSpec, extra: list[FieldSpec]) -> str:
    fields = list(state.fields)
    have = {f.name for f in fields}
    for f in extra:
        if f.name not in have:
            fields.append(f)
    body = "\n".join(_field_line(f) for f in fields) if fields else "    pass"
    return (
        f'class {name}(BaseModel):\n    """Run-scoped signals (Rule R1)."""\n{body}\n'
    )


# --- node rendering -------------------------------------------------------------


def _return_type(successors: list[str], run_end_type: str) -> str:
    parts = [f"End[{run_end_type}]" if s == END else f'"{s}"' for s in successors]
    return parts[0] if len(parts) == 1 else "Union[" + ", ".join(parts) + "]"


def _ctx(state_name: str, deps_name: str) -> str:
    return f"GraphRunContext[{state_name}, {deps_name}]"


def _humanize(field: str) -> str:
    """A snake_case state-field name as a readable label, e.g. 'main_ideas' → 'Main ideas'."""
    return field.replace("_", " ").strip().capitalize()


def _agent_by_name(spec: SystemSpec, name: str) -> AgentSpec:
    return next(a for a in spec.agents if a.name == name)


def _reads_note(node: NodeSpec) -> str:
    return f" Reads: {', '.join(node.reads)}." if node.reads else ""


def _contract_note(node: NodeSpec) -> str:
    """Reads/writes named in plain text for the hole message (the durable artifact;
    the builder gets the same contract typed via node_contract)."""
    parts: list[str] = []
    if node.reads:
        parts.append(f"Reads: {', '.join(node.reads)}.")
    if node.writes:
        parts.append(f"Writes: {', '.join(node.writes)}.")
    return (" ".join(parts) + " ") if parts else ""


def _state_writes(node: NodeSpec, agent: AgentSpec) -> str:
    """Deterministic `ctx.state.<write> = out.<field>` lines — the renderer wires a
    head's output into State (pairing writes to output fields positionally) so the
    head's result is never silently dropped. No LLM, no hole."""
    out_fields = [f.name for f in agent.output.fields]
    return "".join(
        f"        ctx.state.{w} = out.{f}\n" for w, f in zip(node.writes, out_fields)
    )


def _render_node(
    spec: SystemSpec,
    node: NodeSpec,
    state_name: str,
    deps_name: str,
    gate_keys: dict[str, str],
    run_end_type: str,
) -> str:
    rt = _return_type(node.successors, run_end_type)
    ctx = _ctx(state_name, deps_name)
    doc = f'    """{node.purpose}"""\n' if node.purpose else ""
    head = f"@dataclass\nclass {node.name}"
    succ_list = ", ".join(node.successors)

    # HumanGate nodes (those referenced by a hitl gate).
    if node.name in gate_keys:
        return (
            f"{head}(HumanGate[{state_name}, {deps_name}, {run_end_type}]):\n"
            f"{doc}"
            f"    gate_key = {gate_keys[node.name]!r}\n\n"
            f"    def prompt(self, ctx: {ctx}) -> str:\n"
            f"        return {node.purpose!r}\n\n"
            f"    def decide(self, answer: str, ctx: {ctx}) -> {rt}:\n"
            f"        raise NotImplementedError(\n"
            f'            "Route on the human\'s answer for {node.name} among: {succ_list}."\n'
            f"        )\n"
        )

    is_head = node.kind is NodeKind.HEAD
    single = len(node.successors) == 1

    if is_head:
        agent = _agent_by_name(spec, node.agent)  # type: ignore[arg-type]
        call = (
            f'        out = await run_head(ctx.deps, "{agent.name}", {agent.output.name}, '
            f"{agent.name.upper()}_PROMPT, _user_prompt_{agent.name}(ctx))\n"
        )
        writes = _state_writes(node, agent)
        if single:
            succ = node.successors[0]
            if succ == END:
                # Return the head's output by its DECLARED field name, decided here at
                # render time — deterministic, no runtime guessing, no fallback. (We used
                # to emit a best-effort `out_text` that stringified the object when no
                # string field was found; that is a silent fallback and is gone.)
                primary_field = agent.output.fields[0].name
                tail = f"        return End(out.{primary_field})\n"
            else:
                tail = f"        return {succ}()\n"
            return (
                f"{head}(BaseNode[{state_name}, {deps_name}, {run_end_type}]):\n{doc}"
                f"    async def run(self, ctx: {ctx}) -> {rt}:\n{call}{writes}{tail}"
            )
        # multi-successor head: writes are wired deterministically; routing is the hole.
        return (
            f"{head}(BaseNode[{state_name}, {deps_name}, {run_end_type}]):\n{doc}"
            f"    async def run(self, ctx: {ctx}) -> {rt}:\n{call}{writes}"
            f"        raise NotImplementedError(\n"
            f'            "Route on `out` for {node.name} among: {succ_list}.{_reads_note(node)}"\n'
            f"        )\n"
        )

    # CHECKPOINT spine node: emit the deterministic loop-bounding body.
    # The renderer fills this mechanically — the guard and counter increment are not
    # business logic, they are recipe infrastructure (I6). What the counter is named,
    # which successor is the forward exit and which is the loop-back, are all
    # derivable from the spec; no LLM and no hole.
    if node.control is NodeControl.CHECKPOINT:
        # Find the LoopCap whose name matches the convention set in compile.py:
        # counter_name = f"{node.name.lower()}_loops"
        lc: LoopCap | None = next(
            (c for c in spec.loop_caps if c.name == f"{node.name.lower()}_loops"), None
        )
        if lc is None:
            raise ValueError(
                f"CHECKPOINT node {node.name!r} has no matching LoopCap in spec.loop_caps "
                f"(expected name {node.name.lower()!r}_loops). "
                "A checkpoint node requires a LoopCap so the renderer can emit a terminating body."
            )
        if len(node.successors) < 2:
            raise ValueError(
                f"CHECKPOINT node {node.name!r} has only {len(node.successors)} successor(s); "
                "a checkpoint needs at least 2: successors[0]=forward (backbone exit), "
                "successors[1]=loop-back (area first node)."
            )
        forward_node = node.successors[0]
        loopback_node = node.successors[1]
        forward_return = (
            "End(ctx.state.request)" if forward_node == END else f"{forward_node}()"
        )
        loopback_return = f"{loopback_node}()"
        counter = lc.name
        return (
            f"{head}(BaseNode[{state_name}, {deps_name}, {run_end_type}]):\n{doc}"
            f"    async def run(self, ctx: {ctx}) -> {rt}:\n"
            f"        if loop_allowed(ctx.state.{counter}, ctx.deps.loop_cap):\n"
            f"            ctx.state.{counter} += 1\n"
            f"            return {loopback_return}\n"
            f"        return {forward_return}\n"
        )

    # spine node: business logic is a hole, with its full contract in the message.
    purpose = f"Purpose: {node.purpose} " if node.purpose else ""
    nxt = (
        f"Then return {node.successors[0]}()."
        if single and node.successors[0] != END
        else f"Return among: {succ_list}."
    )
    return (
        f"{head}(BaseNode[{state_name}, {deps_name}, {run_end_type}]):\n{doc}"
        f"    async def run(self, ctx: {ctx}) -> {rt}:\n"
        f'        raise NotImplementedError("Spine node {node.name!r}. {purpose}{_contract_note(node)}{nxt}")\n'
    )


# --- file assembly --------------------------------------------------------------


def _models_py(spec: SystemSpec, state_name: str, input_extra: list[FieldSpec]) -> str:
    parts = [
        '"""Models — generated by forge. Edit the SystemSpec, not this file."""',
        "from __future__ import annotations",
        "",
        "from typing import Literal  # noqa: F401  (used when a field has a closed vocabulary)",
        "",
        "from pydantic import BaseModel, Field",
        "",
        "",
        _render_model(spec.input.model),
    ]
    for e in spec.entities:
        parts.append("")
        parts.append(_render_model(e.model))
    for a in spec.agents:
        parts.append("")
        parts.append(_render_model(a.output))
    parts.append("")
    parts.append(_render_state(state_name, spec.state, input_extra))
    return "\n".join(parts) + "\n"


def _prompts_py(spec: SystemSpec) -> str:
    lines = ['"""Head prompts — generated by forge. Refine the wording freely."""', ""]
    for a in spec.agents:
        lines.append(f"{a.name.upper()}_PROMPT = {a.prompt!r}")
        lines.append("")
    return "\n".join(lines)


def _deps_py(spec: SystemSpec, deps_name: str) -> str:
    """The frozen Deps dataclass: the model handle, the test seam, and the loop cap.

    ``agent_overrides`` is the dialect test seam — a stub keyed by agent name swapped
    in so the whole graph runs with no live model. Typed ``dict[str, object]`` (not
    ``Any``) so the generated package itself passes the dialect's L7 gate.
    """
    return (
        '"""Dependencies — generated by forge: model handle, test seam, caps, store."""\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass, field\n\n"
        "from andamentum.forge.runtime import Store\n\n\n"
        "@dataclass(frozen=True)\n"
        f"class {deps_name}:\n"
        '    """Injected, never serialized (Rule R2): model handle, test seam, caps, store."""\n'
        '    model: str = "test"\n'
        "    #: agent name -> stub with an `.output`; swap in to run without a live model.\n"
        "    agent_overrides: dict[str, object] = field(default_factory=dict)\n"
        f"    loop_cap: int = {max((c.limit for c in spec.loop_caps), default=2)}\n"
        "    #: cross-run memory Port (dialect L1). Default in-memory (forgets at exit); the\n"
        "    #: run entry rebinds it to Store(path) for durable rung-2 memory. Reach it via\n"
        "    #: ctx.deps.store — a node never sees a path. Five ops: add/get/list/remove.\n"
        "    store: Store = field(default_factory=Store)\n"
    )


def _nodes_py(
    spec: SystemSpec, state_name: str, deps_name: str, gate_keys: dict[str, str]
) -> str:
    run_end_type = spec.run_end_type
    model_names = (
        [spec.input.model.name]
        + [e.model.name for e in spec.entities]
        + [a.output.name for a in spec.agents]
        + [state_name]
    )
    prompt_names = [f"{a.name.upper()}_PROMPT" for a in spec.agents]
    # Build a map from agent name to the head node that uses it (first match wins for
    # the rare case of shared agents) so we can scope each user-prompt to that node's
    # declared reads — avoids leaking unrelated state fields into the prompt.
    _agent_reads: dict[str, list[str]] = {}
    for _n in spec.nodes:
        if _n.kind is NodeKind.HEAD and _n.agent and _n.agent not in _agent_reads:
            _agent_reads[_n.agent] = list(_n.reads)

    def _user_prompt_body(agent_name: str) -> str:
        # Assemble a readable, labelled user message from the head's declared inputs — one
        # "Label:\n<value>" block per read, blank-line separated. NOT a JSON dump: the model
        # reasons better over framed prose, and the node's job lives in its system prompt.
        reads = _agent_reads.get(agent_name, [])
        if not reads:
            # A head with no declared inputs: its system prompt (the node's job) drives it.
            return '    return ""'
        pieces = ", ".join(f'f"{_humanize(r)}:\\n{{ctx.state.{r}}}"' for r in reads)
        return f'    return "\\n\\n".join([{pieces}])'

    user_prompts = "\n\n".join(
        f"def _user_prompt_{a.name}(ctx: {_ctx(state_name, deps_name)}) -> str:\n"
        f"    # The {a.name} head's inputs, labelled, as its user message (refine the wording freely).\n"
        f"{_user_prompt_body(a.name)}"
        for a in spec.agents
    )
    nodes = "\n\n".join(
        _render_node(spec, n, state_name, deps_name, gate_keys, run_end_type)
        for n in spec.nodes
    )
    # Omit the prompts import entirely when there are no agents (all-spine systems);
    # preserve exactly two blank lines before the first function definition either way.
    prompts_import = (
        f"from .prompts import {', '.join(prompt_names)}\n" if prompt_names else ""
    )
    # When there are no agents, user_prompts is empty and we must not emit the trailing
    # "\n\n\n" separator (which would produce 3+ blank lines and fail ruff E303).
    nodes_section = f"{user_prompts}\n\n\n{nodes}" if user_prompts else nodes
    # `Union` only when some node has multiple successors (a router / checkpoint); `Any`
    # is no longer needed (the out_text best-effort helper is gone — terminal heads return
    # their declared output field by name, deterministically, with no fallback).
    typing_import = (
        "from typing import Union\n"
        if any(len(n.successors) > 1 for n in spec.nodes)
        else ""
    )
    return (
        '"""Nodes — generated by forge. Heads are complete; spine bodies, routing,\n'
        'and gate decisions are NotImplementedError holes for you to fill."""\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n"
        f"{typing_import}\n"
        "from pydantic_graph import BaseNode, End, GraphRunContext\n\n"
        "from andamentum.forge.runtime import loop_allowed, run_head  # noqa: F401\n\n"
        f"from .deps import {deps_name}\n"
        f"from .models import {', '.join(model_names)}\n"
        f"{prompts_import}\n\n"
        f"{nodes_section}"
    )


def _graph_py(spec: SystemSpec, state_name: str, deps_name: str) -> str:
    """The generated orchestration file: Graph assembly + the dialect entry function.

    A standard one-shot ``run_<name>`` driver (``graph.run()``) — no bespoke runtime.
    The input is validated at the door (Input law) and threaded into State.
    """
    node_names = [n.name for n in spec.nodes]
    primary = spec.input.primary_text_field
    run_end_type = spec.run_end_type
    rules = (
        "\n".join(f"    # - {r}" for r in spec.input.validation_rules)
        or "    # (no extra rules declared)"
    )
    return (
        '"""Graph assembly + entry — generated by forge (the agentic-dialect orchestration file)."""\n'
        "from __future__ import annotations\n\n"
        "from pydantic_graph import Graph\n\n"
        "from andamentum.forge.runtime import Store\n\n"
        f"from .deps import {deps_name}\n"
        f"from .models import {spec.input.model.name}, {state_name}\n"
        f"from . import nodes as _n\n\n\n"
        f"graph = Graph(nodes=[{', '.join('_n.' + n for n in node_names)}])\n\n\n"
        f"def validate_input(text: str) -> {spec.input.model.name}:\n"
        '    """Validate at the door (Input law). Rules from the spec:\n'
        f"{rules}\n"
        '    """\n'
        "    if not text or not text.strip():\n"
        '        raise ValueError("input must not be blank")\n'
        f"    return {spec.input.model.name}({primary}=text)\n\n\n"
        f"async def run_{spec.name}(\n"
        f"    text: str, *, model: str, store: str | None = None\n"
        f") -> {run_end_type}:\n"
        '    """Validate the input, build initial State + Deps, and run the graph to End.\n\n'
        "    ``store`` is a path to a durable database for cross-run memory (a stateful\n"
        "    function); ``None`` (the default) means an ephemeral in-memory store, so the\n"
        '    function behaves statelessly. The path is resolved to a Store here, once."""\n'
        f"    data = validate_input(text)\n"
        f"    state = {state_name}({primary}=data.{primary})\n"
        f"    deps = {deps_name}(model=model, store=Store(store))\n"
        f"    out = await graph.run(_n.{spec.entry_node}(), state=state, deps=deps)\n"
        f"    return out.output\n"
    )


def _init_py(spec: SystemSpec, state_name: str, deps_name: str) -> str:
    entry = f"run_{spec.name}"
    return (
        f'"""{spec.name} — generated by forge from a SystemSpec.\n\n'
        f"{spec.description}\n\n"
        "CONSTITUTION — this system follows the Agent Graph Recipe / agentic dialect:\n"
        f"- {_DICHOTOMY}\n"
        f"- {_PURITY}\n"
        f"- {_STATE_RULE}\n"
        "The originating spec is frozen in spec.json; tests/test_recipe.py re-validates\n"
        "it (so an edit that breaks the recipe fails CI). Fill the NotImplementedError\n"
        'holes in nodes.py; everything else — models, prompts, graph wiring — is done.\n"""\n'
        f"from .graph import graph, validate_input, {entry}\n"
        f"from .deps import {deps_name}\n"
        f"from .models import {spec.input.model.name}, {state_name}\n\n"
        f'__all__ = ["graph", "{entry}", "validate_input", "{deps_name}", '
        f'"{spec.input.model.name}", "{state_name}"]\n'
    )


def _main_py(spec: SystemSpec) -> str:
    """A tiny CLI launcher so the package runs as ``python -m <name> "<text>" --model <id>``.

    An *adapter*, not an engine file (dialect): it imports the package's ``run_<name>``
    entry and prints the result — flow control stays in the graph, no engine import here.
    ``--model`` is required (no hidden default, no env var — the andamentum convention);
    the input is validated at the door by ``run_<name>`` and fails loud on blank/invalid.
    """
    entry = f"run_{spec.name}"
    primary = next(
        f for f in spec.input.model.fields if f.name == spec.input.primary_text_field
    )
    help_text = (primary.description or "the input text").replace('"', "'")
    return (
        f'"""CLI launcher — run this system as `python -m {spec.name} "<text>" --model <id>`.\n\n'
        "Generated by forge: an adapter over the package\\'s run entry (the graph drives flow).\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import asyncio\n"
        "import sys\n\n"
        f"from .graph import {entry}\n\n\n"
        "def main(argv: list[str] | None = None) -> int:\n"
        f'    parser = argparse.ArgumentParser(prog="{spec.name}")\n'
        f'    parser.add_argument("text", help="{help_text}")\n'
        "    parser.add_argument(\n"
        '        "--model",\n'
        "        required=True,\n"
        '        help="pydantic-ai model id (e.g. ollama:..., anthropic:...) — required, no default",\n'
        "    )\n"
        "    parser.add_argument(\n"
        '        "--store",\n'
        "        default=None,\n"
        '        help="path to a database file for cross-run memory; omitted means in-memory (the function forgets)",\n'
        "    )\n"
        "    args = parser.parse_args(argv)\n"
        "    try:\n"
        f"        result = asyncio.run({entry}(args.text, model=args.model, store=args.store))\n"
        "    except ValueError as e:\n"
        '        print(f"Error: {e}", file=sys.stderr)\n'
        "        return 1\n"
        "    print(result)\n"
        "    return 0\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


def _recipe_test_py() -> str:
    """A generated test that re-validates the frozen spec against the recipe — so a
    downstream edit that breaks the contract is caught, and the constitution travels
    with the code as an executable check, not just a comment."""
    return (
        '"""This system still obeys the Agent Graph Recipe — generated by forge.\n'
        'Re-validates the frozen SystemSpec; fails if an edit drifted from the recipe."""\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "from andamentum.forge.spec import SystemSpec\n\n\n"
        "def test_system_obeys_the_recipe() -> None:\n"
        "    spec_json = (Path(__file__).resolve().parent.parent / 'spec.json').read_text()\n"
        "    SystemSpec.model_validate(json.loads(spec_json))  # raises if it drifted\n"
    )


def _test_py(spec: SystemSpec) -> str:
    """The assembly smoke test: the package imports and the graph assembles with all
    nodes. (The dialect topology test is the deeper structural check; this is the
    fast green-on-render guard.)"""
    node_names = sorted(n.name for n in spec.nodes)
    return (
        '"""Generated assembly test — the package imports and the graph assembles."""\n'
        "from __future__ import annotations\n\n"
        f"from {spec.name}.graph import graph\n\n\n"
        "def test_graph_assembles():\n"
        f"    assert sorted(d.node.__name__ for d in graph.node_defs.values()) == {node_names!r}\n"
    )


def _sample_literal(f: FieldSpec) -> str:
    """A Python-literal sample for a head-output field, for the smoke-test stub."""
    if f.enum_values:
        return repr(f.enum_values[0])
    ann = f.annotation.replace(" ", "")
    return {
        "str": "'sample'",
        "int": "1",
        "float": "1.0",
        "bool": "True",
        "list[str]": "['sample']",
        "list[int]": "[1]",
        "set[str]": "{'sample'}",
    }.get(ann, "'sample'")


def _smoke_test_py(spec: SystemSpec, state_name: str, deps_name: str) -> str:
    """A generated end-to-end smoke test: stub every head, drive the graph to End.

    This is what the audit (stage 4) executes in the sandbox — the assembled graph runs
    with no live model, so a node body that crashes or drops its input is caught.
    """
    primary = spec.input.primary_text_field
    out_models = [a.output.name for a in spec.agents]
    model_imports = ", ".join([state_name, *out_models])

    override_lines: list[str] = []
    for a in spec.agents:
        required = [f for f in a.output.fields if not f.optional and f.default is None]
        args = ", ".join(f"{f.name}={_sample_literal(f)}" for f in required)
        override_lines.append(f"        {a.name!r}: _Stub({a.output.name}({args})),")
    overrides_block = (
        "\n".join(override_lines)
        if override_lines
        else "        # (no heads — all-spine system)"
    )

    stub_class = (
        "class _Stub:\n"
        '    """An agent_overrides stub: exposes a fixed `.output` (run_head reads it)."""\n'
        "    def __init__(self, output: object) -> None:\n"
        "        self.output = output\n\n\n"
        if spec.agents
        else ""
    )
    return (
        '"""Generated smoke test — the graph runs end-to-end with stub agents."""\n'
        "from __future__ import annotations\n\n"
        "import asyncio\n\n"
        "from andamentum.forge.runtime import Store\n\n"
        f"from {spec.name} import nodes as _n\n"
        f"from {spec.name}.deps import {deps_name}\n"
        f"from {spec.name}.graph import graph\n"
        f"from {spec.name}.models import {model_imports}\n\n\n"
        f"{stub_class}"
        "def test_smoke_runs_end_to_end() -> None:\n"
        "    overrides = {\n"
        f"{overrides_block}\n"
        "    }\n"
        "    # Store(None) is the in-memory store: the same object production uses, so the\n"
        "    # smoke exercises the real load/save paths offline (no file, no Ollama).\n"
        f'    deps = {deps_name}(model="smoke", agent_overrides=overrides, store=Store(None))\n'
        f'    state = {state_name}({primary}="smoke test request")\n'
        f"    out = asyncio.run(graph.run(_n.{spec.entry_node}(), state=state, deps=deps))\n"
        "    assert out is not None\n"
    )


def render(spec: SystemSpec, dest: Path) -> list[Path]:
    """Render ``spec`` into an importable package under ``dest/<spec.name>/``.

    Deterministic — the recipe's Assembly law at the meta level: an agent (or a
    human) fills the typed ``SystemSpec``; *this code*, no LLM, assembles the
    package. Returns the list of files written. ``dest`` is created if needed.
    """
    state_name = "".join(p.capitalize() for p in spec.name.split("_")) + "State"
    deps_name = "".join(p.capitalize() for p in spec.name.split("_")) + "Deps"
    # v1 renders consequential nodes as spine holes (a human-approval step the author
    # fills), not the persistence-backed HumanGate machinery — so no gate keys.
    gate_keys: dict[str, str] = {}

    # The input's primary text must reach the entry node; thread it through state.
    primary = next(
        f for f in spec.input.model.fields if f.name == spec.input.primary_text_field
    )
    input_extra = [FieldSpec(name=primary.name, annotation="str", default="''")]

    pkg = dest / spec.name
    tests = pkg / "tests"
    pkg.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    files = {
        pkg / "models.py": _models_py(spec, state_name, input_extra),
        pkg / "prompts.py": _prompts_py(spec),
        pkg / "deps.py": _deps_py(spec, deps_name),
        pkg / "nodes.py": _nodes_py(spec, state_name, deps_name, gate_keys),
        pkg / "graph.py": _graph_py(spec, state_name, deps_name),
        pkg / "__init__.py": _init_py(spec, state_name, deps_name),
        pkg / "__main__.py": _main_py(spec),
        pkg / "spec.json": spec.model_dump_json(indent=2),
        tests / "__init__.py": "",
        tests / "test_graph.py": _test_py(spec),
        tests / "test_recipe.py": _recipe_test_py(),
        tests / "test_smoke.py": _smoke_test_py(spec, state_name, deps_name),
    }
    written: list[Path] = []
    for path, content in files.items():
        path.write_text(content)
        written.append(path)
    return written
