"""The static contract + safety gate — pure AST, no execution, never flaky.

Walks a filled node body and rejects: a ``ctx.state.<field>`` the node never
declared, a ``return <Node>()`` that is not a declared successor, dynamic state
access, forbidden imports (process control, raw files/sockets, clock/random),
code-eval builtins, and a broad ``except`` that swallows errors (a silent fallback —
``check_fail_loud``). A network client is allowed only when the node declared
``network=True`` (and therefore runs behind the container sandbox).

Runs after ``py_compile`` and before any execution, so most hallucinations are caught
for free, in-process, with a precise message. Ported from the ``forge`` dump. Leaf
worker: ``stdlib`` only, no graph engine.

**Trust boundary.** This purity gate is a *quality* gate and defense-in-depth — it keeps
generated bodies deterministic and free of obvious process/network/eval capability. It is
NOT the containment boundary and must not be relied on as one: static AST analysis cannot
soundly prove the absence of capability in Python (aliasing, dynamic dispatch, and
interpreter reflection make it undecidable in general — the gate closes the known vectors,
not all conceivable ones). The actual containment of LLM-authored code is the **sandbox**
(``sandbox.py`` — Podman, host-isolated), which is where that code executes.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _is_ctx_state(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "state"
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctx"
    )


def _is_ctx_deps(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "deps"
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctx"
    )


def _find_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for m in cls.body:
                if (
                    isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and m.name == method_name
                ):
                    return m
    return None


def _return_target(value: ast.expr) -> str | None:
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name):
            return func.id
    return None


# Never allowed in a node body, network node or not: process control, raw filesystem,
# raw sockets, serialization-exec, and non-determinism.
_ALWAYS_BANNED_MODULES = frozenset(
    {
        "os", "subprocess", "sys", "shutil", "socket", "pathlib", "io",
        "pickle", "marshal", "shelve", "ctypes", "multiprocessing", "threading", "signal",
        "time", "random", "uuid", "datetime", "secrets",
        # dynamic-import / interpreter-internals vectors (a body that reaches these can
        # re-obtain any banned module by name, defeating the static import check):
        "importlib", "builtins", "gc", "posix", "pty", "code", "codeop",
    }
)  # fmt: skip

# Attribute names that walk the object graph to re-derive banned capabilities without an
# import — the classic no-import escape ``().__class__.__bases__[0].__subclasses__()`` and
# ``__globals__``/``__builtins__`` reach-arounds. No legitimate node body needs these.
_BANNED_ATTRS = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__subclasshook__",
        "__base__",
    }
)

# Network clients: allowed ONLY in a node that declares network access.
_NETWORK_MODULES = frozenset(
    {
        "requests",
        "httpx",
        "urllib",
        "http",
        "aiohttp",
        "ftplib",
        "smtplib",
        "websockets",
    }
)

# Builtins that execute code, escape the sandbox, or read the world dynamically.
_BANNED_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
    }
)


def _module_violation(name: str, top: str, allow_network: bool) -> str:
    if top in _NETWORK_MODULES and not allow_network:
        return (
            f"imports {name!r} (network), but this node does not declare network access. Mark the node "
            "network=True — it then runs behind the container sandbox — or drop the import."
        )
    return (
        f"imports {name!r}, forbidden in a node body: no process control, raw files/sockets, or "
        "non-determinism (os/subprocess/socket/pathlib/clock/random)."
    )


def check_purity(
    file: Path, class_name: str, method_name: str, *, allow_network: bool = False
) -> list[str]:
    """Return safety violations in a node body — forbidden imports or unsafe calls."""
    banned = (
        _ALWAYS_BANNED_MODULES
        if allow_network
        else (_ALWAYS_BANNED_MODULES | _NETWORK_MODULES)
    )
    method = _find_method(ast.parse(file.read_text()), class_name, method_name)
    if method is None:
        return []
    # Names the body binds locally (assignment / for / with-as / comprehension / walrus
    # targets, and any parameters). A LOAD of such a name refers to the local, NOT the
    # builtin — so a body may legitimately use `input`/`open`/`vars` as a variable name.
    bound = {node.arg for node in ast.walk(method) if isinstance(node, ast.arg)} | {
        node.id
        for node in ast.walk(method)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    out: list[str] = []
    for n in ast.walk(method):
        if isinstance(n, ast.Import):
            for alias in n.names:
                top = alias.name.split(".")[0]
                if top in banned:
                    out.append(_module_violation(alias.name, top, allow_network))
        elif isinstance(n, ast.ImportFrom):
            top = (n.module or "").split(".")[0]
            if top in banned:
                out.append(_module_violation(n.module or "", top, allow_network))
        elif (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id in _BANNED_CALLS
            and n.id not in bound
        ):
            # Flag any LOAD of a banned builtin, not just a direct call — this catches
            # aliasing (`bad = eval; bad(...)`) as well as `eval(...)`. A name the body binds
            # locally is excluded (it shadows the builtin), so a legit variable named
            # `input`/`open`/`vars` is fine while aliasing (`bad = eval` — `eval` is not
            # bound) is still caught.
            out.append(
                f"references {n.id}, which is forbidden in a node body (code-eval / dynamic "
                "world access — unsafe or non-deterministic, even when aliased)."
            )
        elif isinstance(n, ast.Attribute) and n.attr in _BANNED_ATTRS:
            out.append(
                f"accesses {n.attr!r}, an interpreter-internals attribute used to escape the "
                "sandbox by re-deriving banned capabilities without an import — forbidden."
            )
    return out


def _is_broad_except(handler: ast.ExceptHandler) -> bool:
    """A bare ``except:`` or one catching ``Exception`` / ``BaseException`` (incl. tuples)."""
    t = handler.type
    if t is None:
        return True  # bare except:
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    for name in names:
        ident = name.id if isinstance(name, ast.Name) else getattr(name, "attr", "")
        if ident in ("Exception", "BaseException"):
            return True
    return False


def _reraises(handler: ast.ExceptHandler) -> bool:
    """True if the handler re-raises anywhere (re-raise or translate) — i.e. does not swallow."""
    return any(
        isinstance(n, ast.Raise) for stmt in handler.body for n in ast.walk(stmt)
    )


def check_fail_loud(file: Path, class_name: str, method_name: str) -> list[str]:
    """Reject a node body that SWALLOWS errors — a silent fallback.

    A broad ``except`` (bare, or catching ``Exception`` / ``BaseException``) that does not
    re-raise hides failures and lets the run continue on wrong/missing data. A narrow
    ``except <SpecificError>`` handling a genuinely expected exception is allowed; so is a
    broad ``except`` that re-raises (translating the error). This turns the dialect's "no
    bare catch that swallows" rule (L7) from a prompt suggestion into a build-time
    guarantee — the model proposes the body, this gate disposes of a swallowing one.
    """
    method = _find_method(ast.parse(file.read_text()), class_name, method_name)
    if method is None:
        return []
    violations: list[str] = []
    for handler in ast.walk(method):
        if (
            isinstance(handler, ast.ExceptHandler)
            and _is_broad_except(handler)
            and not _reraises(handler)
        ):
            violations.append(
                "swallows errors: a broad `except` that does not re-raise hides failures (a silent "
                "fallback). Catch a specific expected exception, or let it propagate — fail loud, "
                "never default or continue silently on error."
            )
    return violations


def check_deps_access(
    file: Path, class_name: str, method_name: str, *, allowed: set[str]
) -> list[str]:
    """Reject a node body that touches an UNDECLARED ``ctx.deps`` attribute.

    The body may read only the deps the rendered ``Deps`` actually provides (``allowed`` —
    derived from the generated ``deps.py`` so the gate and renderer cannot drift). A body
    that reaches for ``ctx.deps.repo_url`` when no such dependency exists is the classic
    "small model invents a handle / two nodes name it differently" wiring bug — caught here
    at build (model proposes, gate disposes), never at runtime. Dynamic access via
    ``getattr(ctx.deps, ...)`` is banned for the same reason it is on ``ctx.state``.
    """
    method = _find_method(ast.parse(file.read_text()), class_name, method_name)
    if method is None:
        return []
    out: list[str] = []
    flagged: set[str] = set()
    for n in ast.walk(method):
        if isinstance(n, ast.Attribute) and _is_ctx_deps(n.value):
            attr = n.attr
            if allowed and attr not in allowed and attr not in flagged:
                flagged.add(attr)
                out.append(
                    f"accesses ctx.deps.{attr}, which is not a declared dependency; available "
                    f"deps: {sorted(allowed)}. A node may not invent a dependency (endpoint, store, "
                    "config) the system does not provide — use only a declared dep, or the work does "
                    "not belong in this node."
                )
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("getattr", "setattr")
            and n.args
            and _is_ctx_deps(n.args[0])
        ):
            out.append(
                f"uses {n.func.id}() on ctx.deps; access declared dependencies directly, not dynamically"
            )
    return out


def check_map_item_body(file: Path, class_name: str, method_name: str) -> list[str]:
    """The simplified contract for a MAP_ITEM hole (an 'each' node's ``_map_one``).

    A map-item body is a PURE per-item transform — stricter and simpler than the node-body
    contract: it must USE the ``item`` parameter and RETURN a value, and it must not touch
    ``ctx`` (state/deps — the rendered scaffold owns them) or ``self`` (a pure function
    needs only its item). Purity and fail-loud are the shared gates; this is the data
    contract. Empty list = clean.
    """
    method = _find_method(ast.parse(file.read_text()), class_name, method_name)
    if method is None:
        return [f"method {class_name}.{method_name} not found"]

    violations: list[str] = []
    flagged: set[str] = set()
    uses_item = False
    returns_value = False
    for n in ast.walk(method):
        if isinstance(n, ast.Name):
            if n.id == "item" and isinstance(n.ctx, ast.Load):
                uses_item = True
            elif n.id == "ctx" and "ctx" not in flagged:
                flagged.add("ctx")
                violations.append(
                    "references ctx — a map-item body is a pure per-item transform with NO "
                    "ctx, state, or deps access at all (the surrounding scaffold owns them)"
                )
            elif n.id == "self" and "self" not in flagged:
                flagged.add("self")
                violations.append(
                    "references self — a map-item body is a pure function of its `item` "
                    "parameter; it may not touch the node instance"
                )
        elif isinstance(n, ast.Return) and n.value is not None:
            returns_value = True
    if not uses_item:
        violations.append(
            "never reads the `item` parameter — the body must transform the ONE item it is given"
        )
    if not returns_value:
        violations.append(
            "never returns a value — the body must return the transformed item"
        )
    return violations


def check_node_body(
    file: Path,
    class_name: str,
    method_name: str,
    *,
    reads: set[str],
    writes: set[str],
    successors: set[str],
) -> list[str]:
    """Return contract violations in ``class_name.method_name``. Empty list = clean.

    Enforces the node's data contract as a build-time guarantee (not a prompt hope):
    a body reads/writes only declared fields, returns only declared successors, accesses
    state per-field (no bulk dump, no dynamic get/set), AND — coverage — actually reads
    every input it declared and sets every output it declared. The latter two catch a body
    that quietly drops its inputs or never produces its output (a faked node).
    """
    method = _find_method(ast.parse(file.read_text()), class_name, method_name)
    if method is None:
        return [f"method {class_name}.{method_name} not found"]

    # Bulk state access defeats the per-field contract (it would read undeclared fields),
    # so it is banned alongside getattr/setattr — which is also what makes the read/write
    # coverage below exact: every legitimate access is a direct `ctx.state.<field>`.
    _BULK_ACCESSORS = {"model_dump", "model_dump_json", "dict", "json", "copy"}

    violations: list[str] = []
    reads_seen: set[str] = set()
    writes_seen: set[str] = set()
    for n in ast.walk(method):
        if isinstance(n, ast.Attribute) and _is_ctx_state(n.value):
            field = n.attr
            if field in _BULK_ACCESSORS:
                violations.append(
                    f"accesses ctx.state in bulk via .{field}(); read declared fields directly "
                    "(ctx.state.<field>) — bulk access reads undeclared fields and defeats the contract"
                )
            elif isinstance(n.ctx, ast.Store):
                writes_seen.add(field)
                if writes and field not in writes:
                    violations.append(
                        f"writes ctx.state.{field}, which is not a declared output; declared writes: {sorted(writes)}"
                    )
            else:
                reads_seen.add(field)
                if field not in reads | writes:
                    violations.append(
                        f"reads ctx.state.{field}, which is not a declared input; declared reads: {sorted(reads | writes)}"
                    )
        if isinstance(n, ast.Return) and n.value is not None:
            target = _return_target(n.value)
            if target is not None and successors and target not in successors:
                violations.append(
                    f"returns {target}(), which is not a declared successor; declared successors: {sorted(successors)}"
                )
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("getattr", "setattr")
        ):
            if n.args and _is_ctx_state(n.args[0]):
                violations.append(
                    f"uses {n.func.id}() on ctx.state; access declared fields directly, not dynamically"
                )

    # Coverage: a declared input must actually be read, a declared output actually set.
    # A node that ignores an input it declared, or never produces an output it declared,
    # is faking — surface it. (Setting/reading in only one branch is a runtime concern the
    # sandbox contract-run catches; here we guarantee the field is touched at all.)
    for field in sorted(reads - reads_seen):
        violations.append(
            f"declares it reads ctx.state.{field} but the body never reads it — use the input, "
            "or it should not be a declared read"
        )
    for field in sorted(writes - writes_seen):
        violations.append(
            f"declares it writes ctx.state.{field} but the body never sets it — the node must "
            "produce its declared output"
        )

    return violations
