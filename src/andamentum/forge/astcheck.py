"""The static contract + safety gate — pure AST, no execution, never flaky.

Walks a filled node body and rejects: a ``ctx.state.<field>`` the node never
declared, a ``return <Node>()`` that is not a declared successor, dynamic state
access, forbidden imports (process control, raw files/sockets, clock/random),
code-eval builtins, and a broad ``except`` that swallows errors (a silent fallback —
``check_fail_loud``). A network client is allowed only when the node declared
``network=True`` (and therefore runs behind the container sandbox).

Runs after ``py_compile`` and before any execution, so most hallucinations and all
unsafe code are caught for free, in-process, with a precise message. Ported from the
``forge`` dump. Leaf worker: ``stdlib`` only, no graph engine.
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
    }
)  # fmt: skip

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
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in _BANNED_CALLS
        ):
            out.append(
                f"calls {n.func.id}(), which is forbidden in a node body (unsafe or non-deterministic)."
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


def check_node_body(
    file: Path,
    class_name: str,
    method_name: str,
    *,
    reads: set[str],
    writes: set[str],
    successors: set[str],
) -> list[str]:
    """Return contract violations in ``class_name.method_name``. Empty list = clean."""
    method = _find_method(ast.parse(file.read_text()), class_name, method_name)
    if method is None:
        return [f"method {class_name}.{method_name} not found"]

    violations: list[str] = []
    for n in ast.walk(method):
        if isinstance(n, ast.Attribute) and _is_ctx_state(n.value):
            field = n.attr
            if isinstance(n.ctx, ast.Store):
                if writes and field not in writes:
                    violations.append(
                        f"writes ctx.state.{field}, which is not a declared output; declared writes: {sorted(writes)}"
                    )
            elif field not in reads | writes:
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

    return violations
