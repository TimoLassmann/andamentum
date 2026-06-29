"""The fail-loud gate rejects error-swallowing node bodies (no silent fallbacks).

A suggestion in the build prompt ("fail loud, no silent fallbacks") is a hope; this gate
is the guarantee — it deterministically rejects a body that swallows errors, so a system
forge builds can't quietly continue on wrong data.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.forge.astcheck import (
    check_deps_access,
    check_fail_loud,
    check_node_body,
)


def _write(tmp_path: Path, body: str) -> Path:
    src = "class N:\n    async def run(self, ctx):\n" + body
    f = tmp_path / "n.py"
    f.write_text(src)
    return f


def _contract(
    f: Path, *, reads: set[str], writes: set[str], successors: set[str]
) -> list[str]:
    return check_node_body(
        f, "N", "run", reads=reads, writes=writes, successors=successors
    )


def test_unread_declared_input_is_flagged(tmp_path: Path) -> None:
    # Declares it reads `a` but the body ignores it — a dropped input (faking).
    f = _write(tmp_path, "        ctx.state.b = 'x'\n        return End('done')\n")
    issues = _contract(f, reads={"a"}, writes={"b"}, successors={"End"})
    assert any("never reads" in v for v in issues), issues


def test_unset_declared_output_is_flagged(tmp_path: Path) -> None:
    # Declares it writes `b` but never sets it — the node produces no output.
    f = _write(tmp_path, "        _ = ctx.state.a\n        return End('done')\n")
    issues = _contract(f, reads={"a"}, writes={"b"}, successors={"End"})
    assert any("never sets" in v for v in issues), issues


def test_bulk_state_access_is_flagged(tmp_path: Path) -> None:
    # model_dump reads undeclared fields and defeats the per-field contract.
    f = _write(
        tmp_path,
        "        data = ctx.state.model_dump()\n        return End(str(data))\n",
    )
    issues = _contract(f, reads=set(), writes=set(), successors={"End"})
    assert any("bulk" in v for v in issues), issues


def test_body_using_all_declared_fields_is_clean(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "        x = ctx.state.a\n        ctx.state.b = x\n        return Next()\n",
    )
    assert _contract(f, reads={"a"}, writes={"b"}, successors={"Next", "End"}) == []


def test_undeclared_dep_access_is_rejected(tmp_path: Path) -> None:
    # The classic small-model wiring bug: a body reaches for a dependency the system never
    # declared (here `repo_url`). Must be caught at build, not at runtime (AttributeError).
    f = _write(
        tmp_path,
        "        url = ctx.deps.repo_url\n        return End(url)\n",
    )
    issues = check_deps_access(
        f, "N", "run", allowed={"model", "agent_overrides", "loop_cap"}
    )
    assert any("repo_url" in v and "not a declared dependency" in v for v in issues), (
        issues
    )


def test_declared_dep_access_is_clean(tmp_path: Path) -> None:
    f = _write(tmp_path, "        m = ctx.deps.model\n        return End(m)\n")
    assert (
        check_deps_access(
            f, "N", "run", allowed={"model", "agent_overrides", "loop_cap"}
        )
        == []
    )


def test_dynamic_dep_access_is_rejected(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "        url = getattr(ctx.deps, 'repo_url')\n        return End(url)\n",
    )
    issues = check_deps_access(f, "N", "run", allowed={"model"})
    assert any("getattr" in v for v in issues), issues


def test_broad_except_that_swallows_is_rejected(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "        try:\n            x = ctx.state.a\n        except Exception:\n            return Done()\n",
    )
    assert check_fail_loud(f, "N", "run"), (
        "a broad except that swallows must be flagged"
    )


def test_bare_except_that_swallows_is_rejected(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "        try:\n            x = work()\n        except:\n            x = ''\n        return End(x)\n",
    )
    assert check_fail_loud(f, "N", "run")


def test_broad_except_that_reraises_is_allowed(tmp_path: Path) -> None:
    # Translating/propagating the error is fail-loud, not a swallow.
    f = _write(
        tmp_path,
        "        try:\n            x = work()\n        except Exception as e:\n            raise RuntimeError('failed') from e\n",
    )
    assert check_fail_loud(f, "N", "run") == []


def test_narrow_except_is_allowed(tmp_path: Path) -> None:
    # Handling a genuinely expected, specific exception is legitimate.
    f = _write(
        tmp_path,
        "        try:\n            x = int(ctx.state.a)\n        except ValueError:\n            x = 0\n        return End(x)\n",
    )
    assert check_fail_loud(f, "N", "run") == []


def test_body_without_try_is_clean(tmp_path: Path) -> None:
    f = _write(tmp_path, "        ctx.state.b = ctx.state.a\n        return Done()\n")
    assert check_fail_loud(f, "N", "run") == []
