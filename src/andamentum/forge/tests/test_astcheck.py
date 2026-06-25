"""The fail-loud gate rejects error-swallowing node bodies (no silent fallbacks).

A suggestion in the build prompt ("fail loud, no silent fallbacks") is a hope; this gate
is the guarantee — it deterministically rejects a body that swallows errors, so a system
forge builds can't quietly continue on wrong data.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.forge.astcheck import check_fail_loud


def _write(tmp_path: Path, body: str) -> Path:
    src = "class N:\n    async def run(self, ctx):\n" + body
    f = tmp_path / "n.py"
    f.write_text(src)
    return f


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
