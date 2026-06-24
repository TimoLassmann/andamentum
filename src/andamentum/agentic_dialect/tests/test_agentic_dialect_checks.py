"""The portable static gates: each fires on a violation, stays quiet on clean code."""

from __future__ import annotations

from pathlib import Path

from andamentum.agentic_dialect import check_code


def _write(tmp_path: Path, name: str, src: str) -> Path:
    f = tmp_path / name
    f.write_text(src, encoding="utf-8")
    return f


def _codes(violations: list) -> set[str]:
    return {v.code for v in violations}


def test_engine_import_in_worker_flagged(tmp_path: Path) -> None:
    f = _write(tmp_path, "summarize.py", "from pydantic_graph import End\n")
    assert "engine-import-in-worker" in _codes(check_code(f))


def test_engine_import_in_graph_is_fine(tmp_path: Path) -> None:
    src = "from __future__ import annotations\nfrom pydantic_graph import Graph\n"
    f = _write(tmp_path, "graph.py", src)
    assert "engine-import-in-worker" not in _codes(check_code(f))


def test_untyped_dict_any_flagged(tmp_path: Path) -> None:
    f = _write(tmp_path, "schemas.py", "from typing import Any\nx: dict[str, Any] = {}\n")
    assert "untyped-dict-any" in _codes(check_code(f))


def test_nondeterminism_in_run_flagged(tmp_path: Path) -> None:
    src = (
        "from __future__ import annotations\n"
        "import random\n"
        "class N:\n"
        "    async def run(self, ctx):\n"
        "        return A() if random.random() > 0.5 else B()\n"
    )
    f = _write(tmp_path, "graph.py", src)
    assert "nondeterminism-in-routing" in _codes(check_code(f))


def test_client_in_node_body_flagged(tmp_path: Path) -> None:
    src = (
        "from __future__ import annotations\n"
        "class N:\n"
        "    async def run(self, ctx):\n"
        "        a = Agent('m')\n"
        "        return End(a)\n"
    )
    f = _write(tmp_path, "graph.py", src)
    assert "client-in-node-body" in _codes(check_code(f))


def test_while_true_flagged(tmp_path: Path) -> None:
    src = (
        "from __future__ import annotations\n"
        "class N:\n"
        "    async def run(self, ctx):\n"
        "        while True:\n"
        "            break\n"
        "        return End(1)\n"
    )
    f = _write(tmp_path, "graph.py", src)
    assert "unbounded-loop" in _codes(check_code(f))


def test_literal_loop_bound_flagged(tmp_path: Path) -> None:
    src = (
        "from __future__ import annotations\n"
        "class N:\n"
        "    async def run(self, ctx):\n"
        "        for _ in range(5):\n"
        "            pass\n"
        "        return End(1)\n"
    )
    f = _write(tmp_path, "graph.py", src)
    assert "literal-loop-bound" in _codes(check_code(f))


def test_missing_future_annotations_flagged(tmp_path: Path) -> None:
    f = _write(tmp_path, "graph.py", "class N:\n    pass\n")
    assert "missing-future-annotations" in _codes(check_code(f))


def test_clean_worker_passes(tmp_path: Path) -> None:
    src = (
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "class R(BaseModel):\n"
        "    s: str\n"
        "async def summarize(source: str, *, model: str) -> R:\n"
        "    return R(s=source)\n"
    )
    f = _write(tmp_path, "summarize.py", src)
    assert check_code(f) == []
