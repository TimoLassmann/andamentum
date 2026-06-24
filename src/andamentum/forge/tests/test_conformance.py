"""forge eats its own dog food: the meta-system — and every system it renders — is
dialect-conforming under the portable static gates (``check_code``)."""

from __future__ import annotations

from pathlib import Path

import andamentum.forge as forge_pkg
from andamentum.agentic_dialect import check_code
from andamentum.forge import compile_spec, render
from andamentum.forge.schemas import DesignPlan, ForgeWhy, NodeDraft
from andamentum.forge.spec import NodeKind

_MODULE_DIR = Path(forge_pkg.__file__).resolve().parent


def test_forge_module_passes_the_dialect_gates() -> None:
    violations = check_code(_MODULE_DIR)
    assert violations == [], [
        f"{v.file}:{v.line} [{v.law}] {v.code}" for v in violations
    ]


def test_rendered_package_passes_the_dialect_gates(tmp_path: Path) -> None:
    plan = DesignPlan(
        why=ForgeWhy(
            purpose="Answer questions about a topic.",
            boundary_in="a question",
            boundary_out="an answer",
        ),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Parse the question.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["parsed"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Answer the question.",
                kind=NodeKind.HEAD,
                consumes=["parsed"],
                produces=["answer"],
            ),
        ],
    )
    spec = compile_spec(plan)
    render(spec, tmp_path)
    violations = check_code(tmp_path / spec.name)
    assert violations == [], [
        f"{v.file}:{v.line} [{v.law}] {v.code}" for v in violations
    ]
