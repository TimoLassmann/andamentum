"""Guards that ``andamentum.llm_judge`` stays dialect-conforming and a
correctly layered leaf module — fast, static checks, no LLM calls.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from andamentum.agentic_dialect.checks import check_code

MODULE_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_MODULES = {
    "pydantic_graph",
    "andamentum.epistemic",
    "andamentum.whetstone",
    "andamentum.deep_research",
    "andamentum.document_store",
    "andamentum.scribe",
    "andamentum.figures",
    "andamentum.typeset",
    "andamentum.chunker",
    "andamentum.harvest",
    "andamentum.vision_critique",
}


def _source_files():
    return sorted(p for p in MODULE_ROOT.glob("*.py") if p.name != "__pycache__")


def test_check_code_passes_default():
    violations = check_code(MODULE_ROOT)
    assert violations == []


def test_check_code_passes_strict():
    violations = check_code(MODULE_ROOT, strict=True)
    assert violations == []


def test_no_forbidden_imports_anywhere_in_the_module():
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name is None:
                    continue
                for forbidden in FORBIDDEN_MODULES:
                    assert not (
                        name == forbidden or name.startswith(forbidden + ".")
                    ), (
                        f"{path.name} imports forbidden module {name!r} "
                        f"(matched {forbidden!r})"
                    )


def test_public_surface_matches_documented_set():
    import andamentum.llm_judge as llm_judge

    expected = {
        "judge_score",
        "judge_compare",
        "Criterion",
        "CriterionScore",
        "ScoreResult",
        "CompareResult",
        "JudgeVote",
        "DEFAULT_CRITERIA",
    }
    assert set(llm_judge.__all__) == expected


def test_judge_score_and_judge_compare_are_coroutine_functions_with_keyword_only_model():
    import andamentum.llm_judge as llm_judge

    for fn in (llm_judge.judge_score, llm_judge.judge_compare):
        assert inspect.iscoroutinefunction(fn)
        sig = inspect.signature(fn)
        model_param = sig.parameters["model"]
        assert model_param.kind == inspect.Parameter.KEYWORD_ONLY
