from __future__ import annotations

import pytest

from andamentum.compose.harness.checks.structure import StructureCheck
from andamentum.compose.harness.kernel import CheckContext
from andamentum.compose.harness.kernel.store import ProjectStore

from ._helpers import CTX, make_view


def words(n: int) -> str:
    return " ".join(["w"] * n)


def test_no_config_no_diagnostics(tmp_path):
    view = make_view(("Abstract", 1), (words(500), None), path=str(tmp_path / "d.docx"))
    assert StructureCheck().run(view, CTX) == []


def test_total_words():
    view = make_view(("H", 1), (words(30), None))
    out = StructureCheck({"max_words": 20}).run(view, CTX)
    assert len(out) == 1 and "30 words" in out[0].message
    assert StructureCheck({"max_words": 30}).run(view, CTX) == []


def test_required_sections():
    view = make_view(("Introduction", 1), ("x", None))
    out = StructureCheck({"required_sections": ["introduction", "Methods"]}).run(
        view, CTX
    )
    assert len(out) == 1 and "'Methods'" in out[0].message


def test_section_limits_point_at_heading():
    view = make_view(
        ("Abstract", 1), (words(300), None), ("Methods", 1), (words(2), None)
    )
    check = StructureCheck(
        {
            "sections": {
                "Abstract": {"max_words": 250},
                "Methods": {"min_words": 5},
                "Absent": {"max_words": 1},
            }
        }
    )
    out = check.run(view, CTX)
    assert [d.paragraph_id for d in out] == ["p0", "p2"]
    assert "max_words" in out[0].message and "min_words" in out[1].message


def test_figure_and_table_limits():
    view = make_view(
        ("Figure 1. a", None),
        ("Figure 2. b", None),
        ("Figure 3", None, "Caption"),
        ("Table 1. t", None),
    )
    out = StructureCheck({"max_figures": 2, "max_tables": 1}).run(view, CTX)
    assert len(out) == 1 and "3 figures" in out[0].message


def _project_ctx(tmp_path, yaml_text):
    store = ProjectStore(tmp_path / "d.harness")
    store.path.mkdir()
    (store.path / "checks.yaml").write_text(yaml_text)
    return CheckContext(store=store, offline=True)


def test_loaded_from_the_project_checks_yaml(tmp_path):
    ctx = _project_ctx(tmp_path, "structure:\n  max_words: 5\n")
    view = make_view(("H", 1), (words(10), None), path=str(tmp_path / "d.docx"))
    assert len(StructureCheck().run(view, ctx)) == 1


def test_a_checks_yaml_beside_the_document_is_not_read(tmp_path):
    (tmp_path / "checks.yaml").write_text("structure:\n  max_words: 5\n")
    view = make_view(("H", 1), (words(10), None), path=str(tmp_path / "d.docx"))
    assert StructureCheck().run(view, CTX) == []


def test_yaml_without_structure_key(tmp_path):
    ctx = _project_ctx(tmp_path, "models: {}\n")
    view = make_view(("H", 1), (words(10), None), path=str(tmp_path / "d.docx"))
    assert StructureCheck().run(view, ctx) == []


def test_malformed_structure_section_raises(tmp_path):
    ctx = _project_ctx(tmp_path, "structure:\n  - a\n")
    view = make_view(("H", 1), path=str(tmp_path / "d.docx"))
    with pytest.raises(ValueError):
        StructureCheck().run(view, ctx)


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        StructureCheck({"max_word": 3})
