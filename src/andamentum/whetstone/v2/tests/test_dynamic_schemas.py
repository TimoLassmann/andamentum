"""Tests for whetstone v2's runtime-schema builder.

Covers:

* slugify_criterion is deterministic and total.
* slugify_criterion handles non-ascii input gracefully (transliterates
  combining marks; raises on inputs that collapse to empty).
* create_custom_evaluation_model produces the expected flat field set
  (status + notes per criterion + overall_assessment).
* The runtime model accepts the three-value status enum and rejects
  bad values.
* unpack_custom_evaluations round-trips the filled model into a flat
  list of CustomEvaluation, preserving order.
* Slug collisions raise a clear error before the LLM call.
* The MAX_CUSTOM_CRITERIA cap is enforced at construction time.
"""

from __future__ import annotations

import pytest

from andamentum.whetstone.v2.dynamic_schemas import (
    MAX_CUSTOM_CRITERIA,
    create_custom_evaluation_model,
    slugify_criterion,
    unpack_custom_evaluations,
)
from andamentum.whetstone.v2.schemas import CustomEvaluation


# ── slugify_criterion ─────────────────────────────────────────────────


class TestSlugifyCriterion:
    def test_simple_lowercase(self) -> None:
        assert slugify_criterion("Originality") == "originality"

    def test_spaces_become_underscores(self) -> None:
        assert slugify_criterion("depth of literature") == "depth_of_literature"

    def test_punctuation_collapses_to_underscore(self) -> None:
        assert slugify_criterion("clarity & rigour!") == "clarity_rigour"

    def test_strips_leading_trailing_underscores(self) -> None:
        assert slugify_criterion("  -- abc -- ") == "abc"

    def test_handles_unicode_combining_marks(self) -> None:
        # "café" → NFKD strips the accent, so we get "cafe".
        assert slugify_criterion("café") == "cafe"
        assert slugify_criterion("über klar") == "uber_klar"

    def test_idempotent(self) -> None:
        slug = slugify_criterion("Hello, World!")
        assert slugify_criterion(slug) == slug

    def test_deterministic(self) -> None:
        # Same input → same slug, every time.
        s = "Some Criterion / With Punctuation"
        assert slugify_criterion(s) == slugify_criterion(s) == slugify_criterion(s)

    def test_non_ascii_only_raises(self) -> None:
        # Pure CJK or pure symbols → empty slug → ValueError.
        with pytest.raises(ValueError, match="empty slug"):
            slugify_criterion("日本語")
        with pytest.raises(ValueError, match="empty slug"):
            slugify_criterion("***")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty slug"):
            slugify_criterion("")
        with pytest.raises(ValueError, match="empty slug"):
            slugify_criterion("   ")

    def test_non_string_input_raises(self) -> None:
        with pytest.raises(TypeError):
            slugify_criterion(123)  # type: ignore[arg-type]


# ── create_custom_evaluation_model ────────────────────────────────────


class TestCreateCustomEvaluationModel:
    def test_one_criterion_makes_three_fields(self) -> None:
        # 1 criterion × (status + notes) + overall_assessment = 3 fields.
        model = create_custom_evaluation_model(["originality"])
        assert sorted(model.model_fields.keys()) == [
            "originality_notes",
            "originality_status",
            "overall_assessment",
        ]

    def test_three_criteria_makes_seven_fields(self) -> None:
        # 3 × (status + notes) + overall_assessment = 7 fields.
        criteria = ["originality", "depth of literature", "clarity of methods"]
        model = create_custom_evaluation_model(criteria)
        assert len(model.model_fields) == 7

    def test_field_descriptions_quote_original_criterion(self) -> None:
        criteria = ["Originality of Approach"]
        model = create_custom_evaluation_model(criteria)
        status_field = model.model_fields["originality_of_approach_status"]
        assert "Originality of Approach" in (status_field.description or "")
        notes_field = model.model_fields["originality_of_approach_notes"]
        assert "Originality of Approach" in (notes_field.description or "")

    def test_status_field_accepts_three_values(self) -> None:
        model = create_custom_evaluation_model(["x"])
        # All three valid.
        for status in ("pass", "fail", "unclear"):
            inst = model(x_status=status, x_notes="some notes", overall_assessment="ok")
            assert getattr(inst, "x_status") == status

    def test_status_field_rejects_bad_value(self) -> None:
        model = create_custom_evaluation_model(["x"])
        with pytest.raises(Exception):
            model(x_status="maybe", x_notes="…", overall_assessment="…")

    def test_round_trip_through_dump(self) -> None:
        criteria = ["a", "b"]
        model = create_custom_evaluation_model(criteria)
        inst = model(
            a_status="pass",
            a_notes="evidence A",
            b_status="fail",
            b_notes="missing thing",
            overall_assessment="mixed",
        )
        dumped = inst.model_dump()
        restored = model.model_validate(dumped)
        assert restored == inst

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            create_custom_evaluation_model([])

    def test_too_many_criteria_raises(self) -> None:
        criteria = [f"criterion {i}" for i in range(MAX_CUSTOM_CRITERIA + 1)]
        with pytest.raises(ValueError, match="too many criteria"):
            create_custom_evaluation_model(criteria)

    def test_empty_criterion_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty / whitespace"):
            create_custom_evaluation_model(["valid", ""])

    def test_slug_collision_raises(self) -> None:
        # "depth-of-literature" and "depth of literature" both slugify to
        # "depth_of_literature".
        with pytest.raises(ValueError, match="slug collision"):
            create_custom_evaluation_model(
                ["depth of literature", "depth-of-literature"]
            )

    def test_case_collision_raises(self) -> None:
        # "X" and "x" both slugify to "x".
        with pytest.raises(ValueError, match="slug collision"):
            create_custom_evaluation_model(["X", "x"])

    def test_non_ascii_only_criterion_raises(self) -> None:
        with pytest.raises(ValueError, match="empty slug"):
            create_custom_evaluation_model(["日本語"])


# ── unpack_custom_evaluations ─────────────────────────────────────────


class TestUnpackCustomEvaluations:
    def test_preserves_order(self) -> None:
        criteria = ["originality", "depth of literature", "clarity of methods"]
        model = create_custom_evaluation_model(criteria)
        inst = model(
            originality_status="pass",
            originality_notes="A",
            depth_of_literature_status="fail",
            depth_of_literature_notes="B",
            clarity_of_methods_status="unclear",
            clarity_of_methods_notes="C",
            overall_assessment="mixed",
        )
        out = unpack_custom_evaluations(criteria, inst)
        assert [e.criterion for e in out] == criteria

    def test_preserves_status_and_notes(self) -> None:
        criteria = ["a", "b"]
        model = create_custom_evaluation_model(criteria)
        inst = model(
            a_status="pass",
            a_notes="evidence A",
            b_status="fail",
            b_notes="missing thing",
            overall_assessment="mixed",
        )
        out = unpack_custom_evaluations(criteria, inst)
        assert out == [
            CustomEvaluation(criterion="a", status="pass", notes="evidence A"),
            CustomEvaluation(criterion="b", status="fail", notes="missing thing"),
        ]

    def test_strips_whitespace_in_criteria(self) -> None:
        criteria = ["  spaced  "]
        model = create_custom_evaluation_model(criteria)
        inst = model(spaced_status="pass", spaced_notes="ok", overall_assessment="fine")
        out = unpack_custom_evaluations(criteria, inst)
        assert out[0].criterion == "spaced"
