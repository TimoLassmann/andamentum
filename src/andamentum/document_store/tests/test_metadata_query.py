"""Tests for metadata querying: set-membership filters, the metadata-only
read mode, and schema discovery via describe_metadata.

These exercise the pure-SQL read path (no LLM, no embeddings) — documents are
seeded with ``register_document`` (phase-1 only).
"""

from __future__ import annotations

import uuid

import pytest

try:
    import sqlite_vec as _sv  # type: ignore[import-not-found]  # noqa: F401

    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

pytestmark = pytest.mark.skipif(not _HAS_SQLITE_VEC, reason="sqlite_vec not installed")


@pytest.fixture
async def store():
    """A fresh, uniquely-named database, torn down afterwards."""
    from andamentum.document_store import DocumentStore, delete_database

    db_name = f"test_meta_{uuid.uuid4().hex[:8]}"
    s = DocumentStore.for_database(db_name)
    await s.initialize()
    yield s
    delete_database(db_name)


async def _seed_tasks(store) -> None:
    """Three tasks with varied status + a non-task document."""
    await store.register_document(
        title="Task A",
        content="alpha",
        metadata={"record_type": "task", "status": "todo"},
    )
    await store.register_document(
        title="Task B",
        content="bravo",
        metadata={"record_type": "task", "status": "in_progress"},
    )
    await store.register_document(
        title="Task C",
        content="charlie",
        metadata={"record_type": "task", "status": "done"},
    )
    await store.register_document(
        title="An idea",
        content="delta",
        metadata={"record_type": "idea", "stage": "exploring"},
    )


# ---------------------------------------------------------------------------
# find_by_metadata — matching semantics (DB layer)
# ---------------------------------------------------------------------------


class TestFindByMetadataMatching:
    async def test_scalar_is_exact_match(self, store):
        from andamentum.document_store.database import find_by_metadata

        await _seed_tasks(store)
        rows = await find_by_metadata(str(store.db_path), {"status": "todo"})
        assert {r.title for r in rows} == {"Task A"}

    async def test_list_is_set_membership(self, store):
        from andamentum.document_store.database import find_by_metadata

        await _seed_tasks(store)
        rows = await find_by_metadata(
            str(store.db_path),
            {"record_type": "task", "status": ["todo", "in_progress"]},
        )
        assert {r.title for r in rows} == {"Task A", "Task B"}

    async def test_empty_list_matches_nothing(self, store):
        from andamentum.document_store.database import find_by_metadata

        await _seed_tasks(store)
        # Fail-closed: an empty set must NOT silently match every document.
        rows = await find_by_metadata(str(store.db_path), {"status": []})
        assert rows == []

    async def test_string_value_is_not_a_set(self, store):
        from andamentum.document_store.database import find_by_metadata

        await _seed_tasks(store)
        # A bare string must stay an exact-match scalar, not be iterated into
        # a set of characters.
        rows = await find_by_metadata(str(store.db_path), {"status": "todo"})
        assert {r.title for r in rows} == {"Task A"}

    async def test_none_matches_absent_field(self, store):
        from andamentum.document_store.database import find_by_metadata

        await _seed_tasks(store)
        # The idea has no "status" field → json_extract is NULL.
        rows = await find_by_metadata(
            str(store.db_path), {"record_type": "idea", "status": None}
        )
        assert {r.title for r in rows} == {"An idea"}

    async def test_set_membership_combines_with_other_conditions(self, store):
        from andamentum.document_store.database import find_by_metadata

        await _seed_tasks(store)
        # record_type=task AND status IN (...) — the idea is excluded.
        rows = await find_by_metadata(
            str(store.db_path),
            {"record_type": "task", "status": ["todo", "in_progress", "done"]},
        )
        assert {r.title for r in rows} == {"Task A", "Task B", "Task C"}


# ---------------------------------------------------------------------------
# find_by_metadata — metadata-only read mode (public wrapper)
# ---------------------------------------------------------------------------


class TestIncludeContent:
    async def test_include_content_true_returns_content(self, store):
        from andamentum.document_store import find_by_metadata

        await _seed_tasks(store)
        rows = await find_by_metadata(
            store.database_name, {"status": "todo"}, include_content=True
        )
        assert rows and rows[0].snippet == "alpha"

    async def test_include_content_false_skips_content(self, store):
        from andamentum.document_store import find_by_metadata

        await _seed_tasks(store)
        rows = await find_by_metadata(
            store.database_name, {"record_type": "task"}, include_content=False
        )
        assert rows
        assert all(r.snippet == "" for r in rows)
        # Metadata is still present — the cheap overview path stays useful.
        assert all(r.metadata.get("record_type") == "task" for r in rows)


# ---------------------------------------------------------------------------
# describe_metadata — schema discovery
# ---------------------------------------------------------------------------


class TestDescribeMetadata:
    async def test_lists_fields_with_value_counts(self, store):
        from andamentum.document_store import describe_metadata

        await _seed_tasks(store)
        schema = await describe_metadata(store.database_name)

        assert schema["record_type"].present_in == 4
        assert schema["record_type"].distinct == 2
        assert schema["record_type"].values == {"task": 3, "idea": 1}

        # status is only on the three tasks
        assert schema["status"].present_in == 3
        assert schema["status"].values == {"todo": 1, "in_progress": 1, "done": 1}

    async def test_filter_scopes_the_profile(self, store):
        from andamentum.document_store import describe_metadata

        await _seed_tasks(store)
        schema = await describe_metadata(
            store.database_name, filters={"record_type": "task"}
        )
        # Within tasks: status present on all three, no "stage" field leaks in.
        assert schema["status"].present_in == 3
        assert "stage" not in schema

    async def test_high_cardinality_values_omitted(self, store):
        from andamentum.document_store import describe_metadata

        await _seed_tasks(store)
        # Three distinct statuses, but cap at 2 → breakdown suppressed, counts kept.
        schema = await describe_metadata(store.database_name, max_values=2)
        assert schema["status"].distinct == 3
        assert schema["status"].values is None
        # record_type has 2 distinct, at the cap → still enumerated.
        assert schema["record_type"].values == {"task": 3, "idea": 1}

    async def test_internal_fields_excluded(self, store):
        from andamentum.document_store import describe_metadata

        await store.register_document(
            title="With history",
            content="x",
            metadata={"record_type": "note", "_history": [{"changed_at": "t"}]},
        )
        schema = await describe_metadata(store.database_name)
        assert "record_type" in schema
        assert "_history" not in schema
