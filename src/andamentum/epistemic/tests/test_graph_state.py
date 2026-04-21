"""Tests for EpistemicGraphState quarantine tracking."""

from andamentum.epistemic.graph.state import EpistemicGraphState


def test_quarantine_records_record():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    err = ValueError("upstream LLM died")
    state.quarantine("ev-42", "evidence", "extract_evidence", err)
    assert len(state.quarantined) == 1
    record = state.quarantined[0]
    assert record.entity_id == "ev-42"
    assert record.entity_type == "evidence"
    assert record.operation == "extract_evidence"
    assert record.exception_type == "ValueError"
    assert record.message == "upstream LLM died"


def test_is_quarantined_returns_true_after_quarantine():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    assert not state.is_quarantined("ev-42")
    state.quarantine("ev-42", "evidence", "extract_evidence", RuntimeError("x"))
    assert state.is_quarantined("ev-42")


def test_is_quarantined_false_for_unquarantined():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    state.quarantine("ev-42", "evidence", "op", RuntimeError("x"))
    assert not state.is_quarantined("ev-99")


def test_quarantine_idempotent_across_calls():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    state.quarantine("ev-42", "evidence", "op1", RuntimeError("one"))
    state.quarantine("ev-42", "evidence", "op2", RuntimeError("two"))
    # Both records retained, but single membership in the skip set
    assert len(state.quarantined) == 2
    assert state.is_quarantined("ev-42")
