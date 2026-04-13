"""Tests for circuit breaker pattern."""

import pytest
from unittest.mock import patch

from ..circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_searxng_breaker,
    reset_searxng_breaker,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global breakers between tests."""
    reset_searxng_breaker()
    yield
    reset_searxng_breaker()


class TestCircuitBreakerStates:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

    def test_closed_allows_requests(self):
        cb = CircuitBreaker(name="test")
        assert cb.allow_request() is True

    def test_closed_to_open_after_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        cb = CircuitBreaker(name="test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is False

    def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate time passing beyond recovery timeout
        last_failure_time = cb._last_failure_time or 0.0
        with patch(
            "andamentum.deep_research.circuit_breaker.time.monotonic",
            return_value=last_failure_time + 11.0,
        ):
            assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_requests(self):
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=10.0,
            half_open_max_calls=1,
        )
        cb.record_failure()

        last_failure_time = cb._last_failure_time or 0.0
        with patch(
            "andamentum.deep_research.circuit_breaker.time.monotonic",
            return_value=last_failure_time + 11.0,
        ):
            assert cb.allow_request() is True  # First call allowed
            assert cb.allow_request() is False  # Second call rejected

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()

        last_failure_time = cb._last_failure_time or 0.0
        with patch(
            "andamentum.deep_research.circuit_breaker.time.monotonic",
            return_value=last_failure_time + 11.0,
        ):
            _ = cb.state  # Trigger transition to HALF_OPEN
            cb.record_success()
            assert cb.state == CircuitState.CLOSED
            assert cb._failure_count == 0

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()

        last_failure_time = cb._last_failure_time or 0.0
        with patch(
            "andamentum.deep_research.circuit_breaker.time.monotonic",
            return_value=last_failure_time + 11.0,
        ):
            _ = cb.state  # Trigger transition to HALF_OPEN
            cb.record_failure()
            assert cb._state == CircuitState.OPEN


class TestCircuitBreakerCounting:
    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_all_state(self):
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._last_failure_time is None
        assert cb._half_open_calls == 0


class TestCircuitOpenError:
    def test_error_message(self):
        err = CircuitOpenError("searxng")
        assert "searxng" in str(err)
        assert err.breaker_name == "searxng"

    def test_is_exception(self):
        with pytest.raises(CircuitOpenError):
            raise CircuitOpenError("test")


class TestGlobalBreakers:
    def test_get_searxng_breaker_singleton(self):
        b1 = get_searxng_breaker()
        b2 = get_searxng_breaker()
        assert b1 is b2
        assert b1.name == "searxng"

    def test_reset_searxng_breaker(self):
        b = get_searxng_breaker()
        b.record_failure()
        assert b._failure_count == 1
        reset_searxng_breaker()
        assert b._failure_count == 0
