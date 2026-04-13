"""Circuit breaker pattern for fault tolerance.

Prevents cascading failures by fast-failing when a service is down.
When SearXNG goes down, every search waits for timeout then fails.
With 9 searches, that's 4.5 minutes of wasted time. Circuit breaker
detects the pattern and fails fast.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing fast, requests rejected immediately
- HALF_OPEN: Testing if service recovered
"""
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker with three states.

    State transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN → CLOSED: On successful request
    - HALF_OPEN → OPEN: On failed request
    """
    name: str
    failure_threshold: int = 5        # Consecutive failures to open
    recovery_timeout: float = 60.0    # Seconds before trying again
    half_open_max_calls: int = 1      # Test calls in half-open state

    # Internal state (not part of __init__)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"[{self.name}] Circuit half-open, testing recovery")
        return self._state

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """Record successful request."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"[{self.name}] Circuit closed, service recovered")
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record failed request."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"[{self.name}] Circuit re-opened, recovery failed")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"[{self.name}] Circuit opened after {self._failure_count} failures"
            )

    def reset(self) -> None:
        """Reset circuit breaker to initial state (for testing)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


class CircuitOpenError(Exception):
    """Raised when circuit is open and request rejected."""
    def __init__(self, breaker_name: str):
        super().__init__(f"Circuit breaker '{breaker_name}' is open")
        self.breaker_name = breaker_name


# Global circuit breaker for SearXNG
_searxng_breaker: Optional[CircuitBreaker] = None


def get_searxng_breaker() -> CircuitBreaker:
    """Get or create the SearXNG circuit breaker."""
    global _searxng_breaker
    if _searxng_breaker is None:
        _searxng_breaker = CircuitBreaker(
            name="searxng",
            failure_threshold=5,
            recovery_timeout=60.0
        )
    return _searxng_breaker


def reset_searxng_breaker() -> None:
    """Reset the global SearXNG breaker (for testing)."""
    global _searxng_breaker
    if _searxng_breaker is not None:
        _searxng_breaker.reset()
