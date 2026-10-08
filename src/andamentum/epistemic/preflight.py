"""Preflight checks for epistemic research runs.

Validates that all required services (LLM, SearXNG, external APIs) are
reachable before starting a potentially expensive research run. Each
component advertises its own health check via ``check_health()`` — the
preflight system discovers and calls these.

Usage::

    from andamentum.epistemic.preflight import preflight

    result = await preflight(model="anthropic:claude-haiku-4-5", providers=providers)
    if not result.ok:
        for c in result.checks:
            if c.status == "fail":
                print(f"FAIL: {c.name} — {c.message}")
        sys.exit(1)

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .operations.base import ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: Literal["pass", "fail", "skip"]
    message: str
    elapsed_ms: float


@dataclass
class PreflightResult:
    """Aggregate result of all preflight checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


@runtime_checkable
class HealthCheckable(Protocol):
    async def check_health(self) -> CheckResult: ...


async def _run_check(component: Any, fallback_name: str) -> CheckResult:
    """Run check_health() on a component with timing and error handling."""
    name = fallback_name
    t0 = time.monotonic()
    try:
        result = await component.check_health()
        return result
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return CheckResult(name=name, status="fail", message=str(e), elapsed_ms=elapsed)


async def preflight(
    *,
    model: str,
    providers: ProviderRegistry | None = None,
    verbose: bool = False,
    searxng_url: str | None = None,
    browser_cdp_url: str | None = None,
) -> PreflightResult:
    """Run preflight checks on LLM, web search, and evidence providers.

    Creates the components, discovers which ones implement ``check_health()``,
    and runs all checks concurrently.

    Args:
        model: LLM model string (same format as DefaultAgentRunner).
        providers: Optional dict of named evidence providers. Each value
            that has a ``check_health()`` method will be checked.
        verbose: If True, log progress.
        searxng_url: The SearXNG instance web search is checked at.
        browser_cdp_url: A Chrome's debugging address; web search is checked
            there instead of at SearXNG. With neither, web search is not
            checked (and the run that follows must say where it searches).

    Returns:
        PreflightResult with individual check outcomes.
    """
    tasks: list[tuple[str, Any]] = []

    # 1. LLM connectivity
    try:
        from .runner import DefaultAgentRunner

        runner = DefaultAgentRunner(model=model)
        tasks.append(("LLM", runner))
    except ImportError:
        # pydantic-ai not installed — skip LLM check
        pass
    except Exception as e:
        # Model resolution failed (e.g., bad boto3 config)
        return PreflightResult(
            checks=[
                CheckResult(
                    name="LLM",
                    status="fail",
                    message=f"Model init failed: {e}",
                    elapsed_ms=0.0,
                ),
            ]
        )

    # 2. Web search (a Chrome when given, else SearXNG); without an address
    # there is nothing to check, and a run would refuse to start.
    skipped: list[CheckResult] = []
    if searxng_url or browser_cdp_url:
        try:
            from .evidence_gathering import WebSearchGatherer

            web = WebSearchGatherer(
                model=model, searxng_url=searxng_url, browser_cdp_url=browser_cdp_url
            )
            tasks.append(("WebSearch", web))
        except ImportError:
            pass
    else:
        skipped.append(
            CheckResult(
                name="WebSearch",
                status="skip",
                message="no address given (--searxng-url or --browser-cdp-url)",
                elapsed_ms=0.0,
            )
        )

    # 3. Evidence providers
    if providers:
        for name, provider in providers.items():
            if hasattr(provider, "check_health"):
                tasks.append((name, provider))

    if not tasks:
        if skipped:
            return PreflightResult(checks=skipped)
        return PreflightResult(
            checks=[
                CheckResult(
                    name="preflight",
                    status="skip",
                    message="No checkable components found",
                    elapsed_ms=0.0,
                ),
            ]
        )

    if verbose:
        logger.info(
            "Running %d preflight checks: %s", len(tasks), [t[0] for t in tasks]
        )

    # Run all checks concurrently
    coros = [_run_check(component, name) for name, component in tasks]
    results = await asyncio.gather(*coros)

    return PreflightResult(checks=[*skipped, *results])
