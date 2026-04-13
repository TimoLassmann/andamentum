"""Tests for epistemic preflight checks."""

from ..preflight import CheckResult, PreflightResult, HealthCheckable, preflight


class FakeHealthy:
    """Component that passes health check."""

    async def check_health(self) -> CheckResult:
        return CheckResult(
            name="FakeHealthy", status="pass", message="ok", elapsed_ms=1.0
        )


class FakeUnhealthy:
    """Component that fails health check."""

    async def check_health(self) -> CheckResult:
        return CheckResult(
            name="FakeUnhealthy", status="fail", message="down", elapsed_ms=2.0
        )


class FakeNoHealth:
    """Component without check_health — should be skipped by preflight."""

    async def gather(self, query: str) -> list:
        return []


class TestCheckResult:
    def test_fields(self):
        r = CheckResult(name="test", status="pass", message="ok", elapsed_ms=5.0)
        assert r.name == "test"
        assert r.status == "pass"
        assert r.message == "ok"
        assert r.elapsed_ms == 5.0


class TestPreflightResult:
    def test_ok_when_all_pass(self):
        result = PreflightResult(
            checks=[
                CheckResult(name="a", status="pass", message="ok", elapsed_ms=1.0),
                CheckResult(name="b", status="pass", message="ok", elapsed_ms=2.0),
            ]
        )
        assert result.ok is True

    def test_not_ok_when_any_fail(self):
        result = PreflightResult(
            checks=[
                CheckResult(name="a", status="pass", message="ok", elapsed_ms=1.0),
                CheckResult(name="b", status="fail", message="down", elapsed_ms=2.0),
            ]
        )
        assert result.ok is False

    def test_ok_when_skip(self):
        result = PreflightResult(
            checks=[
                CheckResult(name="a", status="pass", message="ok", elapsed_ms=1.0),
                CheckResult(
                    name="b", status="skip", message="not applicable", elapsed_ms=0.0
                ),
            ]
        )
        assert result.ok is True

    def test_empty_is_ok(self):
        result = PreflightResult(checks=[])
        assert result.ok is True


class TestHealthCheckableProtocol:
    def test_protocol_matches_healthy(self):
        assert isinstance(FakeHealthy(), HealthCheckable)

    def test_protocol_matches_unhealthy(self):
        assert isinstance(FakeUnhealthy(), HealthCheckable)

    def test_protocol_does_not_match_no_health(self):
        assert not isinstance(FakeNoHealth(), HealthCheckable)


class TestPreflightProviderDiscovery:
    """Test that preflight discovers and runs check_health() on providers."""

    async def test_healthy_providers_all_pass(self):
        result = await preflight(
            model="__skip__",
            providers={
                "healthy1": FakeHealthy(),
                "healthy2": FakeHealthy(),
            },
        )
        # Should have checks for the providers (LLM/WebSearch may fail/skip)
        provider_checks = [c for c in result.checks if c.name.startswith("Fake")]
        assert len(provider_checks) == 2
        assert all(c.status == "pass" for c in provider_checks)

    async def test_unhealthy_provider_fails(self):
        result = await preflight(
            model="__skip__",
            providers={
                "healthy": FakeHealthy(),
                "unhealthy": FakeUnhealthy(),
            },
        )
        provider_checks = [c for c in result.checks if c.name.startswith("Fake")]
        assert len(provider_checks) == 2
        statuses = {c.name: c.status for c in provider_checks}
        assert statuses["FakeHealthy"] == "pass"
        assert statuses["FakeUnhealthy"] == "fail"

    async def test_provider_without_check_health_skipped(self):
        result = await preflight(
            model="__skip__",
            providers={
                "no_health": FakeNoHealth(),
                "healthy": FakeHealthy(),
            },
        )
        names = [c.name for c in result.checks]
        assert "FakeHealthy" in names
        # FakeNoHealth should NOT appear — it has no check_health()
        assert "FakeNoHealth" not in names

    async def test_no_providers_no_model(self):
        # With a garbage model and no providers, LLM init will fail
        result = await preflight(model="__skip__", providers=None)
        # Should still return a result (skip or fail, not crash)
        assert isinstance(result, PreflightResult)
