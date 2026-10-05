"""Check protocol, registry, and the interfaces of the cache and promotion policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from andamentum.core.decisions import CalibrationRecord

from .diagnostics import Diagnostic, Tier
from .store import ProjectStore
from .view import DocumentView

if TYPE_CHECKING:
    from andamentum.references import ReferenceHttp

PROMOTE_AT = 0.9
MODEL_NEEDS = frozenset({"model", "decision_model"})


class CheckSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tier: Tier
    scope: Literal["span", "paragraph", "section", "document"]
    needs: list[str]  # e.g. ["network"], ["results"], ["model"]


@dataclass
class RunStats:
    """What one run spent: counted by the checks and the HTTP client, reported."""

    network_calls: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        self.model_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass(frozen=True)
class CheckContext:
    store: ProjectStore | None
    offline: bool
    models: dict[str, str] = field(default_factory=dict)  # role -> model id
    results_dir: Path | None = None
    # The one HTTP client of the run (cached through the project store); None
    # when offline. Network checks raise when they need it and it is absent.
    http: ReferenceHttp | None = None
    # Decision-model calibration records by check id, when the project has them.
    calibrations: dict[str, CalibrationRecord] = field(default_factory=dict)
    stats: RunStats = field(default_factory=RunStats)


class Check(Protocol):
    spec: CheckSpec
    #: Bump when the check's logic changes; it is part of the cache key.
    version: str

    def run(
        self, view: DocumentView, ctx: CheckContext
    ) -> list[Diagnostic] | Awaitable[list[Diagnostic]]: ...


_REGISTRY: dict[str, Check] = {}


def register(check: Check) -> Check:
    """Add a check; a duplicate id raises."""
    if check.spec.id in _REGISTRY:
        raise ValueError(f"check {check.spec.id!r} is already registered")
    _REGISTRY[check.spec.id] = check
    return check


def all_checks() -> list[Check]:
    return list(_REGISTRY.values())


def select(
    *,
    ids: Iterable[str] | None = None,
    tiers: Iterable[int] | None = None,
    needs: Iterable[str] | None = None,
) -> list[Check]:
    """Registered checks matching every given filter.

    ``ids`` and ``tiers`` keep checks whose id/tier is listed (an unknown id
    raises). ``needs`` is the set of capabilities available: a check is kept
    only when everything it needs is in that set.
    """
    chosen = all_checks()
    if ids is not None:
        wanted = list(ids)
        unknown = [i for i in wanted if i not in _REGISTRY]
        if unknown:
            raise KeyError(f"unknown check id(s): {', '.join(unknown)}")
        chosen = [c for c in chosen if c.spec.id in wanted]
    if tiers is not None:
        tier_set = set(tiers)
        chosen = [c for c in chosen if c.spec.tier in tier_set]
    if needs is not None:
        available = set(needs)
        chosen = [c for c in chosen if set(c.spec.needs) <= available]
    return chosen


def _clear_registry() -> None:
    """Test helper."""
    _REGISTRY.clear()


def cache_key(check: Check, *, version: str, inputs: Iterable[str]) -> str:
    """Key = (check id, check version, hash of the declared inputs).

    ``inputs`` are the texts the check declares it reads (a paragraph, a
    section, a results file); the key changes when any of them, the check id or
    the check version changes, and only then.
    """
    parts = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in inputs]
    payload = json.dumps([check.spec.id, version, parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CACHE_NAMESPACE = "checks"


def load_cached(store: ProjectStore, key: str) -> list[Diagnostic] | None:
    """Diagnostics stored under *key*, or None when there is no entry."""
    raw = store.cache_get(CACHE_NAMESPACE, key)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"check cache entry {key} is not a list")
    return [Diagnostic.model_validate(item) for item in raw]


def save_cached(store: ProjectStore, key: str, diagnostics: list[Diagnostic]) -> None:
    store.cache_put(CACHE_NAMESPACE, key, [d.model_dump(mode="json") for d in diagnostics])


def effective_tier(check: Check, *, precision: float | None) -> Tier:
    """The tier a check reports at.

    A check that needs no model is deterministic: its declared tier stands. A
    model-judged check (it needs ``model`` or ``decision_model``) keeps its
    declared tier only with a measured precision of at least ``PROMOTE_AT``;
    without a record, or below it, it is advisory (tier 2).
    """
    if not set(check.spec.needs) & MODEL_NEEDS:
        return check.spec.tier
    if precision is not None and precision >= PROMOTE_AT:
        return check.spec.tier
    return 2


def load_precisions(path: Path) -> dict[str, float]:
    """Measured precision per check id from a JSON file ``{id: {"precision": x}}``.

    A missing file means no check has a record.
    """
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold an object keyed by check id")
    return {str(k): float(v["precision"]) for k, v in data.items()}
