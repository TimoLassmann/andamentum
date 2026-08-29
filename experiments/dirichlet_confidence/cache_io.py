"""Append-only JSONL cache for elicitations — the boundary between the
expensive GATHER step and the free ANALYZE step.

One row per (kind, claim_id, model, phrasing_id) elicitation:

    {"kind": "baseline"|"method", "claim_id": str, "model": str,
     "phrasing_id": int, "temperature": float, "row": [s, c, n] | null}

``row`` is the raw 0-100 verbalized distribution (or null if the call failed).
The cache is the single source of truth the analysis reads, so a frozen cache
means perfectly reproducible analysis.

Backward compatibility: the previous experiment wrote rows keyed by
``phase``/``sample`` where ``sample`` was an int (baseline) or a temperature
string like ``"t0.7"`` (the old temperature-ensemble). Those rows are still
readable via :meth:`get_legacy`, so nothing already paid for is lost — but new
gathering uses the clean ``kind``/``phrasing_id`` schema above.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_PATH = Path("experiments/dirichlet_confidence/cache/elicitations.jsonl")

# Field separator for the in-memory dict key.
_SEP = "\x1f"


def _key(kind: str, claim_id: str, model: str, phrasing_id: int) -> str:
    return f"{kind}{_SEP}{claim_id}{_SEP}{model}{_SEP}{phrasing_id}"


class ElicitationCache:
    """Loads an append-only JSONL cache and lets you query/append rows."""

    def __init__(self, path: Path | str = CACHE_PATH) -> None:
        self.path = Path(path)
        self._rows: dict[str, list[float] | None] = {}
        # Legacy index: (claim_id, model) -> {legacy_sample -> row}, for reuse
        # of the previous experiment's cache (see module docstring).
        self._legacy: dict[tuple[str, str], dict[str, list[float] | None]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "kind" in r and "phrasing_id" in r:
                self._rows[_key(r["kind"], r["claim_id"], r["model"], r["phrasing_id"])] = (
                    r["row"]
                )
            elif "phase" in r and "sample" in r:
                # Legacy schema from the temperature-ensemble experiment.
                self._legacy.setdefault((r["claim_id"], r["model"]), {})[
                    str(r["sample"])
                ] = r["row"]

    # ----- queries -----

    def has(self, kind: str, claim_id: str, model: str, phrasing_id: int) -> bool:
        return _key(kind, claim_id, model, phrasing_id) in self._rows

    def get(self, kind: str, claim_id: str, model: str, phrasing_id: int) -> list[float] | None:
        return self._rows.get(_key(kind, claim_id, model, phrasing_id))

    def get_legacy(self, claim_id: str, model: str, sample: str) -> list[float] | None:
        """Read a row written under the old phase/sample schema (e.g. a baseline
        row keyed ``sample=0``, or an old temperature row keyed ``sample='t0.7'``)."""
        return self._legacy.get((claim_id, model), {}).get(sample)

    def size(self) -> int:
        return len(self._rows)

    # ----- mutation -----

    def put(
        self,
        kind: str,
        claim_id: str,
        model: str,
        phrasing_id: int,
        temperature: float,
        row: list[float] | None,
    ) -> None:
        """Append one row to disk and the in-memory index."""
        self._rows[_key(kind, claim_id, model, phrasing_id)] = row
        with self.path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "kind": kind,
                        "claim_id": claim_id,
                        "model": model,
                        "phrasing_id": phrasing_id,
                        "temperature": temperature,
                        "row": row,
                    }
                )
                + "\n"
            )

    def migrate_legacy_baselines(self, models: list[str]) -> int:
        """Copy legacy baseline rows (phase=baseline, sample=0) into the new
        schema so the 90 already-paid baseline calls are reused without re-asking.

        Returns the number of rows migrated. Idempotent.
        """
        migrated = 0
        for (claim_id, model), samples in self._legacy.items():
            if model not in models:
                continue
            if "0" not in samples:
                continue
            if self.has("baseline", claim_id, model, 0):
                continue
            self.put("baseline", claim_id, model, 0, 0.0, samples["0"])
            migrated += 1
        return migrated
