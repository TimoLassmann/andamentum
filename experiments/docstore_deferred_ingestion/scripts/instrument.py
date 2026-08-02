"""The instruments. Every one attaches at a boundary the shipped API already offers.

Nothing under ``src/andamentum`` is patched, stubbed, copied or reconfigured —
otherwise the experiment would be measuring a modified system, and the gap it
exists to close ("the real path has never run") would still be open.

Three instruments:

* :func:`counting_convert_fn` wraps the *injected* ``convert_fn`` parameter of
  ``process_pending``. Counting invocations is genuine black-box observation.
  The ledger line is fsynced BEFORE the wrapper returns, which is precisely what
  makes it valid evidence under a SIGKILL.
* :func:`http_recorder` wraps ``httpx.AsyncClient.send`` in the EXPERIMENT'S OWN
  process. All model traffic goes through httpx (``document_store/embeddings.py``
  builds an ``httpx.AsyncClient``; pydantic-ai's Ollama provider is an
  OpenAI-compatible httpx client), so this is a complete, provider-agnostic
  transport counter that needs no change to src/.
* :func:`poll_until` drives the kill gate from a read-only connection.

This module is deliberately import-guard-free so the offline harness tests can
exercise it without a document store in sight.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator
from urllib.parse import urlsplit


# --------------------------------------------------------------------------
# Append-only JSONL ledgers
# --------------------------------------------------------------------------


def append_jsonl(path: str | Path, record: dict[str, Any], *, fsync: bool = True) -> None:
    """Append one JSON object as a line, optionally fsyncing before returning.

    ``fsync=True`` is not decoration: the converter ledger is read back after a
    SIGKILL, and a line still sitting in the OS page cache would make the
    checkpoint evidence unfalsifiable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL ledger. A missing file is an empty ledger; a corrupt line raises."""
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(target.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{lineno} is not valid JSON: {exc}") from exc
    return records


def truncate_ledger(path: str | Path) -> Path:
    """Empty a ledger at rule start.

    Every rule truncates its OWN ledgers, so a Snakemake re-run of a partially
    failed rule cannot double-count. The single exception is ``claim_c_resume``,
    which APPENDS to ``claim_c_kill``'s ledger and therefore declares that file
    as both an input and an output.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("")
    return target


# --------------------------------------------------------------------------
# Converter ledger — the load-bearing instrument for the checkpoint claim
# --------------------------------------------------------------------------


def counting_convert_fn(
    ledger_path: str | Path,
    inner: Callable[[str], Awaitable[str]],
) -> Callable[[str], Awaitable[str]]:
    """Wrap a ``ConvertFn`` so every invocation leaves a durable ledger line.

    The experiment NEVER passes ``pipeline.harvest_convert_fn()`` to
    ``process_pending`` directly — always through this. Claim (c) then reduces to
    a line count a reader can verify by opening a text file.
    """

    async def _counted(source: str) -> str:
        started = time.time()
        started_mono = time.monotonic()
        try:
            markdown = await inner(source)
        except Exception as exc:
            # A failed conversion is still an invocation: the count is what
            # proves work was or was not repeated, so it must be recorded even
            # when the work did not succeed.
            append_jsonl(
                ledger_path,
                {
                    "ts_start": started,
                    "ts_end": time.time(),
                    "seconds": time.monotonic() - started_mono,
                    "source": source,
                    "pid": os.getpid(),
                    "chars": 0,
                    "sha256": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                },
            )
            raise
        import hashlib

        append_jsonl(
            ledger_path,
            {
                "ts_start": started,
                "ts_end": time.time(),
                "seconds": time.monotonic() - started_mono,
                "source": source,
                "pid": os.getpid(),
                "chars": len(markdown),
                "sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                "error_type": None,
                "error_message": None,
            },
        )
        return markdown

    return _counted


def kill_process_group(proc: Any, *, wait_seconds: float = 60.0) -> None:
    """SIGKILL a child's whole process group, tolerating an already-dead group.

    ``proc.kill()`` signals only the DIRECT child. Every worker here is spawned as
    ``uv run python -m ...``, a two-process chain, and Docling forks further, so a
    direct kill leaves the real work alive. A leaked worker keeps writing to the
    experiment database and keeps calling Ollama, which silently corrupts the NEXT
    run — strictly worse than a crash. Spawn with ``start_new_session=True`` and
    clean up through this.
    """
    import signal
    import subprocess

    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        print(f"WARNING: pid={proc.pid} survived group SIGKILL", flush=True)


def ledger_sources(ledger_path: str | Path) -> list[str]:
    """The source strings, in invocation order, from a converter ledger."""
    return [r["source"] for r in read_jsonl(ledger_path)]


def duplicate_sources(ledger_path: str | Path) -> list[str]:
    """Source strings converted more than once — must be empty for H-c1."""
    seen: dict[str, int] = {}
    for source in ledger_sources(ledger_path):
        seen[source] = seen.get(source, 0) + 1
    return sorted(s for s, n in seen.items() if n > 1)


# --------------------------------------------------------------------------
# HTTP transport recorder
# --------------------------------------------------------------------------


@dataclass
class HttpLedger:
    """Derived scalars over a recorded set of HTTP requests."""

    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.records)

    def to_host(self, host_fragment: str) -> list[dict[str, Any]]:
        return [r for r in self.records if host_fragment in (r.get("host") or "")]

    def by_path(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            key = str(record.get("path"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def max_in_flight(self) -> int:
        """Peak GLOBAL concurrency observed across every endpoint at once.

        This is the whole-process depth, sampled at each request's start. It is a
        conservative UPPER bound on any single endpoint's concurrency and must
        never be reported as one — use :meth:`max_in_flight_by_path` for that.
        """
        return max((r.get("global_depth_at_start", 0) for r in self.records), default=0)

    def max_in_flight_by_path(self) -> dict[str, int]:
        """Peak concurrency observed per endpoint, counted PER ENDPOINT.

        WHY PER-PATH: a single process can be strictly sequential in its OWN
        awaits while a library it calls fans out internally. ``core.embeddings.
        make_embedder`` gathers up to ``Semaphore(8)`` concurrent
        ``/api/embeddings`` calls inside the chunker's ``embedding_fn``, so a
        whole-ledger peak conflates "this script issued concurrent calls" with
        "a callee did". Attributing concurrency to the endpoint is what makes
        the distinction measurable instead of arguable.

        AND WHY IT IS ITS OWN COUNTER. An earlier version of this method grouped
        the GLOBAL depth by path, which is not the same quantity: it reported
        ``/v1/chat/completions: 7`` for a code path bounded at
        ``Semaphore(_INGEST_CONCURRENCY=5)``, because concurrent /api/embeddings
        calls inflated the global reading sampled at a chat request's start. Each
        request now carries ``in_flight_at_start_path`` from a per-path counter,
        and this method reads only that.
        """
        peaks: dict[str, int] = {}
        for record in self.records:
            key = str(record.get("path"))
            peaks[key] = max(peaks.get(key, 0), record.get("in_flight_at_start_path", 0))
        return peaks

    def max_in_flight_for(self, *path_fragments: str) -> int:
        """Peak PER-PATH concurrency across only the endpoints matching a fragment."""
        return max(
            (
                record.get("in_flight_at_start_path", 0)
                for record in self.records
                if any(frag in str(record.get("path")) for frag in path_fragments)
            ),
            default=0,
        )

    def latencies(self, *path_fragments: str) -> list[float]:
        """Per-request latencies, optionally restricted to matching endpoints."""
        return [
            r["ts_end"] - r["ts_start"]
            for r in self.records
            if r.get("ts_end") is not None
            and r.get("ts_start") is not None
            and (
                not path_fragments
                or any(frag in str(r.get("path")) for frag in path_fragments)
            )
        ]

    @staticmethod
    def _latency_stats(values: list[float]) -> dict[str, Any]:
        lat = sorted(values)
        return {
            "n": len(lat),
            "min": lat[0] if lat else None,
            "median": lat[len(lat) // 2] if lat else None,
            "max": lat[-1] if lat else None,
            "sum": sum(lat) if lat else 0.0,
        }

    def latency_seconds_by_path(self) -> dict[str, dict[str, Any]]:
        """Latency summary PER ENDPOINT.

        Pooling a 10-second chat call with a 2-millisecond embedding call gives a
        spread that describes neither. Anything that wants "the variance of an
        LLM call" must read a single endpoint's entry here.
        """
        buckets: dict[str, list[float]] = {}
        for record in self.records:
            if record.get("ts_end") is None or record.get("ts_start") is None:
                continue
            buckets.setdefault(str(record.get("path")), []).append(
                record["ts_end"] - record["ts_start"]
            )
        return {path: self._latency_stats(values) for path, values in buckets.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "ledger_file": str(self.path),
            "requests_total": self.total,
            "requests_by_path": self.by_path(),
            "max_in_flight": self.max_in_flight,
            "max_in_flight_note": (
                "max_in_flight is the GLOBAL process depth (an upper bound on any one "
                "endpoint); max_in_flight_by_path is counted per endpoint"
            ),
            "max_in_flight_by_path": self.max_in_flight_by_path(),
            "latency_seconds": self._latency_stats(self.latencies()),
            "latency_seconds_by_path": self.latency_seconds_by_path(),
        }


@contextmanager
def http_recorder(ledger_path: str | Path, *, truncate: bool = True) -> Iterator[HttpLedger]:
    """Record every httpx request this process makes, with in-flight depth.

    Patches ``httpx.AsyncClient.send`` for the duration of the block and restores
    it afterwards. Scope is deliberately this process only — that is exactly the
    scope the hypotheses are stated over.
    """
    import httpx

    target = Path(ledger_path)
    if truncate:
        truncate_ledger(target)

    ledger = HttpLedger(path=target)
    original = httpx.AsyncClient.send
    # TWO counters, deliberately. The global one answers "how many inference
    # requests did this process have open at once"; the per-path one answers
    # "which endpoint produced that fan-out". Grouping the global reading by path
    # answers neither, and reads as the second while measuring the first.
    global_depth = {"n": 0}
    depth_by_path: dict[str, int] = {}

    async def _recording_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        parts = urlsplit(str(request.url))
        path = parts.path
        global_depth["n"] += 1
        depth_by_path[path] = depth_by_path.get(path, 0) + 1
        record: dict[str, Any] = {
            "ts_start": time.time(),
            "ts_end": None,
            "method": request.method,
            "host": parts.netloc,
            "path": path,
            "global_depth_at_start": global_depth["n"],
            "in_flight_at_start_path": depth_by_path[path],
            "status": None,
            "error_type": None,
        }
        try:
            response = await original(self, request, **kwargs)
            record["status"] = response.status_code
            return response
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            global_depth["n"] -= 1
            depth_by_path[path] -= 1
            record["ts_end"] = time.time()
            ledger.records.append(record)
            # No fsync: this ledger is not read across a process kill, and
            # fsyncing every HTTP call would perturb the latencies it measures.
            append_jsonl(target, record, fsync=False)

    httpx.AsyncClient.send = _recording_send  # type: ignore[method-assign]
    try:
        yield ledger
    finally:
        httpx.AsyncClient.send = original  # type: ignore[method-assign]


def load_http_ledger(path: str | Path) -> HttpLedger:
    """Rebuild an :class:`HttpLedger` from a file written by a subprocess."""
    return HttpLedger(path=Path(path), records=read_jsonl(path))


# --------------------------------------------------------------------------
# Polling
# --------------------------------------------------------------------------


def poll_until(
    predicate: Callable[[], Any],
    *,
    timeout_seconds: float,
    interval_seconds: float = 0.25,
    on_poll: Callable[[Any], None] | None = None,
    description: str = "condition",
) -> Any:
    """Poll ``predicate`` until it returns something truthy, else FAIL LOUD.

    Timing out must never degrade into "act anyway at an arbitrary moment": a
    SIGKILL delivered before the conversion has committed would make the whole
    checkpoint claim vacuous, so the caller wants an exception, not a guess.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = predicate()
        if on_poll is not None:
            on_poll(value)
        if value:
            return value
        time.sleep(interval_seconds)
    raise TimeoutError(
        f"Timed out after {timeout_seconds:.0f}s waiting for {description}. "
        "Failing loud rather than proceeding from an unknown state."
    )
