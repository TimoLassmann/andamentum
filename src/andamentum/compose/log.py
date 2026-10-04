"""JSONL action log: ``<bundle>/log.jsonl``, one object per model call or event."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .schemas import Role

LOG_NAME = "log.jsonl"


def sha(text: str) -> str:
    """Short stable digest used for prompt / input / output fingerprints."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ActionLog:
    """Append-only log. Each line is flushed as written, so a crash keeps it."""

    def __init__(self, bundle_dir: str | Path) -> None:
        self.path = Path(bundle_dir) / LOG_NAME

    def _append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def call(
        self,
        *,
        node: str,
        role: Role,
        model: str,
        prompt: str,
        input_text: str,
        output_text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self._append(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "node": node,
                "role": role,
                "model": model,
                "prompt_sha": sha(prompt),
                "input_sha": sha(input_text),
                "output_sha": sha(output_text),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def event(self, *, node: str, event: str, **details: object) -> None:
        self._append(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "node": node,
                "event": event,
                **details,
            }
        )

    def records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def usage(self) -> dict[str, int]:
        """Tokens per role (input plus output); unavailable counts add nothing."""
        totals: dict[str, int] = {}
        for rec in self.records():
            role = rec.get("role")
            if not isinstance(role, str):
                continue
            n = 0
            for key in ("input_tokens", "output_tokens"):
                value = rec.get(key)
                if isinstance(value, int):
                    n += value
            totals[role] = totals.get(role, 0) + n
        return totals
