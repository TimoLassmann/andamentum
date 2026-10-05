"""The claims ledger (``claims.yaml`` in the project store), as far as bindings write it.

One entry per bound number: the sentence, ``kind: result``, an evidence pointer
to the result file and key, and a ``binding`` block that lets ``refresh`` write
the number again in the same shape. Entries written by anyone else are kept
byte for byte in meaning: this module reads the whole file as plain mappings,
changes only its own entries (those with a ``binding`` block) and writes the
rest back.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from ..kernel import ProjectStore

CLAIMS_FILE = "claims.yaml"


class NumberBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: str
    sentence: str
    file: str
    key: str
    value: float  # as stored in the result file
    context: str = ""  # the row label the key had when bound (csv rows)
    written: str  # the text the number had when it was bound or last refreshed
    scale: float  # shown value = stored value * scale (100 for a fraction shown as %)


def _entry(binding: NumberBinding) -> dict[str, object]:
    return {
        "id": binding.anchor_id,
        "text": binding.sentence,
        "kind": "result",
        "evidence": [
            {
                "type": "result_number",
                "file": binding.file,
                "key": binding.key,
                "value": binding.value,
            }
        ],
        "strength": "matches result file",
        "status": "bound",
        "approved_by": None,
        "binding": binding.model_dump(),
    }


class Ledger:
    def __init__(self, store: ProjectStore) -> None:
        self.path = store.path / CLAIMS_FILE

    def _read(self) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list):
            raise ValueError(f"{self.path} must hold a mapping with a 'claims' list")
        return claims

    def _write(self, claims: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump({"claims": claims}, sort_keys=False, allow_unicode=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def bindings(self) -> dict[str, NumberBinding]:
        """Every number binding, by anchor id."""
        found: dict[str, NumberBinding] = {}
        for entry in self._read():
            block = entry.get("binding")
            if isinstance(block, dict):
                binding = NumberBinding.model_validate(block)
                found[binding.anchor_id] = binding
        return found

    def record(self, bindings: list[NumberBinding]) -> None:
        """Add or replace the entries of *bindings* (matched by anchor id)."""
        by_id = {b.anchor_id: b for b in bindings}
        claims = [
            c
            for c in self._read()
            if not (isinstance(c.get("binding"), dict) and c["id"] in by_id)
        ]
        claims.extend(_entry(b) for b in bindings)
        self._write(claims)
