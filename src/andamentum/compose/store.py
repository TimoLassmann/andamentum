"""The durable run artefacts of a compose bundle, as plain JSON files.

The bundle on disk is the store (dialect L1): the brief, sources, facts, outline
and round records live beside the bundle sections, so a run can be resumed or
inspected from the folder alone. Writes are atomic and idempotent.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import TypeAdapter

from .schemas import Brief, Fact, Outline, RoundRecord, Source

BRIEF = "brief.json"
OUTLINE = "outline.json"
SOURCES = "evidence/sources.json"
FACTS = "evidence/facts.json"
ROUNDS = "rounds.json"

_SOURCES = TypeAdapter(list[Source])
_FACTS = TypeAdapter(list[Fact])
_ROUNDS = TypeAdapter(list[RoundRecord])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def has(bundle_dir: str | Path, name: str) -> bool:
    return (Path(bundle_dir) / name).is_file()


def _read(bundle_dir: str | Path, name: str) -> str:
    return (Path(bundle_dir) / name).read_text(encoding="utf-8")


def save_brief(bundle_dir: str | Path, brief: Brief) -> None:
    _write(Path(bundle_dir) / BRIEF, brief.model_dump_json(indent=2))


def load_brief(bundle_dir: str | Path) -> Brief:
    return Brief.model_validate_json(_read(bundle_dir, BRIEF))


def save_outline(bundle_dir: str | Path, outline: Outline) -> None:
    _write(Path(bundle_dir) / OUTLINE, outline.model_dump_json(indent=2))


def load_outline(bundle_dir: str | Path) -> Outline:
    return Outline.model_validate_json(_read(bundle_dir, OUTLINE))


def save_sources(bundle_dir: str | Path, sources: list[Source]) -> None:
    _write(Path(bundle_dir) / SOURCES, _SOURCES.dump_json(sources, indent=2).decode())


def load_sources(bundle_dir: str | Path) -> list[Source]:
    return _SOURCES.validate_json(_read(bundle_dir, SOURCES))


def save_facts(bundle_dir: str | Path, facts: list[Fact]) -> None:
    _write(Path(bundle_dir) / FACTS, _FACTS.dump_json(facts, indent=2).decode())


def load_facts(bundle_dir: str | Path) -> list[Fact]:
    return _FACTS.validate_json(_read(bundle_dir, FACTS))


def save_rounds(bundle_dir: str | Path, rounds: list[RoundRecord]) -> None:
    _write(Path(bundle_dir) / ROUNDS, _ROUNDS.dump_json(rounds, indent=2).decode())


def load_rounds(bundle_dir: str | Path) -> list[RoundRecord]:
    if not has(bundle_dir, ROUNDS):
        return []
    return _ROUNDS.validate_json(_read(bundle_dir, ROUNDS))
