"""The result files as a catalogue of addressable numbers.

``checks.numbers.load_results`` reads every number of every csv, tsv and json
file under a folder and labels each with where it came from (``file row 2
column 'effect'``, ``file $.models[0].auc``). That label is the key a binding
keeps: positional, so a csv that gains a row above the bound one moves the key
to another number. A key that no longer exists is reported by ``refresh``; one
that now points at a different number cannot be told apart from a changed value.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ..checks.numbers import _SUFFIXES, _to_float, load_results

_ROW = re.compile(r" row (\d+) column ")


@dataclass(frozen=True)
class ResultEntry:
    file: str  # path relative to the results folder
    key: str  # the label ``load_results`` gives (starts with the file)
    value: float
    #: The row's first cell when it is text (``effect`` in ``effect,0.234``), so
    #: a reader of the key can tell rows apart and ``refresh`` can notice that a
    #: positional key now points at another row. Empty for json and numeric rows.
    context: str = ""


@dataclass(frozen=True)
class Catalogue:
    entries: list[ResultEntry]
    digest: str

    def get(self, key: str) -> ResultEntry | None:
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None


def _row_labels(directory: Path, files: list[str]) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    for file in files:
        delimiter = _SUFFIXES[Path(file).suffix.lower()]
        if delimiter is None:
            continue
        lines = (directory / file).read_text(encoding="utf-8").splitlines()
        for row_number, row in enumerate(csv.reader(lines, delimiter=delimiter), 1):
            if row_number > 1 and row and row[0].strip() and _to_float(row[0]) is None:
                labels[(file, row_number)] = row[0].strip()
    return labels


def _axis_keys(directory: Path, files: list[str]) -> set[str]:
    """Keys of the numeric first-column cells of csv and tsv tables.

    In a table with several columns and rows, a numeric first column (week,
    dose, id) labels the rows; it is not a result, and a text number such as
    "week 4" must not be bound to it.
    """
    keys: set[str] = set()
    for file in files:
        delimiter = _SUFFIXES[Path(file).suffix.lower()]
        if delimiter is None:
            continue
        lines = (directory / file).read_text(encoding="utf-8").splitlines()
        rows = list(csv.reader(lines, delimiter=delimiter))
        if len(rows) < 3 or len(rows[0]) < 2:
            continue
        for row_number, row in enumerate(rows[1:], 2):
            if row and _to_float(row[0]) is not None:
                keys.add(f"{file} row {row_number} column {rows[0][0]!r}")
    return keys


def load_catalogue(directory: Path) -> Catalogue:
    """Every number in the result files under *directory*, except row labels."""
    numbers, digests = load_results(directory)
    files = sorted(
        (
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUFFIXES
        ),
        key=len,
        reverse=True,
    )
    labels = _row_labels(directory, files)
    axes = _axis_keys(directory, files)
    entries = []
    for number in numbers:
        if number.where in axes:
            continue
        file = next(
            (f for f in files if number.where == f or number.where.startswith(f + " ")),
            None,
        )
        if file is None:
            raise ValueError(f"cannot tell which result file holds {number.where!r}")
        row = _ROW.search(number.where)
        context = labels.get((file, int(row[1])), "") if row else ""
        entries.append(
            ResultEntry(
                file=file, key=number.where, value=number.value, context=context
            )
        )
    digest = hashlib.sha256("".join(digests).encode()).hexdigest()
    return Catalogue(entries=entries, digest=digest)
