"""Concatenate the per-rule event logs, in rule order, into JSONL + a flat CSV.

Interoperable and self-describing: every headline count becomes re-derivable
offline from plain text rather than trusted as a summary.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.merge_events
"""

from __future__ import annotations

import csv
from typing import Any

from . import _common as C
from .events import EVENT_FIELDS
from .instrument import read_jsonl

#: Rule order — deliberately the execution order, so the concatenation reads as a
#: timeline rather than an alphabetical jumble.
RULE_ORDER = (
    "claim_a.jsonl",
    "claim_b.jsonl",
    "claim_c_kill.jsonl",
    "claim_c_resume.jsonl",
    "claim_d.jsonl",
    "claim_f.jsonl",
)


def main() -> int:
    records: list[dict[str, Any]] = []
    present: list[str] = []
    for name in RULE_ORDER:
        path = C.EVENTS / name
        if not path.exists():
            continue
        present.append(name)
        records.extend(read_jsonl(path))

    out_jsonl = C.RESULTS / "events.jsonl"
    with out_jsonl.open("w") as fh:
        for record in records:
            fh.write(__import__("json").dumps(record, default=str) + "\n")

    out_csv = C.RESULTS / "events.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EVENT_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in EVENT_FIELDS})

    print(f"merged {len(records)} events from {len(present)} logs -> {out_jsonl}, {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
