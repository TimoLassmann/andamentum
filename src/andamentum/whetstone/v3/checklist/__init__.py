"""Checklist mode — the inspector beside review mode's reader.

Review mode asks evaluative questions ("is the evaluation adequate?"). A small
model answers "yes, broadly, though the reader might benefit from…" and
produces nothing you can act on. Checklist mode asks **defect-scoped extraction
under a quota** ("find the 3 claims the reported results support least well;
quote each one") — a task the same model does reliably, returning the same
shape every run.

Reach for checklist mode when the question is *"run the standard check"*:
pre-submission sweeps, comparing draft v2 against v3, anything on a schedule.
Reach for review mode when the question is *"help me improve this draft"* — the
cascade threads findings forward, asks open questions, and notices defects no
taxonomy names. Neither replaces the other; the cost of having both is one
extra run.

What you get here that review mode cannot give:

- **Comparability.** Fixed slots, deterministic quote anchoring, and a summary
  that is computed rather than written: one table row per probe, in bank order,
  never a function of what was found. Two runs — over the same draft or over
  two versions of it — line up row for row, so a ``diff`` reads as a change in
  the findings rather than a change in the report's shape.
- **Predictable cost.** One agent call per probe, with its tool turns bounded
  by a per-probe usage limit, on top of the shared claim-extraction pass. No
  gap loop, no consolidation call, no synthesis call. Only
  ``document_type="auto"`` adds a call, for the classifier — which is why it
  is not the default here, unlike review mode.
- **Co-detection as signal.** Two probes landing on the same anchor is recorded
  (``corroborated_by``), not merged away.
- **A crashed probe reads ``not run``, never ``0``.** "Looked and found
  nothing" and "never looked" are opposite results, and the coverage table
  keeps them apart in every cell.

The machinery is fixed and the probe list is data: pass ``probes=[...]`` to run
a house checklist, or let ``document_type`` route to one of the six default
banks (``"general"`` unless you say otherwise; the banks are exported by name
so you can build on one instead of replacing it). Output is a plain
``whetstone.schemas.ReviewResult``, so all three renderers consume it
unchanged.

Both entry points enforce the confidentiality-marker tripwire before any model
is called — like the rest of whetstone, this is for drafts you wrote yourself.
"""

from .bank import (
    ACADEMIC_PROBES,
    CREATIVE_PROBES,
    ESSAY_PROBES,
    EXTERNAL_COMMUNICATION_PROBES,
    GENERAL_PROBES,
    TUTORIAL_PROBES,
    probe_bank_for,
)
from .graph import checklist_document_v3, run_checklist_v3
from .probes import Probe, ProbeFinding

__all__ = [
    "run_checklist_v3",
    "checklist_document_v3",
    "Probe",
    "ProbeFinding",
    "probe_bank_for",
    "ACADEMIC_PROBES",
    "GENERAL_PROBES",
    "EXTERNAL_COMMUNICATION_PROBES",
    "ESSAY_PROBES",
    "TUTORIAL_PROBES",
    "CREATIVE_PROBES",
]
