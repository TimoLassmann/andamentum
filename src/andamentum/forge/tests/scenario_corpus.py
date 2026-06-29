"""Scenario corpus: the rung ladder forge must build or refuse.

This is the acceptance fixture for the forge function-generator refinement.
It encodes every scenario from the PRD (C-STORE-PRD.md §9 "The functions forge
must be able to create") and the ladder rows in A-WHY-FUNCTIONS.md §2.

The rung ladder (A §2):
  rung 1 -> stateless function  (the system owns the loop, fixed path)
  rung 2 -> stateful function   (fixed path + durable load-at-start/save-at-end)
  rung 3 -> app                 (the caller chooses among operations)
  rung 4 -> agent               (the caller drives a multi-turn session)
  rung 5 -> service             (the world emits triggering events)

forge's scope is the line under rung 2: functions are built, everything above
is refused at the door with a concrete reshape (A §0, C §8). The disqualifying
axis is *external control* (A §2): an operation-chooser, a session, or an
event-source. "axis" names that driver for refuse cases; it is "" for in-scope
functions.

This module is a plain importable fixture, NOT a pytest test file (pytest only
collects ``test_*.py``); ``test_scenario_corpus.py`` validates it is well-formed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Rung = Literal["function", "stateful_function", "app", "agent", "service"]
Expected = Literal["build", "refuse"]


class Scenario(BaseModel):
    """One brief on the rung ladder, with forge's expected disposition.

    ``expected`` is "build" iff ``rung`` is a function rung
    ("function" or "stateful_function"); otherwise "refuse".
    """

    model_config = {"frozen": True}

    brief: str
    """The natural-language brief, as a user would phrase it."""

    rung: Rung
    """The system class the brief actually demands."""

    expected: Expected
    """forge's disposition: "build" for function rungs, "refuse" above them."""

    axis: str
    """The external driver that disqualifies an out-of-scope brief
    (e.g. "operation-chooser", "session", "event-source"); "" when in-scope."""

    note: str
    """One line. For refuse cases: the concrete reshape suggestion.
    For build cases: what the scenario exercises."""


CORPUS: list[Scenario] = [
    # --- rung 1: stateless functions (A §2, A §4 "summarise, classify, route,
    # extract, translate, research-until-enough") -----------------------------
    Scenario(
        brief="Summarise this document.",
        rung="function",
        expected="build",
        axis="",
        note="Pure stateless transform: one input, one output, nothing remembered.",
    ),
    Scenario(
        brief="Classify this support ticket and route it to the right team.",
        rung="function",
        expected="build",
        axis="",
        note="Classify-and-route: a fixed-path decision over one input.",
    ),
    Scenario(
        brief="Research X, looping until the evidence is sufficient.",
        rung="function",
        expected="build",
        axis="",
        note="Bounded internal loop (generate/check/retry) is still rung 1: "
        "the loop is internal control flow, not an external driver.",
    ),
    Scenario(
        brief="Extract the recommendations from this chat message.",
        rung="function",
        expected="build",
        axis="",
        note="Extraction over one input; the rung-1 function hiding inside "
        "'manage my reading list'.",
    ),
    Scenario(
        brief="Translate this text into French.",
        rung="function",
        expected="build",
        axis="",
        note="Pure stateless transform; canonical rung-1 example (A §4).",
    ),
    # --- rung 2: stateful functions (C §9 "must build", A §2) -----------------
    Scenario(
        brief="Given my current reading list and a new chat message, "
        "return the updated list.",
        rung="stateful_function",
        expected="build",
        axis="",
        note="Single-record entity, constant key: load the list, apply the "
        "message, save the list, return it (C §9.1).",
    ),
    Scenario(
        brief="Append a note to my notebook and return the new total count.",
        rung="stateful_function",
        expected="build",
        axis="",
        note="Multi-record entity, uuid key: add the note, then "
        "len(list(coll)); count must increase across file-backed runs (C §9.2).",
    ),
    Scenario(
        brief="Record this outcome, then report how many outcomes have been recorded.",
        rung="stateful_function",
        expected="build",
        axis="",
        note="Same shape as append-note-return-count; confirms the "
        "load/compute/save pattern generalises (C §9.3).",
    ),
    Scenario(
        brief="Update the saved record for this id with this change and return it.",
        rung="stateful_function",
        expected="build",
        axis="",
        note="Multi-record entity: get-by-id then add-by-id (overwrite) "
        "via the same add verb (C §9.4).",
    ),
    # --- rung 3+: must refuse + reshape (C §9, A §2) --------------------------
    Scenario(
        brief="Manage my personal reading list.",
        rung="app",
        expected="refuse",
        axis="operation-chooser",
        note="Reshape -> 'given my current list and a new message, return the "
        "updated list' (rung 2) or 'extract the recommendations from this "
        "message' (rung 1) (C §9.5, A §5).",
    ),
    Scenario(
        brief="A chatbot that answers questions about my documents.",
        rung="agent",
        expected="refuse",
        axis="session",
        note="Reshape -> 'answer ONE question about a document set' (rung 1) (C §9.6).",
    ),
    Scenario(
        brief="Watch my inbox and file incoming mail.",
        rung="service",
        expected="refuse",
        axis="event-source",
        note="Reshape -> 'classify ONE email into a folder' (rung 1) (C §9.7).",
    ),
    Scenario(
        brief="Find all my notes mentioning a given topic.",
        rung="app",
        expected="refuse",
        axis="operation-chooser",
        note="Beyond the closed store surface (content query). Reshape -> "
        "rung-1 'given these notes and a topic, return the matching ones', "
        "or point at document_store for find-by-meaning (C §9.8).",
    ),
]
