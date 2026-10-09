"""The statement on the use of AI, written from the project's event log.

Nothing in the statement is asserted from memory: each sentence is built from
events the run recorded (``draft_step`` for every model call that shaped the
text, ``check_run`` for every check pass, ``draft_run`` for what a run put in
the document) and a sentence whose events are absent is left out. It is one
paragraph, inserted under :data:`HEADING` as a tracked insertion, and also goes
into the run report.
"""

from __future__ import annotations

import re

from collections.abc import Sequence

HEADING = "Statement on the use of AI"
SETUP_STEPS = ("brief", "evidence", "outline")
SEARCHED = {
    "web": "the web",
    "scholarly": "scholarly databases",
    "both": "the web and scholarly databases",
}


def _listing(items: Sequence[str]) -> str:
    unique = list(dict.fromkeys(items))
    if len(unique) <= 2:
        return " and ".join(unique)
    return ", ".join(unique[:-1]) + " and " + unique[-1]


def _by_model(pairs: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for model, title in pairs:
        if title not in grouped.setdefault(model, []):
            grouped[model].append(title)
    return grouped


def _sections(titles: list[str]) -> str:
    return ("Section " if len(titles) == 1 else "Sections ") + _listing(
        [f'"{t}"' for t in titles]
    )


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _count(value: object) -> int:
    return value if isinstance(value, int) else 0


def _check_ids(events: Sequence[dict[str, object]]) -> list[str]:
    ids: list[str] = []
    for event in events:
        listed = event.get("checks")
        if event.get("type") == "check_run" and isinstance(listed, list):
            ids += [str(item) for item in listed]
    return ids


def ai_use_statement(events: Sequence[dict[str, object]]) -> str:
    """One paragraph describing what the AI did, from the project's *events*."""
    steps = [e for e in events if e.get("type") == "draft_step"]
    setup_models = [
        _text(e.get("model")) for e in steps if _text(e.get("step")) in SETUP_STEPS
    ]
    drafted = _by_model(
        [
            (_text(e.get("model")), _text(e.get("detail")))
            for e in steps
            if e.get("step") == "draft"
        ]
    )
    revised = _by_model(
        [
            (_text(e.get("model")), _text(e.get("detail")))
            for e in steps
            if e.get("step") == "revise"
        ]
    )
    researched = [e for e in steps if e.get("step") == "research"]
    cite_steps = [e for e in steps if e.get("step") == "cite"]
    checks = _check_ids(events)
    runs = [e for e in events if e.get("type") == "draft_run"]
    citations = sum(_count(e.get("citations")) for e in runs)
    bound = sum(_count(e.get("numbers_bound")) for e in runs)

    sentences = [
        "This document was drafted with the help of an AI system (andamentum compose)."
    ]
    if setup_models:
        sentences.append(
            "The brief, the extraction of facts from the supplied sources and the "
            f"outline were produced by {_listing(setup_models)}."
        )
    if researched:
        where = _listing(
            [SEARCHED.get(_text(e.get("detail")), "its sources") for e in researched]
        )
        models = _listing([_text(e.get("model")) for e in researched])
        sentences.append(
            f"Further evidence was found by searching {where}: {models} planned "
            "the search and read the works it found, and only passages quoted "
            "word for word from those works were used."
        )
    for model, titles in drafted.items():
        sentences.append(
            f"{_sections(titles)} {'was' if len(titles) == 1 else 'were'} drafted by {model}."
        )
    for step in cite_steps:
        model = _text(step.get("model"))
        counts = _cite_counts(_text(step.get("detail")))
        checked, cited = counts.get("checked", 0), counts.get("cited", 0)
        noun = "sentence" if checked == 1 else "sentences"
        if cited == 0:
            sentences.append(
                f"After drafting, {model} checked {checked} {noun} that cited "
                "nothing and found no supporting paper, so none was cited."
            )
            continue
        changed = counts.get("changed", 0)
        text = (
            f"After drafting, {model} checked {checked} {noun} that cited nothing, "
            "read the papers found and cited one only where a passage of it, "
            f"quoted word for word, supports the sentence; {cited} "
            f"{'sentence was' if cited == 1 else 'sentences were'} cited"
        )
        if changed:
            text += (
                f", {changed} of them changed to match a narrower claim in the paper"
            )
        sentences.append(text + ".")
    for model, titles in revised.items():
        sentences.append(
            f"{_sections(titles)} {'was' if len(titles) == 1 else 'were'} "
            f"revised by {model} after automated checks reported errors."
        )
    if citations:
        sentences.append(
            "Citations were inserted only for works whose DOI and metadata were "
            "verified and that are not retracted."
        )
    if bound:
        sentences.append(
            "Numbers taken from result files are bound to those files so that they "
            "can be refreshed when the results change."
        )
    if checks:
        sentences.append(f"Automated checks run on the draft: {_listing(checks)}.")
    sentences.append(
        "Statements that need the authors' judgement are marked with comments. "
        "The authors are responsible for the content of this document."
    )
    return " ".join(sentences)


def _cite_counts(detail: str) -> dict[str, int]:
    """The counts in a cite step's detail ("3 checked, 1 cited, 1 changed")."""
    return {
        word: int(number)
        for number, word in re.findall(r"(\d+) (checked|cited|changed)", detail)
    }
