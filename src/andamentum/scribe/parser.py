"""Lightweight inline parsers for scribe markdown bodies.

For v1 we cover three concerns:
  - Pandoc-style citation spans: `[@key]` or `[@k1; @k2; @k3]`
  - Unresolved citation markers: `[verify]`, `[citation needed]`
    (compatible with the manuscript-tools/section-draft convention)
  - Inline formatting runs (bold/italic/code) for the docx renderer

We deliberately do NOT build a full markdown AST here — that's
typeset's job (via python-markdown). This module is for structural
extraction and cheap inline-styling tasks only.
"""

from __future__ import annotations

import re

# Match @key inside [...] but reject @keys that are part of an email
# (i.e. preceded by alphanumeric — `me@example.com`).
_CITATION_KEY_RE = re.compile(r"(?<![A-Za-z0-9])@([A-Za-z][A-Za-z0-9_:.-]*)")
_BRACKET_GROUP_RE = re.compile(r"\[(?P<body>[^\[\]]*)\]")
_UNRESOLVED_MARKERS = ("verify", "citation needed")

# Inline-formatting tokenizer: order matters — code spans first (so we
# don't mis-tokenise asterisks inside backticks), then bold, then italic.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")


def extract_citation_keys(markdown: str) -> list[str]:
    """Return citation keys (in order, preserving duplicates) from a body."""
    keys: list[str] = []
    for match in _BRACKET_GROUP_RE.finditer(markdown):
        body = match.group("body")
        for k in _CITATION_KEY_RE.findall(body):
            keys.append(k)
    return keys


def find_unresolved_markers(markdown: str) -> list[str]:
    """Return unresolved-citation markers found in body.

    Recognises `[verify]` and `[citation needed]`, the convention used
    by manuscript-tools/section-draft.
    """
    out: list[str] = []
    for match in _BRACKET_GROUP_RE.finditer(markdown):
        body = match.group("body").strip().lower()
        if body in _UNRESOLVED_MARKERS:
            out.append(body)
    return out


def inline_runs(markdown: str) -> list[tuple[str, set[str]]]:
    """Tokenise paragraph markdown into (text, styles) runs.

    Styles in `{"bold", "italic", "code"}`. Plain text has an empty
    set. Used by the docx renderer to emit styled python-docx runs.
    Unsupported markdown (links, etc.) falls back to plain text.
    """
    spans: list[tuple[int, int, str]] = []  # (start, end, style)
    for m in _CODE_RE.finditer(markdown):
        spans.append((m.start(), m.end(), "code"))
    for m in _BOLD_RE.finditer(markdown):
        if any(s <= m.start() < e for s, e, _ in spans):
            continue
        spans.append((m.start(), m.end(), "bold"))
    for m in _ITALIC_RE.finditer(markdown):
        if any(s <= m.start() < e for s, e, _ in spans):
            continue
        spans.append((m.start(), m.end(), "italic"))

    spans.sort(key=lambda t: t[0])

    runs: list[tuple[str, set[str]]] = []
    cursor = 0
    for start, end, style in spans:
        if start > cursor:
            runs.append((markdown[cursor:start], set()))
        if style == "code":
            text = markdown[start + 1 : end - 1]
        elif style == "bold":
            text = markdown[start + 2 : end - 2]
        else:  # italic
            text = markdown[start + 1 : end - 1]
        runs.append((text, {style}))
        cursor = end
    if cursor < len(markdown):
        runs.append((markdown[cursor:], set()))
    return runs
