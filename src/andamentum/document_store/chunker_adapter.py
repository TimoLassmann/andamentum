"""Adapter from ``andamentum.chunker.Unit`` to document_store's ``Chunk``.

The structural-first chunker carries an immediate ``Unit.title`` but not
the full heading breadcrumb. document_store's ``ChunkMetadataFields``
expects ``section_path`` like ``"Methods > ODE Solver"``. This module
recovers that breadcrumb by walking the chunker's own section tree, then
maps each ``Unit`` into a document_store ``Chunk``.

No new dependencies; everything used here already lives in
``andamentum.chunker.structural`` and ``andamentum.document_store.chunking``.
"""

from __future__ import annotations

from collections.abc import Iterable

from andamentum.chunker.structural import (
    Section,
    build_section_tree,
    find_headings,
)
from andamentum.chunker.types import Unit

from .chunking import Chunk

_PATH_SEP = " > "


def _path_for_offset(roots: list[Section], offset: int) -> list[str]:
    """Walk the section tree to find the deepest section containing `offset`.

    Returns the heading titles from the outermost containing section down
    to the deepest. Empty list if no section contains the offset (preamble
    or a source with no headings).
    """
    path: list[str] = []
    cursor: list[Section] | None = roots
    while cursor:
        # Find a child whose span contains `offset`. Sections within a level
        # are non-overlapping by construction (build_section_tree).
        match: Section | None = None
        for sec in cursor:
            if sec.start <= offset < sec.end:
                match = sec
                break
        if match is None:
            break
        path.append(match.title)
        cursor = match.children
    return path


def _compute_section_paths(content: str, units: Iterable[Unit]) -> list[str]:
    """Return one section_path per unit, ordered to match `units`.

    A unit's path is the heading breadcrumb of the deepest section that
    contains its ``source_start`` offset. Returns an empty string for
    units whose offset falls outside any heading (e.g. preamble) or for
    sources with no headings at all.
    """
    headings = find_headings(content)
    if not headings:
        return ["" for _ in units]

    roots = build_section_tree(content, headings)
    paths: list[str] = []
    for unit in units:
        path_titles = _path_for_offset(roots, unit.source_start)
        paths.append(_PATH_SEP.join(path_titles))
    return paths


def units_to_chunks(content: str, units: list[Unit]) -> list[Chunk]:
    """Convert chunker ``Unit`` records into document_store ``Chunk`` records.

    Preserves byte-identical spans — ``content[c.start_char:c.end_char] == c.text``
    — and recovers ``section_path`` from the chunker's own structural API.
    ``chunk_index`` is assigned by enumeration in input order.
    """
    paths = _compute_section_paths(content, units)
    return [
        Chunk(
            text=unit.text,
            section_path=path,
            chunk_index=i,
            start_char=unit.source_start,
            end_char=unit.source_end,
        )
        for i, (unit, path) in enumerate(zip(units, paths))
    ]
