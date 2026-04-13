"""Simple markdown chunker.

Splits markdown text into chunks by headers, then by token window.
No external dependencies — pure Python.

Strategy:
1. Split on markdown headers (# ## ### etc.) into sections
2. Each section inherits its heading path ("Methods > ODE Solver")
3. If a section fits within max_tokens, it's one chunk
4. If a section is too long, sliding window with overlap
5. Plain text with no headers: sliding window over entire text
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """A chunk of document content with positional metadata."""

    text: str
    section_path: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class _Section:
    """Internal: a markdown section between headers."""

    heading: str
    level: int
    path: str
    text: str
    start_char: int
    end_char: int


def _build_section_path(heading_stack: list[tuple[int, str]], level: int, heading: str) -> str:
    """Build a heading path like 'Methods > ODE Solver' from a stack."""
    # Pop headings at the same or deeper level
    while heading_stack and heading_stack[-1][0] >= level:
        heading_stack.pop()
    heading_stack.append((level, heading))
    return " > ".join(h for _, h in heading_stack)


def _split_into_sections(text: str) -> list[_Section]:
    """Split markdown into sections by header boundaries."""
    matches = list(_HEADER_RE.finditer(text))

    if not matches:
        # No headers — entire text is one section
        return [_Section(heading="", level=0, path="", text=text, start_char=0, end_char=len(text))]

    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []

    # Content before the first header (if any)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                _Section(heading="", level=0, path="", text=preamble, start_char=0, end_char=matches[0].start())
            )

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        path = _build_section_path(heading_stack, level, heading)

        # Section text: from end of this header line to start of next header (or end of text)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[content_start:content_end].strip()

        # Include the header line in the chunk text for context
        header_line = match.group(0)
        full_text = f"{header_line}\n{section_text}" if section_text else header_line

        sections.append(
            _Section(
                heading=heading,
                level=level,
                path=path,
                text=full_text,
                start_char=match.start(),
                end_char=content_end,
            )
        )

    return sections


def _window_split(text: str, max_chars: int, overlap_chars: int, start_offset: int = 0) -> list[tuple[str, int, int]]:
    """Split text into overlapping windows. Returns list of (text, start_char, end_char)."""
    if len(text) <= max_chars:
        return [(text, start_offset, start_offset + len(text))]

    stride = max_chars - overlap_chars
    if stride <= 0:
        stride = max_chars

    windows: list[tuple[str, int, int]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        window_text = text[pos:end]
        if window_text.strip():
            windows.append((window_text, start_offset + pos, start_offset + end))
        if end >= len(text):
            break
        pos += stride

    return windows


def chunk_markdown(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    chars_per_token: int = 4,
) -> list[Chunk]:
    """Split markdown into chunks.

    Args:
        text: Markdown text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks when windowing.
        chars_per_token: Character-to-token ratio estimate.

    Returns:
        List of Chunk objects with section_path and positional info.
    """
    if not text or not text.strip():
        return []

    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token

    sections = _split_into_sections(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        if len(section.text) <= max_chars:
            # Section fits in one chunk
            chunks.append(
                Chunk(
                    text=section.text,
                    section_path=section.path,
                    chunk_index=chunk_index,
                    start_char=section.start_char,
                    end_char=section.end_char,
                )
            )
            chunk_index += 1
        else:
            # Section too long — window split
            windows = _window_split(section.text, max_chars, overlap_chars, section.start_char)
            for window_text, start, end in windows:
                chunks.append(
                    Chunk(
                        text=window_text,
                        section_path=section.path,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

    return chunks
