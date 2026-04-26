"""Window slicing with lookahead.

A `Window` is a slice of the source `text` starting at `cursor`,
spanning `window_size` chars (the region the cursor will advance INTO),
plus a lookahead region the model can SEE but cannot claim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Window:
    """A window of source text + lookahead.

    `text` is the substring sent to the model.
    `cursor` is the absolute offset where this window starts in the source.
    `window_end_offset` is `min(cursor + window_size, len(source))`.
    `full_end_offset` is `min(cursor + window_size + lookahead, len(source))`.
    """

    text: str
    cursor: int
    window_end_offset: int
    full_end_offset: int


def make_window(
    source: str,
    *,
    cursor: int,
    window_size: int,
    lookahead: int,
) -> Window:
    """Slice `source` starting at `cursor` with the given size + lookahead."""
    window_end = min(cursor + window_size, len(source))
    full_end = min(cursor + window_size + lookahead, len(source))
    return Window(
        text=source[cursor:full_end],
        cursor=cursor,
        window_end_offset=window_end,
        full_end_offset=full_end,
    )
