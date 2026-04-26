"""Tests for window slicing + cursor mechanics."""

from andamentum.chunker.windowing import (
    Window,
    make_window,
)


def test_window_within_bounds():
    src = "x" * 30_000
    w = make_window(src, cursor=0, window_size=10_000, lookahead=4_000)
    assert isinstance(w, Window)
    assert w.text == src[0:14_000]
    assert w.window_end_offset == 10_000
    assert w.full_end_offset == 14_000


def test_window_at_end_of_source():
    src = "x" * 5_000
    w = make_window(src, cursor=0, window_size=10_000, lookahead=4_000)
    # Source is shorter than window — return what we have
    assert w.text == src
    assert w.window_end_offset == 5_000  # capped
    assert w.full_end_offset == 5_000


def test_window_partial_lookahead_at_end():
    src = "x" * 12_000
    w = make_window(src, cursor=0, window_size=10_000, lookahead=4_000)
    # Window fits, lookahead is partial
    assert w.text == src
    assert w.window_end_offset == 10_000
    assert w.full_end_offset == 12_000


def test_window_starting_mid_source():
    src = "y" * 30_000
    w = make_window(src, cursor=5_000, window_size=10_000, lookahead=4_000)
    assert w.text == src[5_000:19_000]
    assert w.cursor == 5_000
    assert w.window_end_offset == 15_000  # cursor + window_size
    assert w.full_end_offset == 19_000
