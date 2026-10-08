"""Tests for the deterministic chunker: examples and seeded property checks."""

import random

import pytest

from andamentum.chunker import Chunk, chunk


def _texts(chunks):
    return [c.text for c in chunks]


def test_empty_and_whitespace_give_no_chunks():
    assert chunk("", max_chars=100) == []
    assert chunk(" \n\n \t\n", max_chars=100) == []


def test_max_chars_must_be_positive():
    with pytest.raises(ValueError):
        chunk("text", max_chars=0)


def test_small_document_is_one_chunk_with_offsets():
    src = "\n\nHello world.\n\n"
    [c] = chunk(src, max_chars=100)
    assert c == Chunk("Hello world.", 2, 14, ())
    assert src[c.start : c.end] == c.text


def test_sections_and_heading_paths():
    src = "intro\n\n# A\n\na body\n\n## B\n\nb body\n\n## C ##\n\nc body\n\n# D\n\nd body\n"
    chunks = chunk(src, max_chars=20)
    assert [c.heading_path for c in chunks] == [
        (),
        ("A",),
        ("A", "B"),
        ("A", "C"),
        ("D",),
    ]
    assert chunks[1].text == "# A\n\na body"


def test_heading_in_code_fence_is_not_a_heading():
    src = "# Real\n\n```\n# not a heading\n\ncode\n```\n\ntail\n"
    chunks = chunk(src, max_chars=10_000)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Real",)


def test_fence_closing_rules():
    # A shorter or different-character fence does not close the block.
    src = "# T\n\n````\n```\n# inside\n~~~\n````\n\n## After\n\nx\n"
    chunks = chunk(src, max_chars=10_000)
    assert [c.heading_path for c in chunks] == [("T",), ("T", "After")]
    assert "# inside" in chunks[0].text


def test_unclosed_fence_runs_to_end():
    src = "# T\n\n```\ncode\n\n# still code\n"
    chunks = chunk(src, max_chars=5)
    assert len(chunks) == 1
    assert chunks[0].text == src.strip()


def test_atomic_blocks_stay_whole_and_may_exceed_max():
    code = "```\n" + "line\n\n" * 30 + "```"
    table = "\n".join(f"| a{i} | b{i} |" for i in range(20))
    src = f"intro text.\n\n{code}\n\nmiddle text.\n\n{table}\n\nend text.\n"
    chunks = chunk(src, max_chars=50)
    assert code in _texts(chunks)
    assert table in _texts(chunks)
    big = [c for c in chunks if len(c.text) > 50]
    assert {c.text for c in big} == {code, table}


def test_front_matter_is_atomic_and_its_hashes_are_not_headings():
    src = "---\ntitle: x\n# comment\n---\n\nBody text here.\n"
    chunks = chunk(src, max_chars=15)
    assert chunks[0].text == "---\ntitle: x\n# comment\n---"
    assert all(c.heading_path == () for c in chunks)


def test_heading_never_stands_alone():
    src = "# Title\n\n## Sub\n\nbody\n\n## Other\n\n" + "word " * 30
    chunks = chunk(src, max_chars=40)
    assert not any(
        c.text.strip().startswith("#") and "\n" not in c.text for c in chunks
    )
    assert chunks[0].text.startswith("# Title\n\n## Sub")
    assert chunks[0].heading_path == ("Title", "Sub")


def test_last_empty_heading_is_kept():
    chunks = chunk("# A\n\ntext\n\n# B\n", max_chars=5)
    assert chunks[-1].text == "# B"


def test_paragraphs_then_sentences_then_whitespace():
    sent = "This is a sentence of moderate length."
    src = " ".join([sent] * 6)
    chunks = chunk(src, max_chars=len(sent) * 2 + 2)
    assert all(len(c.text) <= len(sent) * 2 + 2 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    long_sentence = "word " * 50
    chunks = chunk(long_sentence, max_chars=33)
    assert all(len(c.text) <= 33 for c in chunks)
    assert all(not c.text.endswith(" ") for c in chunks)


def test_unbreakable_token_is_hard_cut():
    chunks = chunk("x" * 25, max_chars=10)
    assert [len(c.text) for c in chunks] == [10, 10, 5]


def test_deterministic():
    src = "# A\n\n" + "Sentence one. Sentence two. " * 40 + "\n\n## B\n\nmore\n"
    assert chunk(src, max_chars=120) == chunk(src, max_chars=120)


# -- property checks over seeded random markdown ----------------------------


def _random_markdown(rng: random.Random) -> tuple[str, bool]:
    words = ["alpha", "beta", "gamma", "delta", "Epsilon", "zeta", "eta", "theta"]

    def sentence() -> str:
        n = rng.randint(2, 40)
        return " ".join(rng.choice(words) for _ in range(n)).capitalize() + rng.choice(
            ".!?"
        )

    parts: list[str] = []
    for _ in range(rng.randint(1, 14)):
        kind = rng.choice(
            ["heading", "para", "para", "fence", "table", "long", "tilde"]
        )
        if kind == "heading":
            parts.append(
                "#" * rng.randint(1, 4)
                + " "
                + rng.choice(words)
                + rng.choice(["", " ##"])
            )
        elif kind == "para":
            parts.append(" ".join(sentence() for _ in range(rng.randint(1, 5))))
        elif kind == "long":
            parts.append(" ".join(sentence() for _ in range(rng.randint(8, 20))))
        elif kind == "fence":
            body = ["# not a heading", "", "code line", "", "```not closing? no"]
            rng.shuffle(body)
            parts.append("```\n" + "\n".join(body[: rng.randint(1, 5)]) + "\n```")
        elif kind == "tilde":
            parts.append("~~~py\n# comment\n\nx = 1\n~~~")
        else:
            parts.append(
                "\n".join(f"| c{i} | d{i} |" for i in range(rng.randint(1, 6)))
            )
    sep = rng.choice(["\n\n", "\n\n\n", "\n \n"])
    src = sep.join(parts)
    if rng.random() < 0.3:
        src = "---\nk: v\n---\n\n" + src
    if rng.random() < 0.3:
        src = "\n\n" + src + "\n\n"
    return src, True


def _atomic_text(c: Chunk) -> bool:
    lines = [ln for ln in c.text.split("\n") if ln.strip()]
    while lines and lines[0].startswith("#") and len(lines) > 1:
        lines.pop(0)
    return lines[0].startswith(("```", "~~~", "|", "---"))


@pytest.mark.parametrize("seed", range(100))
def test_invariants_on_random_markdown(seed):
    rng = random.Random(seed)
    src, _ = _random_markdown(rng)
    max_chars = rng.choice([30, 60, 120, 400, 5000])
    chunks = chunk(src, max_chars=max_chars)

    prev_end = 0
    covered = [False] * len(src)
    for c in chunks:
        assert c.text == src[c.start : c.end]
        assert c.start >= prev_end, "chunks overlap or are out of order"
        assert c.text == c.text.strip("\n") and c.text.strip()
        assert not c.text.startswith(("\n", " \n"))
        prev_end = c.end
        for i in range(c.start, c.end):
            covered[i] = True
        if len(c.text) > max_chars:
            assert _atomic_text(c) or not any(ch.isspace() for ch in c.text), c.text
    for i, ch in enumerate(src):
        if not ch.isspace():
            assert covered[i], f"character {i} ({ch!r}) is in no chunk"
    assert chunks == chunk(src, max_chars=max_chars)
