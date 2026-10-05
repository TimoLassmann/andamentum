"""Builders for the bindings tests: a synthetic manuscript and result files (invented)."""

from __future__ import annotations

from pathlib import Path

from andamentum.compose.harness.spike.make import build_docx, paragraph, run

EFFECT = (
    "In the SaltLog pilot the mean sodium intake fell by 23.4% in the intervention arm."
)
SAMPLE = "A total of 1,234 meals were logged by 159 participants over twelve weeks."
PVALUE = "The reduction was significant (t(157) = 3.10, p = .03) in the pilot."
INTERVAL = "The adjusted odds ratio was 1.40 (95% CI [1.10, 1.80]) for high intake."
FILLER = "This sentence has no numbers in it at all, so nothing is examined."

PARAGRAPHS = [
    ("SaltLog pilot", "Heading1"),
    (EFFECT, None),
    (SAMPLE, None),
    (PVALUE, None),
    (INTERVAL, None),
    (FILLER, None),
]


def make_manuscript(path: Path, paragraphs=PARAGRAPHS) -> Path:
    body = "".join(paragraph(run(text), style=style) for text, style in paragraphs)
    return build_docx(path, body)


def make_results(folder: Path, *, effect: str = "0.234", p: str = "0.03") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.csv").write_text(
        f"measure,value\neffect,{effect}\nmeals,1234\nparticipants,159\n", "utf-8"
    )
    (folder / "model.json").write_text(
        '{"t": {"p": %s, "odds_ratio": 1.4, "ci_low": 1.1, "ci_high": 1.8}}' % p,
        "utf-8",
    )
    return folder
