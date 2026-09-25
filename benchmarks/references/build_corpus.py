"""Build a test set of Word documents, one per real citation style.

Real styles come from the Citation Style Language repository; real reference
metadata from Crossref (CSL-JSON by DOI). ``citeproc-py`` renders each work's
in-text citation and bibliography entry in each style, and every document
carries the same planted errors and the same distractors, with the answers
written beside it:

- a work cited in the text but missing from the list;
- a work in the list that is never cited;
- in author-year styles, one citation whose year is off by one;
- distractors a reader must not mistake for citations: affiliation marks in
  the author block, "10⁵ cells", "3 m²", "Fig. 2", "(n = 12)" and a numbered
  step list "(1) ... (2) ...".

Each superscript style is also written a second time as ``NAME.flat.docx``
with every raised character set as plain text — what a converted PDF gives:
"patients1." with nothing to say the 1 was raised. That variant is where the
model earns its place.

Run (citeproc-py is needed only here, not by the package)::

    uv run --with citeproc-py python benchmarks/references/build_corpus.py OUT_DIR

Downloads are cached in OUT_DIR/cache, so a rerun is offline.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

STYLES = [
    "apa",
    "harvard-cite-them-right",
    "chicago-author-date",
    "elsevier-harvard",
    "springer-basic-author-date",
    "nature",
    "cell",
    "ieee",
    "american-medical-association",
    "nucleic-acids-research",
    "plos",
    "bmj",
    "modern-language-association",
]
DOIS = [
    "10.1093/nar/gky1016",
    "10.1002/humu.24309",
    "10.1038/s41586-020-2434-2",
    "10.1038/nature14539",
    "10.1038/s41586-021-03819-2",
    "10.1093/bioinformatics/btp324",
    "10.1038/nmeth.1923",
    "10.1186/s13059-014-0550-8",
    "10.1038/nbt.1754",
    "10.1101/gr.107524.110",
]
MISSING = 4  # cited, but left out of the list (0-based index into DOIS)
UNCITED = 9  # in the list, never cited
WRONG_YEAR = 2  # author-year styles: this citation's year is off by one
SENTENCES = [
    "Rare disease diagnosis still fails for most patients",
    "Population databases shape how variants are read",
    "Genome sequencing in a national health system raised the yield",
    "Deep learning has changed many fields",
    "Protein structure can now be predicted",
    "Short reads are aligned with fast index methods",
    "Gapped alignment scales to large genomes",
    "Count data need their own statistical models",
    "Visual inspection of alignments remains useful",
]


def fetch(url: str, path: Path, *, accept: str | None = None) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    path.write_bytes(data)
    time.sleep(0.3)
    return data


def load_items(cache: Path) -> list[dict]:
    items = []
    for index, doi in enumerate(DOIS):
        raw = fetch(
            f"https://api.crossref.org/works/{doi}/transform",
            cache / f"item-{index}.json",
            accept="application/vnd.citationstyles.csl+json",
        )
        item = json.loads(raw)
        for key in ("reference", "assertion", "license", "link", "relation"):
            item.pop(key, None)
        item["id"] = f"r{index}"
        items.append(item)
    return items


def flat(rendered) -> str:
    if isinstance(rendered, str):
        return rendered
    return "".join(flat(part) for part in rendered)


def render(style_path: Path, items: list[dict]) -> tuple[list[str], dict[str, str], str]:
    from citeproc import Citation, CitationItem, CitationStylesBibliography, CitationStylesStyle, formatter
    from citeproc.source.json import CiteProcJSON

    style = CitationStylesStyle(str(style_path), validate=False)
    bibliography = CitationStylesBibliography(style, CiteProcJSON(items), formatter.plain)
    cited = [i for i in range(len(items)) if i != UNCITED]
    citations = [Citation([CitationItem(f"r{i}")]) for i in cited]
    for citation in citations:
        bibliography.register(citation)
    # Register the uncited work too, so it is in the list.
    bibliography.register(Citation([CitationItem(f"r{UNCITED}")]))
    shown = {f"r{i}": flat(bibliography.cite(c, lambda _: None)).strip() for i, c in zip(cited, citations)}
    entries = [flat(entry).strip() for entry in bibliography.bibliography()]
    ids = [str(key) for key in bibliography.keys]
    fmt = "numeric"
    text = style_path.read_text()
    if 'citation-format="author-date"' in text:
        fmt = "author_year"
    elif 'citation-format="author"' in text:
        fmt = "author_only"
    superscript = 'vertical-align="sup"' in text.split("<citation", 1)[-1].split("</citation>", 1)[0]
    return ids, shown | {"__entries__": json.dumps(entries)}, fmt + ("_superscript" if superscript else "")


def build(out: Path) -> None:
    from docx import Document

    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    items = load_items(cache)
    manifest = []
    for name in STYLES:
        for flat_text in (False, True):
            case = _build_one(name, flat_text, cache=cache, items=items, out=out, document_cls=Document)
            if case is not None:
                manifest.append(case)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _build_one(name, flat_text, *, cache, items, out, document_cls):
    Document = document_cls
    if True:
        style_path = cache / f"{name}.csl"
        try:
            fetch(
                f"https://raw.githubusercontent.com/citation-style-language/styles/master/{name}.csl",
                style_path,
            )
            ids, shown, kind = render(style_path, items)
        except Exception as exc:  # a style citeproc-py cannot render
            print(f"skip {name}: {type(exc).__name__}: {exc}")
            return None
        entries = json.loads(shown.pop("__entries__"))
        superscript = kind.endswith("_superscript")
        if flat_text and not superscript:
            return None
        fmt = kind.removesuffix("_superscript")
        numbered_list = fmt == "numeric"

        doc = Document()
        paragraphs: list[str] = []  # the projection, paragraph by paragraph
        truth_citations: list[tuple[int, int]] = []
        planted: dict[str, object] = {}

        def para(parts: list[tuple[str, bool]], *, heading: bool = False) -> None:
            if heading:
                p = doc.add_heading("", level=1)
            else:
                p = doc.add_paragraph()
            for text, raised in parts:
                run = p.add_run(text)
                run.font.superscript = raised and not flat_text
            paragraphs.append("".join(t for t, _ in parts))

        def offset() -> int:
            return sum(len(p) + 1 for p in paragraphs)

        para([("A test of citation reading", False)], heading=True)
        para([("Ada Lovelace", False), ("1", True), (", Alan Turing", False), ("2", True)])
        para([("Introduction", False)], heading=True)
        for position, index in enumerate(i for i in range(len(items)) if i != UNCITED):
            key = f"r{index}"
            cite = shown[key]
            if fmt == "author_year" and index == WRONG_YEAR:
                year = str(items[index]["issued"]["date-parts"][0][0])
                cite = cite.replace(year, str(int(year) - 1))
                planted["year_mismatch_citation"] = cite
            sentence = SENTENCES[position]
            if superscript:
                start = offset() + len(sentence)
                para([(sentence, False), (cite, True), (".", False)])
            else:
                start = offset() + len(sentence) + 1
                para([(f"{sentence} {cite}.", False)])
            truth_citations.append((start, start + len(cite)))
            if index == MISSING:
                planted["missing_citation"] = [start, start + len(cite)]
        para([("We grew 10", False), ("5", True), (" cells on 3 m", False), ("2", True), (" plates (n = 12), as in Fig. 2.", False)])
        # Harder: gene, exon and day labels that look like glued citations in
        # a text that lost its formatting, and a repeated citation, which the
        # citation-order rule cannot settle.
        para([("Reads on chr11 and exon2 were counted at day3.", False)])
        if superscript:
            sentence = "This agrees with earlier work"
            start = offset() + len(sentence)
            para([(sentence, False), ("2", True), (".", False)])
            truth_citations.append((start, start + 1))
        para([("(1) prepare the samples.", False)])
        para([("(2) sequence them.", False)])
        para([("References", False)], heading=True)
        list_entries = []
        for key, entry in zip(ids, entries):
            if key == f"r{MISSING}":
                continue
            list_entries.append((key, entry))
        for number, (key, entry) in enumerate(list_entries, start=1):
            if numbered_list and not re.match(r"\[?\d{1,3}[\].)]?", entry):
                entry = f"{number}. {entry}"
            if key == f"r{UNCITED}":
                planted["uncited_entry_start"] = offset()
            para([(entry, False)])

        path = out / f"{name}{'.flat' if flat_text else ''}.docx"
        doc.save(str(path))
        case = (
            {
                "style": name + (".flat" if flat_text else ""),
                "file": path.name,
                "format": fmt,
                "superscript": superscript,
                "expected_style": (
                    "author_year" if fmt == "author_year"
                    else "numeric_superscript" if superscript
                    else "numeric_brackets" if "[" in shown["r0"]
                    else "numeric_parentheses" if "(" in shown["r0"]
                    else "unsupported"
                ) if fmt != "author_only" else "unsupported",
                "text": "\n".join(paragraphs),
                "citations": truth_citations,
                "planted": planted,
                "sample_citation": shown["r0"],
                "sample_entry": list_entries[0][1][:120],
            }
        )
        print(f"built {case['style']}: {case['expected_style']}  e.g. {shown['r0']!r}")
        return case


if __name__ == "__main__":
    build(Path(sys.argv[1]))
