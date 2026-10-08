# Reference checker: style test set

Measures how well `andamentum.references` reads citations across real
citation styles. No Crossref lookup — this is about reading.

1. Build the documents (downloads CSL styles and Crossref metadata once, into
   `OUT/cache`; `citeproc-py` is needed only here):

       uv run --with citeproc-py python benchmarks/references/build_corpus.py OUT

2. Score them, from evidence alone or with a model for the open cases:

       uv run python benchmarks/references/evaluate.py OUT
       uv run python benchmarks/references/evaluate.py OUT --model ollama:gemma4:26b

   `--pdf` renders each document to PDF with LibreOffice and reads it the way
   a PDF submission is read (harvest with `preserve_superscripts`, then
   whetstone's baseline Word file). Citation spans do not carry over through
   a conversion, so it scores the style read and the planted problems by
   kind. Needs `soffice` (`--soffice PATH`) and loads docling.

Each document has the same planted errors (a cited work missing from the
list, a listed work never cited, a wrong year in author-year styles) and the
same distractors (affiliation marks, `10⁵`, `m²`, `Fig. 2`, `(n = 12)`, a
numbered step list, and in the superscript styles `chr11`, `exon2`, `day3`
and a repeated citation). Superscript styles are also written as `*.flat.docx`
with the formatting stripped — what a converted PDF gives.

Two notes documents use Chicago notes-and-bibliography with real Word
footnotes (`chicago-notes.docx`, and `chicago-notes.nobib.docx` without a
bibliography): full notes, short forms, `Ibid.` after a single source and
after a note citing two (planted), a note citing a work missing from the
bibliography (planted), a short form before its full note (planted without a
bibliography), commentary, and two notes the rules cannot read — an interview
(not a source) and a report by an organisation (a source).

Columns: whether the style was read correctly and how (evidence or model);
recall and precision of citations against the known spans; planted errors
found; other problems reported (false alarms); candidates judged by the model;
candidates left unjudged.
