# Reference checker: style test set

Measures how well `andamentum.references` reads citations across real
citation styles. No Crossref lookup — this is about reading.

1. Build the documents (downloads CSL styles and Crossref metadata once, into
   `OUT/cache`; `citeproc-py` is needed only here):

       uv run --with citeproc-py python benchmarks/references/build_corpus.py OUT

2. Score them, from evidence alone or with a model for the open cases:

       uv run python benchmarks/references/evaluate.py OUT
       uv run python benchmarks/references/evaluate.py OUT --model ollama:gemma4:26b-nvfp4

Each document has the same planted errors (a cited work missing from the
list, a listed work never cited, a wrong year in author-year styles) and the
same distractors (affiliation marks, `10⁵`, `m²`, `Fig. 2`, `(n = 12)`, a
numbered step list, and in the superscript styles `chr11`, `exon2`, `day3`
and a repeated citation). Superscript styles are also written as `*.flat.docx`
with the formatting stripped — what a converted PDF gives.

Columns: whether the style was read correctly and how (evidence or model);
recall and precision of citations against the known spans; planted errors
found; other problems reported (false alarms); candidates judged by the model;
candidates left unjudged.
