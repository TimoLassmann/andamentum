# data/

Local benchmark data. Everything in this folder except this file is ignored by git
(`data/*` in the repository `.gitignore`) and must never be committed, published or
attached to a release: the datasets belong to their authors, and several carry no
licence that allows redistribution. The folder holds data only for local evaluation.

Each dataset has its own folder, `data/benchmarks/<dataset>/`, with a `PROVENANCE.md`
(source URL, pinned revision, download date, files with sizes and sha256, the licence as
declared with the page it was read from, the citation the authors ask for, and the usage
rule). Derived files (indexes, extracts) are listed there as derived. Results computed from
a dataset go in a folder beside it or under the experiment's output directory, also
ignored; reports may quote counts and scores, never records.

Fetch a dataset with the command named below; each command writes its `PROVENANCE.md`.

## benchmarks/litsearch

- What: the 597 queries of LitSearch and the `corpus_clean` configuration (titles,
  abstracts and citation ids of 64,183 papers; the full-text column and the `corpus_s2orc`
  configuration are not downloaded), with a BM25 and citation index built from it.
- Source: https://huggingface.co/datasets/princeton-nlp/LitSearch, pinned revision
  `9573fb284a1026c998df47024b888a163f0f0e25` (see `PROVENANCE.md` there for the hash of
  every file).
- Licence status: the Hugging Face dataset declares no licence; the code repository
  (https://github.com/princeton-nlp/LitSearch) is MIT; the corpus derives from S2ORC, which
  its official page (https://github.com/allenai/s2orc) states is released under ODC-By 1.0.
  The papers keep their authors' and publishers' rights. Treated as: local evaluation only,
  do not redistribute.
- Citation: Ajith, Xia, Chevalier, Goyal, Chen, Gao. LitSearch: A Retrieval Benchmark for
  Scientific Literature Search. Empirical Methods in Natural Language Processing (EMNLP),
  2024. The S2ORC authors ask for Lo, Wang, Neumann, Kinney, Weld. S2ORC: The Semantic
  Scholar Open Research Corpus. ACL 2020, https://doi.org/10.18653/v1/2020.acl-main.447.
- Fetch: `uv run python -m andamentum.deep_research.v2.bench download litsearch-local
  --dir data/benchmarks/litsearch` (needs the `bench-data` extra: pyarrow).
- Used by: `deep_research` v2, the `litsearch-local` set of the bench and the
  `local_corpus` source.
