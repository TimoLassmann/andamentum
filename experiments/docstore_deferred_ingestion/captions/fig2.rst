**Retrieval recall@1 by signal, before and after the drain**, with the chance
level plotted as its own bar.

This is the behavioural proof of claim (e): the drain is what turns a
registered-but-unenriched document into a semantically retrievable one.
Chunk-embedding recall@1 moves from 0.0 to 1.0 across the drain, and the
unified four-signal RRF stack moves 0.25 to 1.0.

Read against the chance bar, not against 1.0. The probe set discriminates
between five candidate documents drawn from four topically disjoint famous
papers with no hard negatives, so chance recall@1 is 0.2. This demonstrates the
index is neither empty nor scrambled; it does not measure retrieval quality,
and ``claim_e_post.retrieval_power.limitation`` says so on the artefact's face.
