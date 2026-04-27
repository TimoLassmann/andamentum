# Multiple Sequence Alignment Methods

## Introduction

Multiple sequence alignment is a foundational task in bioinformatics. Several methods exist including Clustal Omega, MUSCLE, and MAFFT. Despite decades of work, alignment quality on highly divergent sequences remains an open problem.

## Methods

We benchmarked three popular tools against the BAliBASE 4.0 reference set. All experiments were run on a 16-core AMD EPYC machine with 64 GB RAM. We report sum-of-pairs scores averaged across reference sets.

## Results

Kalign achieved the highest sum-of-pairs score on three of the four BAliBASE reference sets. The largest improvement was on Reference 4, the divergent-sequence set.

## Discussion

Kalign's advantage is most pronounced on the hardest reference set. The runtime gap reaches an order of magnitude at large input sizes, suggesting Kalign is the right choice for proteome-scale alignments.
