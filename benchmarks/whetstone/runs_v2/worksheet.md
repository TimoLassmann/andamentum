## arxiv_1406.2661_v1

For each issue, mark whether you agree it is real, and your own severity/locality. Systems are anonymised.

### 1. The practical claim that alternating k-step optimization keeps the discriminator near optimum is not substantiated in the manuscript excerpt.
- raised by: **System 2 only**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

### 2. Related evaluation details are incomplete or unclear, including baselines/comparisons and the specific benchmark coverage used to assess competitiveness.
- raised by: **both systems**
- judge tags: minor, local
- your verdict (real? severity? locality?): ____

### 3. The relationship between the theoretical result and the actual parameterized training procedure is unclear: the proof/convergence statement is about distributions under idealized updates, while the algorithm optimizes network parameters with no bridge from nonconvex parameter space to distribution-level convergence.
- raised by: **both systems**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

## arxiv_1412.6980_v1

For each issue, mark whether you agree it is real, and your own severity/locality. Systems are anonymised.

### 1. Experimental robustness claims about stochasticity/noise or dropout are not sufficiently specified, including how noise is applied and controlled across training and evaluation.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 2. The paper makes broad motivation/contribution claims without a clear mapping to specific assumptions or evidence, so the story overreaches the presented support.
- raised by: **both systems**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

### 3. Claims of optimizer superiority / faster convergence are unsupported by quantitative metrics, statistical variability, or clearly described figures in the experiment sections.
- raised by: **both systems**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 4. The convergence discussion for non-convex settings is unclear because the manuscript itself says the analysis does not apply there, yet the text still uses convergence language in those contexts.
- raised by: **System 1 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

## arxiv_1512.03385_v1

For each issue, mark whether you agree it is real, and your own severity/locality. Systems are anonymised.

### 1. Novelty framing relative to prior shortcut/residual work is incomplete: the reviews note that the manuscript does not clearly distinguish its contribution from earlier skip-connection or residual-like ideas.
- raised by: **both systems**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

### 2. Section-level terminology and labels are sometimes inconsistent or undefined in the excerpt (for example, option labels, metric qualifiers, and residual/shortcut terminology), which can confuse the reader.
- raised by: **both systems**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

### 3. The paper’s discussion of residual learning as a generic principle is broader than the evidence shown; claims about applicability beyond the reported vision settings are not supported by direct experiments in the excerpt.
- raised by: **both systems**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

## arxiv_1706.03762_v1

For each issue, mark whether you agree it is real, and your own severity/locality. Systems are anonymised.

### 1. Cost methodology for FLOPs/time comparisons is under-specified, including hardware and throughput assumptions.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 2. Evaluation baselines and preprocessing/protocol details are incomplete, limiting reproducibility of BLEU comparisons.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 3. The constituency parsing generalization claim is under-supported by the comparison framework and missing baselines.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 4. The constituency parsing evaluation lacks a rigorous metric/baseline specification.
- raised by: **System 2 only**
- judge tags: critical, local
- your verdict (real? severity? locality?): ____

### 5. Novelty/significance claims rely on insufficiently exhaustive prior comparisons.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 6. The efficiency/parallelization story is over-optimistic relative to the evidence shown in the excerpt.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 7. Training/decoding conditions for comparing against prior models are incomplete or not matched clearly.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 8. The BLEU evaluation protocol is not clearly defined enough to reproduce the reported scores.
- raised by: **System 2 only**
- judge tags: critical, local
- your verdict (real? severity? locality?): ____

### 9. The parsing data regimes and semi-supervised settings are unclear, undermining comparability.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 10. The causal link from architectural efficiency properties to empirical performance/training-time claims is not fully established.
- raised by: **System 2 only**
- judge tags: critical, cross_section
- your verdict (real? severity? locality?): ____

### 11. Claims about linguistic functions learned by attention heads are qualitative and not empirically validated in the excerpt.
- raised by: **System 1 only**
- judge tags: minor, local
- your verdict (real? severity? locality?): ____

### 12. The decoder residual/add-norm formula is missing in the provided text.
- raised by: **System 1 only**
- judge tags: minor, local
- your verdict (real? severity? locality?): ____

### 13. The text overgeneralizes from a few visualized heads to broad conclusions about learning.
- raised by: **System 1 only**
- judge tags: minor, local
- your verdict (real? severity? locality?): ____

## arxiv_1810.04805_v1

For each issue, mark whether you agree it is real, and your own severity/locality. Systems are anonymised.

### 1. The learning-rate value for SQuAD is almost certainly mistyped as 5e5 instead of 5e-5.
- raised by: **System 2 only**
- judge tags: critical, local
- your verdict (real? severity? locality?): ____

### 2. The statement that all experiments mask 15% of WordPiece tokens is not supported by the excerpted text.
- raised by: **System 1 only**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

### 3. The narrative about the paper’s core contribution is muddled because it mixes architecture, objective, input representation, training procedure, and results without a tight mapping to specific claims.
- raised by: **System 2 only**
- judge tags: minor, cross_section
- your verdict (real? severity? locality?): ____

### 4. The supervised-transfer framing is too broad for the evidence shown in the excerpt, which gives only high-level examples.
- raised by: **System 1 only**
- judge tags: minor, local
- your verdict (real? severity? locality?): ____
