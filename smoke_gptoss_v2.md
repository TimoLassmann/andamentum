<!-- andamentum-whetstone
     produced-by: andamentum-whetstone v0.2.0 (model=ollama:gpt-oss:20b) on 2026-05-24
     ai-generated: true
-->

---

# Whetstone Review

---

> **⚠ AI-generated review content.** This report was generated for your own drafts. Whetstone is not a peer-review tool — do not use it on manuscripts other authors have sent you confidentially.
> 
> *Produced by andamentum-whetstone v0.2.0 (model=ollama:gpt-oss:20b) on 2026-05-24.*

---

## Summary

The Adam paper introduces an adaptive optimisation algorithm that estimates first and second moments of the gradients to set per‑parameter learning rates.  The authors present clear pseudo‑code (Algorithm 1), discuss bias‑correction for the moving‑average estimates, and provide a theoretical convergence analysis under an online convex‑learning framework.  A suite of experiments – on L2‑regularised multi‑class logistic regression, fully‑connected neural networks, and convolutional neural networks – demonstrates practical performance gains relative to SGD with momentum and AdaGrad.  The work contextualises Adam among related methods such as RMSProp, AdaGrad, and quasi‑Newton approaches.

## Strengths

- The paper offers a concise pseudo‑code of Adam, enabling straightforward implementation.
- Bias‑correction terms are explicitly motivated and empirically evaluated.
- A broad experimental evaluation (logistic regression, multi‑layer neural nets, CNNs) showcases Adam’s practical effectiveness.
- The related‑work discussion situates Adam among prominent first‑order optimisation algorithms.

## Weaknesses

- Section 7 claims a regret bound of O(T) for Adam; the analysis actually yields sub‑quadratic bounds (O(log d √T)), making the statement incorrect.
- The abstract contains a grammatical error – "based an adaptive estimates" – that obscures meaning.
- Subsection heading "2.1 ADAM'S UPDATE RULE" mislabels the content, which actually discusses step‑size bounds rather than the update rule.
- The paper overstates invariance to diagonal rescaling; the theoretical justification is not fully established.
- Experimental comparisons are limited to SGD with momentum and AdaGrad, leaving out systematic evaluation against RMSProp and other state‑of‑the‑art optimisers.
- Related‑work coverage is incomplete; notably, AdaDelta is not discussed.

---

## Findings (LLM-investigated) (7)

### MUST FIX (1)

- **Correctness** _(_major_, medium confidence)_
  Section 7 states a regret bound of O(T) for the Adam algorithm:
- "We give a convergence proof and a regret O ( T ) …."\n
The theoretical analysis of Adam, as cited later in the paper, shows an O(log d sqrt(T)) bound (or O(sqrt(d) T) for AdaGrad), not linear in T. Therefore the stated bound is incorrect.
  · sections: s7
  > [s7] We give a convergence proof and a regret O ( T ) for the online convex function using the Adam algorithm.

### SHOULD FIX (4)

- **Presentation** _(_moderate_, medium confidence)_
  The abstract contains a grammatical error that obscures meaning: the phrase "The method is straightforward to implement and is based an adaptive estimates of lower-order moments of the gradients." incorrectly uses "based an" and plural.
  · sections: s2
  > [s2] The method is straightforward to implement and is based an adaptive estimates of lower-order moments of the gradients.

- **Presentation** _(_moderate_, medium confidence)_
  The subsection heading "2.1 ADAM'S UPDATE RULE" mislabels the content, which actually discusses effective step‑size bounds rather than the update rule itself, potentially confusing readers.
  · sections: s5
  > [s5] The effective stepsize has strong upper and lower bounds: -α · β 1 /β 2 ≤ ∆ t ≤ + α · β 1 /β 2 .

- **Significance** _(_moderate_, medium confidence)_
  Overstated claim of invariance to diagonal rescaling
  · sections: s2
  > [s2] The method exhibits invariance to diagonal rescaling of the gradients by adapting to the geometry of the objective function.

- **Story** _(_moderate_, medium confidence)_
  The core claim that Adam works well “in practice when experimentally compared to other stochastic optimization methods” is not fully supported; the experiments only compare against SGD with momentum and AdaGrad, missing a systematic comparison with RMSProp or other state‑of‑the‑art variants.
  · sections: s2
  > [s2] We demonstrate that Adam works well in practice when experimentally compared to other stochastic optimization methods.

### CONSIDER (2)

- **Presentation** _(_minor_, medium confidence)_
  Section 2 contains a very long, complex sentence that hinders clarity: the description of the moving‑average updates and hyper‑parameters is packed together.
  · sections: s4
  > [s4] The algorithm updates exponential moving averages of the gradient ( m t ) and the squared gradient ( v t ) where the hyper-parameters β 1 ∈ 

- **Significance** _(_minor_, medium confidence)_
  Incomplete related‑work coverage (AdaDelta omitted)
  · sections: s8
  > [s8] AdaGrad: An algorithm that works well for sparse gradients is AdaGrad Duchi et al. (2011).

---

## Document map (16 sections)

- **s1** ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION — Diederik P.
- **s2** ABSTRACT — We introduce Adam , an algorithm for first-order gradient-based optimization of stochastic objective functions.
- **s3** 1 INTRODUCTION — Stochastic gradient-based optimization is of core practical importance in many fields of science and engineering.
- **s4** 2 ALGORITHM — See algorithm 1 for pseudo-code of our proposed algorithm Adam .
- **s5** 2.1 ADAM'S UPDATE RULE — An important property of Adam's update rule is its careful choice of stepsizes.
- **s6** 3 INITIALIZATION BIAS CORRECTION — As explained in section 2, Adam utilizes initialization bias correction terms.
- **s7** 4 CONVERGENCE ANALYSIS — We analyze the convergence of Adam under the online learning framework proposed in Zinkevich (2003).
- **s8** 5 RELATED WORK — Optimization methods bearing a direct relation to Adam include RProp Riedmiller &amp; Braun (1992), RMSProp Tieleman &amp; Hinton (2012); Graves et al.
- **s9** 6 EXPERIMENTS — To empirically evaluate the proposed method, we investigated different popular machine learning models, including logistic regression, multilayer fully connect…
- **s10** 6.1 EXPERIMENT: LOGISTIC REGRESSION — Weevaluate our proposed method on L2-regularized multi-class logistic regression using the MNIST dataset.
- **s11** 6.2 EXPERIMENT: MULTI-LAYER NEURAL NETWORKS — Multi-layer neural network are powerful models with non-convex objective functions.
- **s12** 6.3 EXPERIMENT: CONVOLUTIONAL NEURAL NETWORKS — Convolutional neural networks (CNNs) with several layers of convolutional, pooling and non-linear units have shown considerable success in computer vision task…
- **s13** 6.4 EXPERIMENT: BIAS-CORRECTION TERM — We also empirically evaluate the effect of the bias correction terms explained in sections 2 and 3.
- **s14** 7 CONCLUSION — We have introduced a simple and computationally efficient algorithm for gradient-based optimization of stochastic objective functions.
- **s15** 8 ACKNOWLEDGMENTS — This paper would probably not have existed without the support of Google Deepmind, the collaborations it supports and interesting conversations they sparked.
- **s16** REFERENCES — Amari, Shun-Ichi.
