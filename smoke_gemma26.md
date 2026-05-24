<!-- andamentum-whetstone
     produced-by: andamentum-whetstone v0.2.0 (model=ollama:gemma4:26b-nvfp4) on 2026-05-24
     ai-generated: true
-->

---

# Whetstone Review

---

> **⚠ AI-generated review content.** This report was generated for your own drafts. Whetstone is not a peer-review tool — do not use it on manuscripts other authors have sent you confidentially.
> 
> *Produced by andamentum-whetstone v0.2.0 (model=ollama:gemma4:26b-nvfp4) on 2026-05-24.*

---

## Summary

This document introduces Adam, a stochastic optimization algorithm that utilizes initialization bias correction. The paper provides a theoretical convergence analysis alongside experimental evaluations on various models, including neural networks and CNNs.

## Strengths

- Introduces a new method for first-order gradient-based optimization using bias correction.
- Includes empirical evaluations across various machine learning architectures like MLPs and CNNs.

## Weaknesses

- Omits specific values for primary hyperparameters (e.g., $\alpha, \beta_1, \beta_2$), which may impede the reproduction of experimental results.
- Mathematical notation for update rules contains unrendered placeholders (e.g., 'glyph').
- Claims regarding stepsize bounds and regret ($O(T)$) are presented as potentially inconsistent or trivial.
- Inconsistent notation for decay rates provided ($\beta$ vs. $\beta_1, \beta_2$).

---

## Findings (LLM-investigated) (8)

### MUST FIX (2)

- **Evaluations** _(_major_, medium confidence)_
  The document fails to specify the primary hyperparameter settings—specifically the learning rate ($\alpha$) and the decay rates for the first ($\beta_1$) and second ($\\beta_2$) moments—across the main experimental sections (6.1, 6.2, and 6.3). This omission makes it impossible to reproduce the convergence curves presented in Figures 1, 2, and 3.
  · sections: s10
  > [s10] We compare Adam to accelerated SGD with Nesterov momentum and AdaGrad using mini-batch size of 128.

- **Presentation** _(_major_, medium confidence)_
  Mathematical notation for the update rule in Section 2 and the effective stepsize bounds in Section 2.1 is broken or illegible.
  · sections: s4
  > [s4] θ t ← θ t -1 -( α · √ 1 -(1 -β 2 ) t · (1 -(1 -β 1 ) t ) -1 ) · m t / √ v t .

### SHOULD FIX (5)

- **Correctness** _(_moderate_, medium confidence)_
  The claimed bounds for the effective stepsize ($\pm \alpha\beta_1/\beta_2$) are mathematically inconsistent with the behavior of the bias-corrected estimates in extreme sparsity cases. For a single non-zero gradient at $t=1$, the update magnitude is $\alpha$, which exceeds the provided upper bound since typically $\beta_1 < \beta_2$.
  · sections: s5
  > [s5] The effective stepsize has strong upper and lower bounds: -α · β 1 /β 2 ≤ ∆ t ≤ + α · β 1 /β 2 .

- **Correctness** _(_moderate_, medium confidence)_
  The paper states a regret bound of $O(T)$, which is an extremely loose and trivial statement for online learning that does not imply convergence. This contradicts the much tighter $O(\log d \sqrt{T})$ bound discussed later in the same section, suggesting either a notation error or an incomplete primary claim.
  · sections: s7
  > [s7] We give a convergence proof and a regret O ( T ) for the online convex function using the Adam algorithm.

- **Evaluations** _(_moderate_, medium confidence)_
  Baseline descriptions are incomplete; specifically, the manual learning rate strategy for SGD in Section 6.3 and the identities/parameters of "other stochastic first order methods" in Section 6.2 are not provided.
  · sections: s12
  > [s12] A smaller learning rate for the convolution layers is often used in practice when applying SGD.

- **Presentation** _(_moderate_, medium confidence)_
  Placeholder/Broken symbols: The text contains unrendered mathematical placeholders instead of actual symbols (e.g., 'glyph[lessorapproxeql]'), which degrades the readability and professional quality of the presentation.
  · sections: s5
  > [s5] -α glyph[lessorapproxeql] ∆ t glyph[lessorapproxeql] + α .

- **Significance** _(_moderate_, medium confidence)_
  The experimental validation of Adam is incomplete regarding its intended design goals. While the paper claims Adam combines the advantages of AdaGrad (for sparse gradients) and RMSProp (for non-stationary objectives), the primary performance comparisons in Sections 6.1, 6.2, and 6.3 are conducted against SGD and AdaGrad/SFO, omitting a direct comparison with RMSProp for the main models (MLPs, CNNs). Additionally, because all experimental datasets used (MNIST, CIFAR-10) are stationary, there is no empirical evidence provided to support the specific claim that Adam effectively handles non-stationary objectives.
  · sections: s2
  > [s2] The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.

### CONSIDER (1)

- **Presentation** _(_minor_, medium confidence)_
  Inconsistent hyperparameter notation: The document refers to hyperparameters β1 and β2 individually, but occasionally uses a single β to refer to them collectively, which is slightly inconsistent.
  · sections: s4
  > [s4] especially when the decay rates β are small.

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
