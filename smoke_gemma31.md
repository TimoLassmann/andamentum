<!-- andamentum-whetstone
     produced-by: andamentum-whetstone v0.2.0 (model=ollama:gemma4:31b-nvfp4) on 2026-05-24
     ai-generated: true
-->

---

# Whetstone Review

---

> **⚠ AI-generated review content.** This report was generated for your own drafts. Whetstone is not a peer-review tool — do not use it on manuscripts other authors have sent you confidentially.
> 
> *Produced by andamentum-whetstone v0.2.0 (model=ollama:gemma4:31b-nvfp4) on 2026-05-24.*

---

## Summary

This document introduces Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions. The method is presented as computationally efficient, memory-light, and empirically effective across a variety of large-scale machine learning problems.

## Strengths

- Proposes a computationally efficient first-order optimizer with initialization bias correction.
- Provides empirical validation across diverse architectures, including convolutional neural networks (CNNs) and multi-layer neural networks.
- Combines advantages of AdaGrad (sparse gradients) and RMSProp (non-stationary objectives).

## Weaknesses

- Theoretical convergence guarantees are limited to convex objectives and do not formally extend to the non-convex problems common in deep learning.
- While other adaptive methods like AdaDelta and vSGD are mentioned as related work, they are not extensively benchmarked against Adam compared to methods like AdaGrad or SGD with momentum.

---

## Findings (LLM-investigated) (9)

### MUST FIX (3)

- **Correctness** _(_major_, medium confidence)_
  The theoretical convergence guarantees and regret bounds are overstated because they rely on a decaying stepsize (γt = 1/t) as specified in Theorem 4.1, whereas the actual algorithm is implemented with a constant stepsize (α).
  · sections: s7
  > [s7] With γ t = 1 /t , Adam achieves the following guarantee, for all T ≥ 1 .

- **Presentation** _(_major_, medium confidence)_
  Terminology inconsistency regarding 'decay rate' ($\beta$). The authors define $\beta$ as the 'exponential decay rates' in Sections 2 and 3, but use it as the coefficient for the current gradient ($g_t$) rather than the history ($m_{t-1}$) in their update equations. In standard EMA/Adam notation, the decay rate is the weight of the previous estimate. This leads to non-standard hyperparameter values (e.g., $\beta=0.1$ instead of $0.9$), which may confuse readers familiar with the field.
  · sections: s4
  > [s4] The algorithm updates exponential moving averages of the gradient ( m t ) and the squared gradient ( v t ) where the hyper-parameters β 1 ∈ 

- **Presentation** _(_major_, medium confidence)_
  Logically contradictory guidance on hyperparameters. In Section 3, the authors state that $\beta$ should be chosen to assign 'small weights to gradients too far in the past'. However, based on their update rule $v_t = \beta g^2 + (1-\beta) v_{t-1}$, this would require a large $\beta$. In contrast, their provided default values ($\beta_1=0.1, \beta_2=0.001$) result in very long memory (weights of $0.9$ and $0.999$ for history), directly contradicting the stated design goal.
  · sections: s6
  > [s6] the exponential decay rate β 1 can (and should) be chosen such that the exponential moving average assigns small weights to gradients too fa

### SHOULD FIX (6)

- **Correctness** _(_moderate_, medium confidence)_
  The paper contains a mathematical error regarding the Signal-to-Noise Ratio (SNR). It claims that m̂t / √v̂t ≈ ±1 because 'E[g] / E[g²] = ±1'. This is incorrect; the ratio of the first moment to the second raw moment is not ±1 unless the gradient is constant and equals ±1. The authors likely meant E [ g ] / √ ( E [ g 2 ] ) ≤ 1, but as written, the statement is false.
  · sections: s5
  > [s5] we will have that ̂ m t / √ ̂ v t ≈ ± 1 since E [ g ] / E [ g 2 ] = ± 1 .

- **Correctness** _(_moderate_, medium confidence)_
  The claimed upper bound for the effective stepsize (α · β1/β2) is mathematically incorrect. Because Adam uses bias correction to ensure m̂t and v̂t are unbiased estimates of the first and second moments respectively, the ratio m̂t / √v̂t converges to E[g] / √(E[g²]), which is bounded by 1 regardless of the values of β1 and β2. The suggested bound implies that decreasing β2 could inflate the stepsize indefinitely, which bias correction specifically prevents.
  · sections: s5
  > [s5] The effective stepsize has strong upper and lower bounds: -α · β 1 /β 2 ≤ ∆ t ≤ + α · β 1 /β 2 .

- **Presentation** _(_moderate_, medium confidence)_
  The 'efficiency' note in Section 2 is hard to follow and poorly integrated. It introduces a dense one-line formula that differs significantly in structure from the step-by-step pseudo-code of Algorithm 1, making it difficult for a reader to verify or implement without algebraically deriving the bias correction terms.
  · sections: s4
  > [s4] Note that the efficiency of algorithm 1 can, at the expense of clarity, be improved upon by changing the order of computation, e.g. by repla

- **Significance** _(_moderate_, medium confidence)_
  There is a gap in the empirical evaluation regarding related work. While the authors identify AdaDelta and vSGD as stochastic optimization methods that estimate curvature from first-order information (similar to Adam) in Section 5, they are completely omitted from the experimental benchmarks in Section 6. This undermines the claim that Adam's performance is demonstrably superior or comparable to "best known" adaptive methods.
  · sections: s8
  > [s8] Other stochastic optimization methods include vSGD Schaul et al. (2012) and AdaDelta Zeiler (2012), both setting stepsizes by estimating cur

- **Story** _(_moderate_, medium confidence)_
  The paper claims that Adam's hyperparameters are intuitive and require little tuning, specifically asserting that it is "relatively easy to know the right scale of α in advance". However, there is no evidence provided in the experiments or analysis to support this claim relative to other optimizers; the authors simply report a single set of default values used for their tests.
  · sections: s5
  > [s5] This typically makes it relatively easy to know the right scale of α in advance.

- **Story** _(_moderate_, medium confidence)_
  The paper claims that Adam performs "a form of automatic annealing" based on the Signal-to-Noise Ratio (SNR) becoming closer to zero near an optimum. While this is explained conceptually in Section 2.1, it is not empirically demonstrated or validated in the experiments section to show that this behavior actually occurs during training.
  · sections: s5
  > [s5] For example, the SNR value typically becomes closer to 0 towards an optimum, leading to smaller effective steps in parameter space: a form o

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
