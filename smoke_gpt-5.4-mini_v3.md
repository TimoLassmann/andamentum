<!-- andamentum-whetstone
     produced-by: andamentum-whetstone v0.2.0 (model=openai:gpt-5.4-mini) on 2026-05-24
     ai-generated: true
-->

---

# Whetstone Review

---

> **⚠ AI-generated review content.** This report was generated for your own drafts. Whetstone is not a peer-review tool — do not use it on manuscripts other authors have sent you confidentially.
> 
> *Produced by andamentum-whetstone v0.2.0 (model=openai:gpt-5.4-mini) on 2026-05-24.*

---

## Summary

This paper presents Adam, an adaptive stochastic optimization method, and the document is generally in solid shape but needs revision in several concentrated areas. The largest issues cluster in the theory sections, where the convergence/regret claims are stated too broadly or imprecisely, and in the experiments, where reproducibility and fair comparison details are under-specified. Related-work and story framing also need tightening so the method’s novelty, scope, and relationship to prior optimizers are described more carefully. Presentation issues are present throughout the algorithm and terminology sections, but they are secondary to the correctness and evaluation gaps. Overall, the document is promising and coherent, but the claims should be narrowed to match the assumptions and evidence actually provided.

## Strengths

- Clear high-level motivation for an adaptive first-order optimizer.
- The paper connects Adam to well-known ideas such as AdaGrad and RMSProp.
- The experiments cover multiple model families and datasets.
- The bias-correction and algorithmic design are presented as a distinct method contribution.

## Weaknesses

- The convergence/regret claim is misstated as O(T) when the proof is aiming at sublinear regret and vanishing average regret.
- Experimental reproducibility and fairness are under-specified: data splits, stopping criteria, tuning budget, held-fixed settings, metrics, and baseline tuning/hardware details are missing or unclear.
- The theoretical guarantees are described too broadly; the abstract/introduction should clearly limit them to online convex optimization under boundedness assumptions.
- The update-rule/theorem discussion needs more precise treatment of bias correction, stepsizes, and bounded-iterate/bounded-domain assumptions.
- The paper relies too heavily on figures and qualitative claims; it should report quantitative metrics, variance, and summary statistics in text or tables.
- The comparison protocol for classification tasks is unclear, especially whether results are based on training, validation, or test performance.
- Presentation of the pseudocode and terminology is inconsistent in places, making the algorithm harder to read on first pass.
- The novelty and scope claims should be tempered and the related-work coverage broadened, especially around adaptive-gradient, momentum, and natural-gradient methods.

---

## Findings (LLM-investigated) (23)

### MUST FIX (4)

- **Correctness** _(_major_, medium confidence)_
  Correct the convergence section’s stated regret rate: it should not say O(T) when the proof and corollary are aiming at sublinear regret and vanishing average regret.
  · sections: s7
  > [s7] We give a convergence proof and a regret O ( T ) for the online convex function using the Adam algorithm.

- **Evaluations** _(_major_, medium confidence)_
  The experiments and ablations need clearer reproducibility and comparison protocol details: specify the exact data splits, stopping criteria, tuning budget, held-fixed settings, and evaluation metric so the reported results can be reproduced and fairly interpreted.
  · sections: s9
  > [s9] "The hyper-parameters, such as learning rate and momentum, are searched over a dense grid and the results are reported using the best hyper-

- **Evaluations** _(_major_, medium confidence)_
  State the SFO baseline setup and fairness conditions more explicitly. The paper says it used the authors' implementation, but does not report whether SFO received the same tuning budget, hardware, or stopping rule as Adam, so the wall-clock and iteration comparisons may not be apples-to-apples.
  · sections: s11
  > [s11] "We used their implementation and compared with Adam to train such models."

- **Story** _(_major_, medium confidence)_
  The theoretical guarantee is described too broadly; the abstract/introduction should make clear that the regret bound is for online convex optimization under specific boundedness assumptions, not a general convergence guarantee.
  · sections: s2
  > [s2] We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparabl

### SHOULD FIX (15)

- **Correctness** _(_moderate_, medium confidence)_
  Explain the bias-correction and stepsize effects more carefully in the update-rule discussion, including the conditions under which the stated bounds or practical benefits hold.
  · sections: s5
  > [s5] The effective stepsize has strong upper and lower bounds: -α · β 1 /β 2 ≤ ∆ t ≤ + α · β 1 /β 2 .

- **Correctness** _(_moderate_, medium confidence)_
  Clarify the theorem assumptions so the bounded-iterates/bounded-domain condition is stated precisely and not phrased as if it were a property generated by Adam itself.
  · sections: s7
  > [s7] Assume that the functions f t have bounded gradients, ‖∇ f t ( θ ) ‖ 2 ≤ G , ‖∇ f t ( θ ) ‖ ∞ ≤ G ∞ for all θ ∈ R d and distance between any

- **Evaluations** _(_moderate_, medium confidence)_
  Report quantitative results in the text or tables instead of relying only on figures and qualitative claims. Several experimental claims assert faster convergence or better performance, but the prose does not give the actual metric values, variance, or summary statistics needed to verify how large the gains are.
  · sections: s10
  > [s10] "According to Figure 1, we found that the Adam yields similar convergence as SGD with momentum and both converge faster than AdaGrad."

- **Evaluations** _(_moderate_, medium confidence)_
  Define the experimental metric and comparison protocol more clearly for the classification tasks. The paper talks about training negative log likelihood and convergence, but does not state whether the comparison is based on training loss, validation loss, or test performance, which makes the claimed improvements hard to interpret.
  · sections: s10
  > [s10] "Figure 1: Logistic regression training negative log likelihood on MNIST images and IMDB movie reviews with 10,000 bag-of-words (BoW) featur

- **Presentation** _(_moderate_, medium confidence)_
  Fix the presentation of the pseudocode and surrounding explanation in Section 2 so the update rule is readable on first pass; the current sentence is hard to follow because the reordered one-line update is densely packed and not typeset clearly in the text.
  · sections: s4
  > [s4] Note that the efficiency of algorithm 1 can, at the expense of clarity, be improved upon by changing the order of computation, e.g. by repla

- **Presentation** _(_moderate_, medium confidence)_
  Use consistent terminology for the second moment throughout the paper; the document alternates between "second moment", "second raw moment", and "uncentered variance" in a way that can confuse readers about whether the same quantity is meant each time.
  · sections: s4
  > [s4] The moving averages themselves are estimates of the 1 st moment (the mean) and the 2 nd raw moments (the uncentered variance) of the gradien

- **Significance** _(_moderate_, medium confidence)_
  The paper should temper its implicit novelty claim by acknowledging that Adam is presented as a combination of existing ideas rather than a clearly new optimization family. In particular, the related-work section shows close dependence on AdaGrad, RMSProp, RProp, and NGD, so the contribution needs a sharper comparison to what is actually new versus inherited.
  · sections: s8
  > [s8] Optimization methods bearing a direct relation to Adam include RProp Riedmiller &amp; Braun (1992), RMSProp Tieleman &amp; Hinton (2012); Gr

- **Significance** _(_moderate_, medium confidence)_
  The related-work coverage should be expanded to cite and distinguish additional adaptive-gradient and momentum-based optimizers that readers would expect to see around Adam. As written, the comparison set is narrow enough that the significance of the method relative to contemporaneous alternatives is under-justified.
  · sections: s8
  > [s8] Other stochastic optimization methods include vSGD Schaul et al. (2012) and AdaDelta Zeiler (2012), both setting stepsizes by estimating cur

- **Significance** _(_moderate_, medium confidence)_
  The paper should compare Adam against natural-gradient and diagonal-preconditioner methods more carefully, or explicitly scope the claim if it is not meant to compete with that literature. The current discussion notes a Fisher-information connection but does not explain how Adam differs from or improves on those broader second-order-inspired methods.
  · sections: s8
  > [s8] Like natural gradient descent (NGD) Amari (1998), Adam employs a preconditioner that adapts to the geometry of the data, since ̂ v t is an a

- **Significance** _(_moderate_, medium confidence)_
  The conclusion makes a broader efficiency claim than the introduction/experiments support: “simple and computationally efficient” is stated as a general property, but the section does not justify efficiency beyond the method being straightforward and memory-light. If this is meant as a substantive claim, it should be qualified or tied to concrete cost comparisons.
  · sections: s14
  > [s14] We have introduced a simple and computationally efficient algorithm for gradient-based optimization of stochastic objective functions.

- **Story** _(_moderate_, medium confidence)_
  The paper overstates Adam’s novelty/fit in the introduction and abstract: it should more explicitly name the limitation in prior adaptive methods that motivated the work and narrow the claims about being suitable for large-scale, noisy, sparse, or non-stationary problems to the tested settings or stated assumptions.
  · sections: s3
  > [s3] Our method is designed to combine the advantages of two recently popular methods: AdaGrad Duchi et al. (2011), which works well with sparse 

- **Story** _(_moderate_, medium confidence)_
  The document should qualify the broad claim that Adam is especially appropriate for sparse or noisy gradients, and frame the AdaGrad connection in that same limited, comparative way rather than implying a general, broadly established property.
  · sections: s2
  > [s2] The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.

- **Story** _(_moderate_, medium confidence)_
  Soften the claim of exact gradient-rescaling invariance: the paper should state the conditions more carefully, since the implemented update is only approximately invariant once bias correction and finite ε are taken into account.
  · sections: s2
  > [s2] The method exhibits invariance to diagonal rescaling of the gradients by adapting to the geometry of the objective function.

- **Story** _(_moderate_, medium confidence)_
  Separate empirical performance claims from theoretical or general superiority claims: the paper should limit statements about outperforming other methods or being well-suited to broad classes of non-convex problems to the reported benchmarks and experiments.
  · sections: s3
  > [s3] Empirically, our method consistently outperforms other methods for a variety of models and datasets, as shown in section 6.

- **Story** _(_moderate_, medium confidence)_
  The summary of the sparse-feature consequence is overstated relative to what the theorem actually establishes, because it says "the adaptive method can achieve O (log d √ T )" without clearly tying that scaling to the specific bounded-sparsity argument and the hidden assumptions from the cited AdaGrad result. In this section opener, that makes the analysis conditions sound more general than they are.
  · sections: s7
  > [s7] The adaptive method can achieve O (log d √ T ) , an improvement over O ( √ dT ) for the non-adaptive method.

### CONSIDER (4)

- **Correctness** _(_minor_, medium confidence)_
  Verify the RProp special-case derivation, because the update shown drops the dependence on the bias-corrected moments and the stated equality only holds under the zero-memory limit after taking the sign of the gradient. The author should either spell out the limit more carefully or remove the claim if it is only heuristic.
  · sections: s8
  > [s8] Rprop can be retrieved as a special case of Adam where β 1 = 1 and β 2 = 1 , i.e. the case with zero memory. In this case Adam's bias correc

- **Presentation** _(_minor_, medium confidence)_
  Clarify the notation in the introduction and algorithm so readers can follow the method without tripping over inconsistent or malformed wording; the current prose has several awkward phrases and a typo-like derivation that make the main algorithm harder to parse than necessary.
  · sections: s3
  > [s3] The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients;

- **Presentation** _(_minor_, medium confidence)_
  Clean up Section 6’s opening so the scope of the experiments is immediately clear; the current section header and lead-in are too generic, and the specific experimental settings are only spelled out later, making the section’s main message harder to locate quickly.
  · sections: s9
  > [s9] To empirically evaluate the proposed method, we investigated different popular machine learning models, including logistic regression, multi

- **Story** _(_minor_, medium confidence)_
  Tighten the statement of the core contribution so it is not just a list of properties. The paper should spell out the algorithmic contribution in one concise claim and then refer back to that claim throughout, instead of alternating between ‘Adam’ as a method, its update-rule properties, and its empirical behavior.
  · sections: s3
  > [s3] We propose Adam , a method for efficient stochastic optimization that only requires first-order gradients and requires little memory.

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
