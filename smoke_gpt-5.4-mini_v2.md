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

This document is a paper introducing Adam as a stochastic, first-order gradient-based optimization method. It presents the algorithm, bias correction, convergence analysis under an online convex-learning framework, related work, and several empirical comparisons. Overall, it is a well-organized but rough draft: the core algorithmic idea is clear, but some derivations and claims need correction or qualification, especially in the update-rule and analysis sections. The experiments are also not fully specified, which makes the benchmark comparisons hard to reproduce or verify. Presentation issues recur in the update-rule exposition and the experimental write-up.

## Strengths

- Clear high-level organization: algorithm, theory, related work, and experiments are separated into distinct sections.
- The paper introduces a simple, computationally efficient optimizer with adaptive moment estimates and an intuitive connection to AdaGrad and RMSProp.
- Related work is broad and situates Adam among several relevant stochastic optimization methods.
- The method is presented as suitable for large-scale, high-dimensional, noisy, sparse, and non-stationary optimization problems.

## Weaknesses

- The AdaGrad special-case derivation is mathematically wrong as written and uses an unsquared gradient term where squared gradients are required.
- The convergence/regret discussion is not fully consistent with the stated online convex-learning assumptions and overstates the scope of the result.
- Several theoretical claims need qualification or cleaner notation, including the RProp special case, the effective step-size bound, and the gradient-scale invariance claim.
- The experiments lack enough protocol detail to trust or reproduce the benchmark comparisons: splits, metrics, tuning ranges, stopping rules, and validation/test usage are unclear.
- The empirical-results presentation is hard to follow in places because figures, references, and aggregation/interpretation are not clearly explained.
- The introduction and related-work discussion make broad superiority and novelty claims that should be narrowed to match the evidence and comparisons actually shown.
- There are multiple minor presentation and wording issues, including inconsistent optimizer naming and compressed equations that reduce readability.

---

## Findings (LLM-investigated) (20)

### MUST FIX (3)

- **Correctness** _(_major_, medium confidence)_
  Correct the AdaGrad special-case derivation, since the limit expression is mathematically wrong as written and uses an unsquared gradient term where squared gradients are required.
  · sections: s8
  > [s8] Note that if we choose an infinitesimal β 2 then lim β 2 → 0 ̂ v t = t -1 · ∑ t i =1 g t .

- **Evaluations** _(_major_, medium confidence)_
  Add the missing evaluation protocol details needed to trust the benchmark comparisons: the paper should report what splits were used, what metric was optimized/evaluated beyond vague “convergence,” and whether the plotted curves are on validation or test data. As written, readers cannot verify that the reported curves reflect a sound experimental protocol.
  · sections: s10
  > [s10] Figure 1: Logistic regression training negative log likelihood on MNIST images and IMDB movie reviews with 10,000 bag-of-words (BoW) feature

- **Story** _(_major_, medium confidence)_
  Make the convergence/regret discussion internally consistent and explicit about its assumptions and scope: the theorem is for online convex learning under boundedness conditions, and the text should not overstate that result as a general convergence guarantee for Adam or mix incompatible rate statements.
  · sections: s7
  > [s7] We analyze the convergence of Adam under the online learning framework proposed in Zinkevich (2003).

### SHOULD FIX (13)

- **Correctness** _(_moderate_, medium confidence)_
  Fix the RProp special-case derivation so the notation is internally consistent and the final sign-update equality is written cleanly without dropping the parameter subscript.
  · sections: s8
  > [s8] In this case Adam's bias correction terms equal 1, and the update is: θ t +1 = θ t -α · m t / √ v t = θ t -α · g t / √ g 2 t = θ -α · sign (

- **Correctness** _(_moderate_, medium confidence)_
  Clarify the effective-step-size bound by stating the assumptions and whether the claim is about signed updates or magnitudes, since the current text mixes a heuristic bound with a signed inequality.
  · sections: s5
  > [s5] The effective stepsize has strong upper and lower bounds: -α · β 1 /β 2 ≤ ∆ t ≤ + α · β 1 /β 2 .

- **Evaluations** _(_moderate_, medium confidence)_
  Provide a more precise description of the hyperparameter search protocol, including the exact ranges/notation and tuning setup, so the benchmark comparisons are reproducible and fair.
  · sections: s9
  > [s9] The hyper-parameters, such as learning rate and momentum, are searched over a dense grid and the results are reported using the best hyper-p

- **Evaluations** _(_moderate_, medium confidence)_
  Provide enough training-detail to reproduce the logistic-regression comparison, including the model parameterization and optimization schedule. The section names the datasets and batch size, but it does not specify the regularization strength, number of epochs/iterations, learning-rate ranges, or any stopping rule.
  · sections: s10
  > [s10] We compare Adam to accelerated SGD with Nesterov momentum and AdaGrad using mini-batch size of 128.

- **Evaluations** _(_moderate_, medium confidence)_
  Make the multi-layer-network comparison reproducible by stating the exact training setup for Adam versus SFO and the other stochastic methods. The section describes the architecture only at a high level and then asserts wall-clock and iteration advantages without giving enough optimization details to replicate the runs.
  · sections: s11
  > [s11] In our experiments, we made model choices that are consistent with previous publications in the area; a neural network model with two fully 

- **Evaluations** _(_moderate_, medium confidence)_
  Clarify the CNN experiment so the reader can reproduce the reported speedup and layer-wise behavior. The paper gives the broad architecture, but omits key training settings such as the exact learning-rate schedule, momentum/beta values, and any run-averaging needed to interpret the wall-clock comparison.
  · sections: s12
  > [s12] Our CNN architecture has three alternating stages of 5x5 convolution filters and 3x3 max pooling with stride of 2 that are followed by a ful

- **Presentation** _(_moderate_, medium confidence)_
  Reformat the compressed update-rule expression into a clearer displayed equation or pseudocode step so the bias-correction factors and update order are easy to parse.
  · sections: s4
  > [s4] Note that the efficiency of algorithm 1 can, at the expense of clarity, be improved upon by changing the order of computation, e.g. by repla

- **Presentation** _(_moderate_, medium confidence)_
  Tighten the presentation of the bias-correction/moment discussion by using consistent terminology and adding a concise signpost that states the practical takeaway of the derivation.
  · sections: s4
  > [s4] The moving averages themselves are estimates of the 1 st moment (the mean) and the 2 nd raw moments (the uncentered variance) of the gradien

- **Presentation** _(_moderate_, medium confidence)_
  Fix the experimental-results presentation so the section’s figures and references are ordered clearly and the reported comparisons explain how results are aggregated and interpreted.
  · sections: s12
  > [s12] Interestingly, although both Adam and AdaGrad have lower cost in the initial stage of the training in Figure 3 (left), Adam and SGD eventual

- **Significance** _(_moderate_, medium confidence)_
  Strengthen the related-work comparison by explaining why the cited optimizers are the relevant baselines and by qualifying the claims about Adam’s relationship to SFO, NGD, and other prior methods.
  · sections: s8
  > [s8] Optimization methods bearing a direct relation to Adam include RProp Riedmiller &amp; Braun (1992), RMSProp Tieleman &amp; Hinton (2012); Gr

- **Story** _(_moderate_, medium confidence)_
  The introduction/contribution statement is too broad and should more clearly state Adam’s specific optimization setting, its relation to prior methods, and which part of the paper is the core novelty versus supporting analysis.
  · sections: s3
  > [s3] Our method is designed to combine the advantages of two recently popular methods: AdaGrad Duchi et al. (2011), which works well with sparse 

- **Story** _(_moderate_, medium confidence)_
  Qualify the gradient-scale invariance claim, since the text currently presents it as unconditional even though it only holds under the moment-scaling assumptions used in the derivation.
  · sections: s5
  > [s5] The effective stepsize ∆ t is also invariant to the scale of the gradients; rescaling the gradients g with factor c will scale ̂ m t with a 

- **Story** _(_moderate_, medium confidence)_
  Soften the broad empirical superiority claim, because the evidence is based on a limited benchmark set and best-of-grid tuning rather than a basis for claiming Adam consistently outperforms other methods in general.
  · sections: s3
  > [s3] Empirically, our method consistently outperforms other methods for a variety of models and datasets, as shown in section 6.

### CONSIDER (4)

- **Presentation** _(_minor_, medium confidence)_
  Standardize terminology for the update ratio by either formally defining SNR as a new shorthand or avoiding the label entirely. As written, the paper introduces the term with a hedge, which can leave readers unsure whether it is a genuine concept or just an informal analogy.
  · sections: s5
  > [s5] With a slight abuse of terminology, we will call the ratio ̂ m t / √ ̂ v t the signal-to-noise ratio ( SNR ).

- **Presentation** _(_minor_, medium confidence)_
  Fix the heading/narrative flow in the experiments so each subsection’s main result is easier to locate. The paper jumps straight into model details and figure references without an upfront takeaway sentence, which makes the empirical section harder to navigate.
  · sections: s10
  > [s10] ## 6.1 EXPERIMENT: LOGISTIC REGRESSION

- **Presentation** _(_minor_, medium confidence)_
  Correct local wording and grammar errors in the experiments section, since they interrupt the presentation of the results and make the text harder to read. A light copyedit would materially improve clarity.
  · sections: s10
  > [s10] Weevaluate our proposed method on L2-regularized multi-class logistic regression using the MNIST dataset.

- **Presentation** _(_minor_, medium confidence)_
  Use consistent terminology for the optimizer name and avoid awkward variants such as “the Adam.” This is a presentation issue that recurs and makes the prose less polished than it should be.
  · sections: s10
  > [s10] According to Figure 1, we found that the Adam yields similar convergence as SGD with momentum and both converge faster than AdaGrad.

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
