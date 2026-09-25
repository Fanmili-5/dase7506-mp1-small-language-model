# Stage168: efficient sharpness-aware training of the retained Transformer family

## Problem-first selection (fixed before outcomes)

Stage143 scores 1.399686162 complete-validation BPB, above the aspirational
1.35. Its sampled-train versus validation mean-NLL gap is 0.635818 nats per
target. Extra width/depth, local multiscale mixing, an input-token mask and
frequency-weighted loss did not replace it. The gap motivates testing a
different optimization bias on the same Stage54 causal Transformer rather
than another architecture grid. The gap alone does not prove that SAM helps.

The ideation check considered ten distinct levers: more width/depth,
more attention blocks, multiscale convolution, learned spelling output,
exact train-text retrieval, a second neural branch, frequency-weighted
loss, whole-token masking, relative-position bias and sharpness-aware
optimization. The first eight have relevant failed or insufficient pilot
evidence in Stages145–165; relative-position bias changes CPU export and
has no specific error evidence. Sharpness-aware optimization has zero
inference cost and directly tests whether optimizer geometry matters.

Bahri, Mobahi and Tay, [*Sharpness-Aware Minimization Improves Language
Model Generalization*](https://aclanthology.org/2022.acl-long.508/), ACL
2022, provide the broad motivation and an efficient quarter-batch ascent
variant. Their reported language-task gains are primarily from fine-tuning
pretrained T5/mT5, **not** this from-scratch causal language model; no
numerical result is imported. We implement the generic SAM first-order
perturbation ourselves, not their code.

Two-sentence hypothesis: this tiny training corpus leaves the Stage54
Transformer at a sharp solution that generalizes poorly to held-out
articles. A gradient-aligned parameter-neighborhood step may improve its
validation loss without spending any of the fixed test-time budget.

## Frozen matched pilot

- Keep Stage54's exact architecture, config, seed 17, training text,
  fixed tokenizer, CPU scorer, batch-32 sampled windows, AdamW settings,
  main/deep/future R-Drop objective, 7,200-step learning-rate curve and
  evaluation every 300 updates. Use the same first **2,400 updates** and
  **19,660,800 primary next-token targets** as its archived control.
- At each update, use the first eight sampled windows for an R-Drop ascent
  gradient at current parameters. Perturb all trainable parameters by
  `rho * g / ||g||_2`, with fixed `rho=0.05`, then compute the ordinary
  R-Drop descent gradient on all 32 windows at the perturbed parameters.
  Restore the parameters before clipping the descent gradient and calling
  the unchanged AdamW update. The ascent pass adds training compute and
  stochastic target presentations; it does not change the set of sampled
  primary windows, optimizer update count or inference model.
- Preflight exact initial weights/inference predictions against Stage54;
  finite gradients and nonzero radius on a synthetic optimizer check;
  exact restoration before optimizer update; fixed-file and causality tests.
  The ascent microbatch uses no validation/test data. No rho, seed or
  checkpoint grid is permitted.
- Compare only the fixed 2,400-step **endpoint** complete GPU FP32
  validation BPB to Stage54's same-schedule **1.5199503686120217**.
  Require improvement at least **0.015 BPB**, i.e. endpoint at most
  **1.5049503686120217**, before a fresh fixed 7,200-step run. Intermediate
  300-step scores are diagnostic only. A passing pilot does not establish
  improvement over Stage143 or resource qualification; it only licenses
  continuation. A failing pilot stops this route before CPU export/test.
- Stage143 checkpoint/graph and the test split remain untouched. The
  final predictor would still require complete CPU FP32 validation below
  Stage143, a matched count/gate rebuild, all three fresh resource checks
  and clean-extract reproduction before promotion.

The strongest objection is that R-Drop and weight decay already regularize
the model, while the ascent gradient on only eight windows may be noisy.
Failure at the prespecified endpoint would reject this recipe, not every
possible SAM variant. Record training seconds and extra stochastic passes
instead of claiming equal training compute.
