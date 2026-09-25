# Stage179: one fixed stronger-dropout generalization control

Stage143 scores **1.399686162 BPB** on complete validation. Its in-sample
sampled-train versus validation mean-NLL difference is **0.635818 nats per
target**; this is not a clean generalization estimate, but overfitting is a
plausible issue. Stage18's row dropout and FFN-hidden dropout gave small
positive matched results, whereas the Stage165 positionwise mask and Stage168
SAM pilots worsened. We therefore test one bounded, simple alternative rather
than claim that stronger regularization is known to work.

## Frozen experiment and advancement gate

- Keep Stage54's eight-block alternating attention/causal-convolution model,
  prefix-copy head, seed 17, tokenizer, train text, random-window sampler,
  batch 32, 7,200 optimizer updates, AdamW/LR schedule, R-Drop and all auxiliary
  objectives exactly as recorded. Present **58,982,400 primary next-token
  targets**, matching Stage54. Change only the main dropout probability from
  `0.10` to **`0.20`**. The separate vocabulary-row dropout stays `0.10`.
- Before Windows training, verify only this config field changed, and the same
  seed gives identical initial tensors and evaluation-mode predictions.
  Run fixed-file verification and inherited hybrid R-Drop tests on Windows.
- Use only the prespecified average of steps **6000/6300/6600/6900/7200**.
  Score the exported model on complete CPU FP32 validation. The Stage54
  same-target control is **1.4285940451894024 BPB**. Advance only if the
  average improves by at least **0.015 BPB**, i.e. scores at most
  **1.4135940451894024 BPB**. The intermediate curve and endpoint are
  diagnostics, not replacement selection rules.
- A passing result would still require a separate continuation/count plan,
  complete CPU FP32 validation below the protected Stage143, exact compact
  export, three fresh CPU/RAM/asset resource repetitions under 5x/4GiB/64MiB,
  and clean-extract reproduction. No test scoring before a later method freeze.

The strongest objection is that R-Drop and existing row dropout may already
regularize the model, while a global `0.20` dropout could impede learning.
Failure rejects this one setting and schedule, not all regularization. A
positive 0.015 early-family gain would not itself establish `<1.35`.

The brainstorming failure-boundary/simplicity check selected this as a
zero-inference-overhead, one-variable control rather than another seed,
output-head or size sweep. Stage143's files remain protected throughout.
