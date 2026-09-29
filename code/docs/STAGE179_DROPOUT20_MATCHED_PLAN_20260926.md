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

## Completed result and stop decision

The Windows one-off job finished successfully at 7,200 updates. The two new
equivalence tests, inherited hybrid R-Drop tests and fixed-file checks passed.
Its 24 scheduled complete-validation calls covered all **376,599 targets**
and **1,148,007 raw bytes** each. The independent exported-average Windows
CPU FP32 score was **1.444744044563801 BPB**, versus Stage54's same-budget
**1.4285940451894024**: the stronger-dropout model is **0.016149999 BPB
worse**, not 0.015 better. The fixed endpoint GPU FP32 score was
1.445773063; it was not used for candidate selection.

The job presented **58,982,400 primary training targets** in 1,934.61
training seconds, with peak CUDA allocated/reserved memory **5.648/5.985 GB**.
Runtime is host-condition-specific; it is not a quality comparison. The
training endpoint SHA-256 was
`3b1d4cf2d3657a7c162385df044400a067faa2ea378421d2a6369d1e133b7936`;
the exported average SHA-256 was
`2f050d88339399f22854265bd9b2ed4b830beeffbb83f28cca2f1d757027de17`.
Both matched fresh Windows file hashes, and the local auditor rechecked all
16 recorded source hashes, job completion, training target count, validation
coverage, checkpoint identity and the prespecified gate. Evidence is in
`../results/stage179-evidence/`; raw PowerShell transcript stays on the
private Windows machine.

**Stop this route.** No continuation, count rebuild, end-to-end CPU/RAM/asset
qualification, clean-extract run or test scoring is justified for Stage179.
This negative result is specific to global dropout 0.20 under this fixed
schedule; it does not prove all stronger regularization fails. Stage143
remains the protected 1.399686162-BPB validation candidate.
