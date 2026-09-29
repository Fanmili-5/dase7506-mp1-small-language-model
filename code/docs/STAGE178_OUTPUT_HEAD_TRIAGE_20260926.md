# Stage178: output-head bottleneck and submission-window triage

## Question

Could a richer vocabulary output head provide the roughly **0.049686 BPB**
needed to move the protected Stage143 full-validation result
(`1.399686162`) below the aspirational `1.35` threshold? This is an evidence
review, not a new candidate, validation fit, method freeze, or test run.

## Existing controlled evidence

| Mechanism | Relevant observation | Decision |
| --- | --- | --- |
| Two-component mixture of softmaxes | Stage14 same-target, seed-17 average `1.596399` versus the tied single-softmax control `1.547833` BPB; the mixture was `0.048566` worse. | Do not repeat the same head. |
| Independent output matrix | Stage44 last-five average `1.466067` versus the matched tied Stage26 `1.464994`; `0.001073` worse. | Untying alone does not explain the gap. |
| Frozen-backbone rank-16 output residual | Stage69's fixed epoch-3/4/5 average `1.419483` versus its exact Stage67 start `1.415098`; `0.004385` worse. Training loss fell while validation worsened. | A larger head on fixed representations is not an evidenced route. |
| Context-conditioned spelling residual | Stage160 exact Stage143 start `1.399686195`; fixed 2,400-update endpoint `1.402680`, `0.002993` worse. | Do not extend this head-only recipe. |
| First/last-byte training-only heads | Stage174 matched endpoint `1.520019` versus Stage54 control `1.519950`; `0.000069` worse. | This particular extra supervision did not improve early generalization. |

These comparisons have different training budgets and baselines. They are
**not** a pooled effect estimate and do not prove every nonlinear output
family fails. They do, however, falsify the simple claim that the existing
softmax rank limit is the demonstrated dominant bottleneck. A fresh
mixture-of-softmax sweep or output-only adaptation is a poor use of the
remaining submission window.

Stage146's in-sample train diagnostic was `2.321660` versus complete
validation `2.957478` nats per target, a `0.635818`-nat difference. Because
the train windows are in-sample, this is not a clean generalization-gap
estimate, but it cautions against adding unregularized head capacity.
Stage151 also localizes high error to medium-frequency next tokens absent
from the current causal window. This group is defined retrospectively with
the true target and cannot be used as an inference gate.

## Decision and guardrails

1. Retain Stage143 as the sole resource-qualified submission fallback. The
   26 September pretest-readiness audit rechecked its checkpoint/graph hashes,
   376,599-target validation identity, clean-extract agreement, Windows
   four-thread CPU ratio `3.617702`, peak RAM `2,176,729,088` bytes and
   assets `55,810,412` bytes. This is a Windows resource pass, **not** a
   universal hardware guarantee: the one-thread Linux ratio was `5.502555`.
2. Do not begin another output-head experiment without a new mechanism and
   a prespecified diagnostic that could plausibly cover a material fraction
   of `0.049686` BPB. A small positive result is not evidence for `<1.35`.
3. Preserve the untouched test split while development continues. Before
   29 September (UTC+8), decide whether to freeze the strongest validated
   candidate, run the one permitted post-freeze full-test score, and submit
   the student ID and score. Immutable code/bundle links and a matching
   <=10-page report are due by 30 September. The Stage143 report preview is
   watermarked **not for submission** and lacks a Stage143 test score.

Source evidence: `results/stage14-evidence/screen.json`,
`docs/STAGE44_UNTIED_OUTPUT_PLAN_20260922.md`,
`docs/STAGE69_OUTPUT_LORA_PLAN_20260923.md`,
`docs/STAGE160_CONTEXTUAL_MORPHOLOGY_RESIDUAL_PLAN_20260925.md`,
`docs/STAGE174_BYTE_AUXILIARY_SUPERVISION_PLAN_20260925.md`,
`results/stage146-evidence/stage143-generalization.json`, and
`results/stage143-evidence/pretest-readiness.json`.
