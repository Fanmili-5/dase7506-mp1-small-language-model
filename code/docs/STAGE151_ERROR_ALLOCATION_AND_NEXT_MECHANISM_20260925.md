# Stage151: quantify the remaining error before another model run

The sub-1.35 validation target is still 0.049686 BPB below the resource-
qualified Stage143 score. Stage147 and Stage150 separately changed width and
depth with matched first-2,400-step training trajectories; their gains were
only 0.011346 and 0.011507 BPB, both below their fixed 0.015 continuation
gates. Stage146/148 exact lookup added only about 0.003 BPB in target-only
diagnostics, and Stage149's distant co-occurrence mixture hurt. This makes
another nearby width/depth/count grid a poor next use of the laptop.

Using the exact cached Stage143 validation target probabilities, the fixed
train-token frequency categories, and only **within-window causal prefix**
membership, Stage151 partitions all 376,599 validation targets:

| Train-token frequency / prefix status | Targets | Mean target NLL (nats) |
| --- | ---: | ---: |
| 100–999, unseen in current window | 57,247 | 4.700063 |
| 100–999, seen in current window | 13,210 | 2.654682 |
| Other frequencies, unseen | 171,156 | 3.215610 |
| Other frequencies, seen | 134,986 | 1.920788 |

To reduce *total* BPB by 0.05 through the first group alone would require
removing **39,786.89 nats**, or **0.6950 nats per one of those 57,247
targets**. This is an accounting identity, not an achievable bound: any new
normalized predictor can redistribute probability and change losses in all
groups. The group itself is defined using the true target for retrospective
diagnosis and must never become an inference-time gate.

## Next high-leverage candidate order

1. **A train-only character/byte conditional expert over the fixed BPE
   vocabulary**, evaluated as a fully normalized probability distribution.
   It could share spelling patterns for medium-frequency tokens absent from
   the current prefix, a failure mode not addressed by exact token lookup.
   Prior simple byte-composed embeddings improved only 0.0008–0.0023 BPB, so
   a new pilot must show a substantially different mechanism. First require
   a bounded full-vocabulary diagnostic on validation and an explicit CPU/
   asset plan; a target-only string likelihood is not a BPB score.
2. **A shared-trunk heterogeneous expert** that keeps two small upper-layer
   branches. Stage91's complete two-neural ensemble reached 1.383202 BPB
   but exceeded the resource envelope; shared lower layers could trade some
   diversity for deployability. It still needs a causal implementation,
   normalized mixture and resource preflight, and even the over-budget
   ensemble did not reach 1.35.
3. **Frequency-aware training regularization**, with zero inference overhead,
   to target the documented train/validation gap. This is easiest to test,
   but prior uniform row dropout and byte sharing make a >0.05 BPB leap
   unlikely. It should use a matched control, not another seed screen.

The first candidate is prioritized for a *feasibility diagnostic*, not
declared a winner. No new model is promoted and no test split is scored.
Full counts and source hashes are in
`../results/stage151-evidence/error-intersection.json`.
