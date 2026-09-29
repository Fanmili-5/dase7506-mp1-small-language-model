# Stage219 result: Stage155 continuation misses its fixed gate

The [pre-outcome plan](STAGE219_EQUAL_BUDGET_CONTINUATION_PLAN_20260928.md)
tested whether Stage155's short training lineage explained its worse result
relative to Stage143. The Windows job completed all 4,800 additional
updates (39,321,600 new primary target presentations, 98,304,000 nominal
cumulative) from the SHA-pinned Stage155 five-checkpoint average. The
scheduled task and trainer both report completion, all archived source
hashes match the repository, and the development loader used supplied
train and validation only. No test score was produced.

Stage155's fixed starting average scored **1.409877270416759 BPB** on the
complete 376,599-target validation split. Stage219's step-4,800 endpoint
scored **1.406290303269237**, and its prespecified last-five parameter
average scored **1.406129732661200**. The selected gain is therefore
**0.003747537755559 BPB**, below the predeclared **0.008** continuation
gate. The selected model is still **0.006443571 above** the protected
Stage143 candidate (1.399686162042141) and **0.056129733 above** the
student's 1.35 validation objective. The early low-LR restart initially
worsened BPB (1.418035 at step 600); the late improvement did not close
the gap.

**Decision:** stop this Stage155 extension. Do not start another
continuation phase, export a new inference bundle, claim resource
qualification, or test-score it. The experiment supports only a modest
benefit from extra training at this scale; it does not establish that
architecture is incapable of better scores under other recipes. Stage143
remains the only protected resource-qualified candidate at 1.399686
validation / 1.415658 frozen test, so the coursework target is unfinished.

The complete validation curve, training counters, source hashes, averaged
checkpoint SHA-256 `553143b7839231f1673fa7259c2a42a5cac6e40713227bc49a95c22e75b7e5a5`,
and terminal job status are archived under
[`../results/stage219-evidence/`](../results/stage219-evidence/).
