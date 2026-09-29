# Stage182: isolate attention head dimension in the hybrid Transformer

## Frozen question and objection

Stage143's validation score is 1.399686162 BPB, 0.049686 above the desired
1.35. Stages147/150/155 changed width or depth and did not replace it.
Stage72/74 used six heads but also changed width, FFN size and global/local
allocation. They cannot isolate the attention-head count. Stage54 uses width
288 and eight heads (36 dimensions/head). The question is whether **four
heads (72 dimensions/head)** improve its contextual representation without
increasing the parameter count or adding another inference expert.

Two-sentence hypothesis: medium-frequency targets absent from the input
prefix remain difficult for the current small-data model. Wider per-head
attention may learn more useful semantic associations from the same 256-token
context while keeping the same output and training objectives.

The strongest objection is that fewer independent heads could reduce pattern
diversity, and OpenVINO's CPU kernel may be slower for 72-dimensional heads.
This is a single structural pilot, not a head-count sweep or seed search.

## Predeclared resource and quality gates

1. Copy Stage54's exact config and change only `heads: 8` to `heads: 4`.
   Check this mechanically before any run. Keep seed 17, batch 32, fixed
   train text/tokenizer, R-Drop, deep/future losses, optimizer, dropout and
   7,200-step learning-rate trajectory unchanged.
2. Before training, use a random-weight **input-only** ONNX/OpenVINO FP32
   feature graph on Windows. Require max hidden error <=3e-4, projected
   graph + 25,000,000-byte head/count reserve + 200,000-byte source reserve
   <=64 MiB, and eight interleaved four-thread feature-call median <=1.25x
   the exact Stage143 graph. No validation targets or test data are scored.
3. If the resource screen passes, train exactly the first 2,400 updates of
   the Stage54 seed-17 trajectory, presenting **19,660,800 primary targets**.
   Compare the fixed step-2,400 complete GPU FP32 validation BPB against
   Stage54's same-step **1.5199503686120217**. Require at least **0.015 BPB**
   improvement (<=1.5049503686120217) to authorize a fresh, fixed 7,200-step
   run. Intermediate validation points are diagnostics, not selection.
4. Even a passing pilot does not establish a new candidate. A full-run
   checkpoint must beat Stage143 on complete CPU FP32 validation and pass
   the complete three-repeat CPU/RAM/assets and clean-extract gates before
   promotion. Stage143 files remain untouched. No test scoring before
   method freeze.

This is a low-overhead representation test, not a forecast that four heads
can close 0.049686 BPB. If the fixed gate fails, stop without follow-up
head-count tuning.

## Input-only resource preflight

The Windows FP32 OpenVINO screen passed before training. Its random-weight
graph was **31,861,057 bytes** and differed from the eager PyTorch hidden
states by at most **3.70e-6**. The projected graph plus conservative
head/count/source reserve was **57,061,057 bytes**, below 64 MiB. Eight
interleaved four-thread feature-call medians were 1.214639 seconds for the
four-head graph and 1.247566 seconds for the exact Stage143 reference,
a **0.973607x** ratio. This is input-only feasibility, not a complete
predictor resource qualification or a quality score. Exact timing samples,
configuration/source/graph hashes and the no-test flag are in
`../results/stage182-head4-preflight.json`.

## Completed matched pilot and stop decision

The Windows run completed all 2,400 updates and **19,660,800 primary train
targets** in 552.06 recorded training seconds. All eight scheduled complete
GPU FP32 validation calls covered **376,599 targets / 1,148,007 raw bytes**.
The fixed endpoint scored **1.5188114758795954 BPB**, versus the archived
same-seed/same-target Stage54 endpoint **1.5199503686120217 BPB**. The gain
is only **0.0011388927 BPB**, not the predeclared 0.015. Intermediate points
were not used for selection. CUDA peak allocated/reserved memory was
4.403/4.742 GB, a training metric rather than the assignment's CPU RAM gate.

The job ended successfully and re-ran the supplied fixed-file verifier.
The remote checkpoint's independently queried SHA-256 matches the training
record: `ee8decea07ae7a9925fa099521259029e41290ef5c2c359c04c153572f7bf2f9`.
The local [audit](../results/stage182-evidence/audit.json) checks the exact
config-only head change, source hashes, job completion, scheduled validation
coverage, target count, control endpoint and checksum. Raw run/metrics/
progress/status JSON are archived beside it; the console transcript and
unpromoted checkpoint stay on the private Windows machine.

**Stop Stage182.** No 7,200-step run, trained graph export, full CPU/RAM/
asset qualification or test scoring is justified. This narrow result shows
that changing 8 to 4 heads under this fixed recipe does not materially close
the 0.049686-BPB gap. It does not claim all head structures are equivalent.
Stage143 remains the protected submission candidate.
