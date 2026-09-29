# Stage212 sliding local attention: resource screen failed

The [pre-outcome contract](STAGE212_SLIDING_LOCAL_ATTENTION_PLAN_20260928.md)
fixed a seven-position content-adaptive local attention replacement for
Stage54's four convolution blocks. This was a synthetic-input feasibility
screen, not a trained model or a validation/test evaluation.

The first Windows attempt stopped before graph export, GPU use or data access:
the script's config guard omitted the already-unused `conv_kernel` key from
its list of allowed local-mixer changes. Commit `4e3924e` corrected only
that guard; it did not alter the model or the fixed gate. The separately
named b attempt completed the unchanged synthetic screen.

The b attempt copied **57** identical named tensors from the Stage54
control. Unit tests verified the local attention against an explicit
seven-position causal mask, train/inference state equivalence, full-vocab
normalization, future-token and cross-row independence, and finite local
query/key/value gradients. The OpenVINO feature graph agreed with eager
PyTorch to **4.0531e-6** maximum hidden error. Its conservative projected
inference assets were **58,748,345 bytes**, below 64 MiB.

The fixed eight-pair, four-thread, batch-32 Windows FP32 OpenVINO median was
**1.864310 s** for Stage212 versus **1.243451 s** for the SHA-pinned
Stage143 reference feature graph: **1.499304x**, above the predeclared
**1.20x** feature-speed gate. This is a failed *preflight gate*, not a
measurement of the course's complete <=5x CPU limit. The script correctly
did not proceed to the synthetic GPU step or a data-bearing training pilot.
There is no Stage212 validation BPB, full CPU/RAM qualification, checkpoint
promotion or new test score. Stage143 remains the protected fallback,
still above the student's score thresholds.

The [raw Windows result](../results/stage212-sliding-preflight-b.json)
includes timings and SHA-256 identities. Its script and implementation
hashes match the committed local files. The random-weight ONNX graph remains
on the Windows host and is not a submission asset.
