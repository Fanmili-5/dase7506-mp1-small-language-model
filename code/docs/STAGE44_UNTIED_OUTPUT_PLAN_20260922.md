# Stage44: untied input and output embeddings

All accepted Transformer runs force the same 2,048x256 matrix to represent both
input-token features and output-class decision vectors. Stage44 tests whether
that constraint is limiting the small model. It keeps one output projection
with the same tensor shape and arithmetic but gives it an independent parameter
matrix, adding 524,288 parameters (2 MiB in FP32) without adding a matrix
multiplication.

The 8x256 backbone, content-copy route, seed 17, AdamW recipe, schedule, batch,
7,200 updates, 58,982,400 primary targets, row dropout, deep supervision and
+2/+3 multi-token prediction remain fixed. The architecture has a different
parameter initialization after the shared backbone because the new output
matrix is real capacity, not a seed perturbation. Selection is the same fixed
last-five checkpoint average and validation-only score.

The exported model retains separate input/output matrices and removes every
training-only head. It must beat Stage26 by at least .003 BPB before MKN and
resource follow-up. CPU arithmetic is unchanged, but the exact checkpoint still
requires asset and timing qualification if advanced. No test split is scored.

## Result

The fixed last-five average scored **1.4660668981 BPB** on independent CPU FP32
validation (checkpoint SHA256
`1045052bc7563b632a5c83e15b7532b967010742e5198c55bcbf2530f0d476c1`).
The matched tied Stage26 average is 1.4649939083, so untying is worse by
0.0010729898 BPB. Its large early advantage disappeared by step 1,200 and the
7200-step endpoint remained worse than the tied control. The candidate is
rejected without MKN calibration or resource qualification. It processed
58,982,400 primary targets in 896.50 GPU training seconds; peak allocated GPU
memory was 2.026 GB. No test split was scored.
