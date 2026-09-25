# Stage186: narrow global path beside each local convolution

## Fixed question before measurement

Stage143 scores 1.399686162 complete-validation BPB, 0.049686 above the
aspirational 1.35. Stage177's four full-width parallel attention paths failed
the input-only Windows CPU feature gate (1.438569x the Stage143 reference);
Stage154's single heterogeneous final block passed resources but gained only
0.005998 BPB at the matched 2,400-step endpoint. The untested middle point
is to put a **narrow** global path beside every local block: retain the exact
Stage54 8x288 architecture and its four original attention/four original
convolution blocks, and add two 36-dimensional RoPE heads (72 total channels)
in parallel to each of conv blocks 2/4/6/8. This is one fixed architecture,
not a head-count sweep.

The extra attention output projection starts at zero. Shared seed-17
parameters, post-construction RNG state and step-zero evaluation output must
be identical to Stage54. Every added branch must receive nonzero gradients.
Its projection and query/key/value tensors become real inference weights;
there is no training-only shortcut. Training uses supplied text only; no
test targets, cross-window state or external data enter the mechanism.

## Predeclared sequence and gates

1. Structural unit tests certify shared weight/RNG equality, zero-start
   output equality, causal independent-window predictions, normalization,
   gradient reachability and strict train-to-inference state transfer.
2. On Windows, export a random-weight input-only FP32 OpenVINO feature graph
   with batch 32 and context 256. Require eager/graph maximum hidden error
   <=3e-4; graph bytes +25,000,000-byte compact head/count reserve +200,000
   source reserve <=64 MiB; and eight interleaved four-thread feature-call
   median <=**1.25x** the exact Stage143 graph. Validation **inputs** but no
   validation labels may be used. Failure stops before training.
3. Only if all preflight gates pass, train one seed-17 pilot through exactly
   the first **2,400** updates of the Stage54 7,200-step LR trajectory,
   batch 32 and **19,660,800 primary target presentations**. Preserve its
   R-Drop, deep/future losses, dropout, optimizer, sampled windows and every
   300-step complete GPU FP32 validation. Compare only the fixed step-2400
   endpoint to Stage54's same-seed **1.5199503686120217 BPB**. Require at
   least **0.020 BPB** improvement (<=1.4999503686120217) to authorize a
   separately specified 7,200-step full run. Earlier points are diagnostics.
4. A passing pilot is not a candidate. The full run would need to beat
   Stage143 on complete CPU FP32 validation after train-only MKN/gate fitting,
   then pass three-repeat complete CPU <=5x, RSS <=4 GiB, assets <=64 MiB and
   clean-extract reproduction. Stage143 stays protected. No test scoring
   before method freeze.

Strongest objection: 72 channels per branch may be too weak to contribute
material global context even if fast enough, while four attention matrices
can still cost more than the input-only gate allows. The preflight and fixed
endpoint test those objections without parameter/seed tuning after outcomes.

## Completed input-only Windows preflight

The exact fixed config and source hashes matched the transferred files. The
random-weight FP32 OpenVINO feature graph contains **8,270,785 neural
parameters** and occupies **32,936,394 bytes**. Eager/portable RMSNorm output
was exactly equal; maximum eager/OpenVINO hidden error was **3.81e-6**, below
3e-4. With the prespecified 25,000,000-byte head/count and 200,000-byte
source reserve, projected assets were **58,136,394 bytes**, below 64 MiB.
Eight interleaved four-thread batch-32 feature calls had medians **1.406147
seconds** for Stage186 and **1.259486 seconds** for the exact Stage143 graph,
a **1.116446x** ratio below 1.25. All predeclared pilot-admission gates
pass. This uses validation inputs only; it is neither a BPB score nor a
complete predictor CPU/RAM/asset qualification. The source/config/graph
hashes and all timings are in
`../results/stage186-evidence/preflight.json`.
