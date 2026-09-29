# Stage204: causal adjacent-token input representation (pre-outcome plan)

This is one fixed architecture test on the supplied train/validation splits.
It does not touch the protected Stage143 checkpoint, graph, package or test.

## Problem-first selection

Stage151 found a high-loss group of medium-frequency targets absent from the
current causal prefix. The current local convolution can learn adjacent-token
interactions, but it must spend depth and training samples to construct them
from separate token embeddings. The proposed model gives each position a
direct, trainable representation of the **observed current and previous BPE
token**. The first position uses a zero pair vector; the pair history resets
for each independent 256-token evaluator window. There is no future-token,
target, cross-window or external-data feature.

An initial mechanism filter considered: wider/deeper attention (Stages
145/147/150/155), sparse MoE (48), trained neural/count routing (176),
word-prefix counts (191), output class context (203), spelling residuals
(160/174), second-neural compression (157/158/169), more of the current
training schedule (109/120), a long-range state-space mixer, and contextual
pair embeddings. The first eight families have measured insufficient or
negative quality evidence. A state-space core is more distinct but has no
demonstrated portable CPU execution path before the deadline. Pair embeddings
are the remaining cheaply falsifiable single-backbone hypothesis, not a
claim that they will close the approximately 0.05-BPB validation gap.

## Fixed architecture and matched control

Keep the Stage54 eight-block alternating attention/causal-convolution
Transformer, tied output head, prefix copy, deep supervision, +2/+3 auxiliary
targets and R-Drop recipe unchanged. After the ordinary token embedding,
add `0.5 * pair_embedding[hash(previous,current)]`. Hash pairs with the
fixed integer rule in `student_stage204_hashed_bigram_rdrop.py` into exactly
8,192 buckets. Bucket 0 is reserved for the first input position and always
zero. The table is learned solely by the ordinary training loss; no pair
counts, held-out labels, tokenizer modification or external weights are used.
Training and evaluation use the same causal input construction.

The strongest objection is that the existing seven-wide local convolution
already learns short interactions, while hash collisions and roughly 2.36M
extra trainable scalars may overfit the 3.61M-token corpus. One fixed seed-17
pilot is therefore compared to Stage54 at the same first 2,400 updates,
batch 32, 19,660,800 primary target presentations and **the same first
2,400 learning rates of its 7,200-step schedule**. The added parameters are
the architecture intervention; there is no seed, bucket count, hash,
embedding scale, dropout, training objective or checkpoint sweep.

## Ordered gates

1. Unit-test zero-scale exact parity with Stage54, fully normalized output,
   causal prefix, row independence and first-position reset. On Windows,
   preflight one synthetic batch-32 BF16 optimizer step with peak allocated
   GPU memory <=7 GiB and below the device total. Export a random-weight
   input-only FP32 feature graph (not a score), check OpenVINO/eager hidden
   error <=3e-4, project feature graph plus Stage143's unchanged other
   inference assets and a 262,144-byte source reserve <=64 MiB, and require
   the feature time <=1.25x the SHA-pinned Stage143 graph in eight interleaved
   four-thread runs. Reject before training if any feasibility gate fails.
2. If admitted, train the fixed matched 2,400-step candidate on train only.
   Score its **step-2,400 endpoint** on the entire fixed validation split.
   Continue to a fresh full 7,200-step training run only if it improves the
   Stage54 step-2,400 control BPB `1.5199503686120217` by at least **0.030**
   (candidate <=`1.4899503686120217`). No early checkpoint is promoted.
3. A full run must beat Stage143's `1.399686162042141` on complete CPU FP32
   validation and pass exact causal/normalization tests, three-repeat CPU
   <=5x baseline, peak RAM <=4 GiB, and <=64 MiB counted assets. The
   aspirational goal is <1.35 complete validation; a merely passing pilot
   does not claim it. New test scoring would require a separately committed
   method freeze and authorization. A failing gate stops this fixed route
   without adjusting the hashed-pair choices.

Report both feasibility and quality failures explicitly. The course's
existing frozen Stage143 full-test score 1.415657616535609 is not an
acceptable <1.38 outcome, and this diagnostic cannot be called a submission.

## Preflight-a export correction, before any quality outcome

The first input-only preflight failed the `<=3e-4` OpenVINO/eager hidden-state
parity gate: max difference 3.88936. ONNX Runtime matched eager within
3.1e-6, while a minimal exported pair-ID graph showed 8,158/8,192 OpenVINO
pair-ID mismatches and zero ONNX Runtime mismatches. This isolates an integer
arithmetic lowering problem, not model quality. The preflight-a failure and
diagnosis are retained unchanged. Before any training or validation score,
the hash expression was rewritten using the exact modular identities
`1315423911 mod 8192 = 1703` and `2654435761 mod 8192 = 6577`, with safe
32-bit intermediates (<17 million). This preserves **every possible pair
bucket**, not merely the sampled inputs. A unit test compares the old and
new expression exactly on 32,768 positions. A new preflight-b run must
repeat every original gate; no threshold or architecture parameter changed.

## Repaired preflight-b outcome, before training

The new one-off Windows task completed with `admit_matched_training_pilot=true`.
The maximum OpenVINO/eager hidden error was `3.69549e-6`; the random-weight
feature graph was 41,229,894 bytes and conservative graph-plus-unchanged-
assets projection was 65,497,409 bytes, below 67,108,864. Eight interleaved
four-thread calls gave candidate/reference feature median ratio `0.983568`.
One batch-32 BF16 synthetic optimizer step completed with 5,551,885,824
allocated and 5,901,385,728 reserved GPU bytes, below the 8 GiB GPU total.
All inputs were synthetic; no train/validation/test file was opened by the
preflight. These numbers qualify only the planned matched training pilot,
not a final CPU/RAM pass or a new BPB. Raw evidence is in
[`../results/stage204-evidence/preflight-b.json`](../results/stage204-evidence/preflight-b.json).
