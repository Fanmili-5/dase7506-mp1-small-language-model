# Stage155: spend the inference budget on one stronger neural predictor

## Problem-first triage

The qualified Stage143 model scores 1.399686162 validation BPB; the target
is 0.049686 lower. The over-budget Stage91 two-neural ensemble scores
1.383202407 without counts and 1.381623389 with its tuned count expert, a
gain of about 0.00158 BPB for that expert in **that** mixture. This is not a
controlled ablation of Stage143's different order-six MKN. It does show that
the resource-heavy count component may be a poor use of the *next* candidate's
finite asset/CPU budget. Stage151 additionally found high loss on
medium-frequency targets absent from the current prefix, which exact
within-window counts cannot directly retrieve. The new question is whether a
larger **single** Transformer, without an MKN inference table, can make a
material generalization gain under the same final CPU/RAM/asset limits.

Failure analysis, decomposition, and the simplicity test narrow the ideas:

| Direction | Existing evidence / decision |
| --- | --- |
| More exact token suffixes | Stage146 target-only gain 0.0030; deprioritize |
| Short byte-suffix table | Stage152 gain 0.0007; deprioritize |
| Static count/gate calibration | Prior gains milliscale; deprioritize |
| Distant co-occurrence table | Stage149 worsened all nonzero weights; reject |
| Two independent neural experts | Stage91 quality 1.3832, but over budget |
| Shared-trunk two paths | Stage153 resource, Stage154 quality gates failed |
| Width only | Stage147 early gain 0.01135, below full-run gate |
| Depth only | Stage150 early gain 0.01151, below full-run gate |
| Frequency-aware training loss | Zero inference cost, but not yet tested; reserve |
| Trained character-conditioned output | Could address spelling generalization; implementation/CPU risk |
| Width **and** depth, no count inference | Test now: reallocates assets and CPU from weak count gain |

The Stage155 fixed architecture combines the Stage147 width 320 with the
Stage150 ten-layer alternating attention/convolution layout and the Stage54
R-Drop recipe. The output remains a fully normalized 2,048-way prefix-copy
distribution over the **fixed supplied tokenizer**. It reads only independent
causal 256-token windows and trains only on supplied train text. No count table
will be deployed. This is a materially different resource allocation, not a
claim that adding capacity guarantees better validation BPB; the 3.6M-token
training corpus and observed train/validation gap make overfitting a strong
objection.

## Fixed feasibility and quality gates

1. On Windows, export a random-weight FP32 OpenVINO feature graph; require
   max hidden error <=3e-4 and graph bytes +5,000,000 bytes reserved for the
   compact head +200,000 source bytes <=64 MiB. Eight interleaved four-thread
   feature calls must have median <=1.60x the frozen Stage143 graph. This is
   input-only and cannot establish official full-predictor eligibility.
2. Verify a single training update with effective batch 32 and context 256
   on the RTX 3070 Ti without out-of-memory. Try physical batch 32 first; if
   it cannot fit, use two physical batches of 16 with accumulated gradients.
   This choice is based only on training memory, not validation quality.
3. Only if both screens pass, run a seed-17 matched 2,400-update pilot using
   the first 2,400 learning rates of Stage54's 7,200-update schedule and the
   same effective batch 32 training sampler. Compare the complete GPU FP32
   validation endpoint to Stage54's 1.519950369 BPB. Require >=0.020 BPB
   gain before a full run; the stronger gate reflects combined training cost
   and the much larger 0.049686 gap to the desired score. If gradient
   accumulation is needed, disclose it as an imperfect control.
4. Even a passing pilot does not promote Stage155. Full training must beat
   Stage143 on complete CPU FP32 validation, then pass three independent
   CPU <=5x, RAM <=4 GiB and assets <=64 MiB repetitions and clean-extract
   reproduction. Test stays untouched until method freeze.

The current Stage143 checkpoint remains protected. The first action is only
the input-only feature and memory preflight; no Stage155 BPB is claimed.

## Feasibility result (2026-09-25)

The random-weight OpenVINO graph was 47,960,933 bytes, with 11,978,561
neural parameters. Its PyTorch/OpenVINO hidden-state error was `4.30e-6`,
the conservative graph + head/source reserve was 53,160,933 bytes, and the
eight-pair feature-time median was **1.500597x** Stage143, within the
prespecified 1.60x screening cap. A separate one-update synthetic GPU probe
completed at physical/effective batch 32 without accumulation, peaking at
5,142,695,936 allocated and 5,404,360,704 reserved CUDA bytes. Both screens
pass, authorizing the fixed 2,400-step quality pilot. They do not establish
trained quality, complete CPU time or final asset eligibility. Source hashes
and raw timings are in `../results/stage155-evidence/`.

## Matched pilot result (2026-09-25)

The scheduled Windows job completed all 2,400 seed-17 updates using physical
and effective batch 32. It presented 19,660,800 primary targets in 662.85
training seconds; the fixed-file checks and inherited hybrid R-Drop tests
passed. Complete GPU FP32 validation at step 2,400 was **1.497499861 BPB**
on 376,599 targets, versus Stage54's same-schedule **1.519950369 BPB**.
The improvement is **0.022450508 BPB**, exceeding the prespecified 0.020
full-run continuation threshold. The trajectory, source/checkpoint hashes,
console and task status are in `../results/stage155-evidence/`.

Decision: start a **fresh** seed-17 7,200-step run using the unchanged
Stage54 learning-rate trajectory and fixed last-five averaging window.
The 2,400-step pilot checkpoint is not treated as a final candidate. The
full run still must beat Stage143 in complete CPU FP32 validation and satisfy
all three official resource limits; no test score has been produced.

## Frozen inference qualification route (before full-run result)

First score the full-run endpoint and prespecified last-five (steps
6000/6300/6600/6900/7200) parameter average on complete validation, selecting
the lower BPB. If neither beats Stage143's 1.399686162, stop without
promotion. For a better checkpoint, export its *trained* ten-block feature
extractor once to an FP32 OpenVINO graph and store only the trained tied
vocabulary/copy head in the checkpoint. Hash-pin the graph in the compact
checkpoint; no train-only or validation cache is embedded. Require max
hidden error <=3e-4, two-batch full-distribution probability error <=3e-6,
log-probability error <=3e-4, normalization error <=1e-5, and causal/row
independence errors <=3e-5 against the same source model. Then score all
376,599 validation targets through the **actual compact CPU FP32 predictor**.
Three fresh baseline/candidate CPU timing and lifetime-RSS repetitions must
show <=5x, <=4 GiB; checkpoint + graph + inference sources must be <=64 MiB.
Only a fully passing result can replace Stage143. Test remains untouched.
