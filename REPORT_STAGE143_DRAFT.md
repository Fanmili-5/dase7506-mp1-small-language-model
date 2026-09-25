# DASE7506 Project 1 — Stage143 report draft

**Draft only; do not submit.** Stage143 has been selected on validation but
has **not** been frozen or scored on test. The repository's `REPORT.pdf` is a
historical report for an earlier Stage10 model, not this candidate. Replace
this draft's pending fields and render/review a final PDF of at most 10 pages
only after a matching checkpoint/source freeze and one full-test score.

## 1. Abstract

We train a small causal language model from random initialization using only
the supplied WikiText-2 training text and fixed 2,048-token BPE tokenizer.
The current development model combines an eight-block alternating-attention/
causal-convolution Transformer, within-window prefix copy, and train-only
order-six modified Kneser–Ney (MKN) statistics through a causal gate.
OpenVINO evaluates its frozen neural features on CPU while the compact
PyTorch head, copy mechanism, count tables and gate produce one normalized
2,048-way distribution per position. The latest Windows CPU FP32 *validation*
score is **1.399686162 BPB** over 376,599 next-token targets. Three
fresh-process local CPU comparisons yield **3.617702x** baseline time,
**2,176,729,088 bytes** maximum peak RSS and **55,810,412 bytes** of
conservatively counted inference assets. Full-test BPB: **PENDING FREEZE**.

## 2. Task, data and evaluation

The [assignment guide](GUIDE.md) ranks complete-test bits per raw UTF-8 byte
under a fixed scorer, independent 256-token causal windows, CPU FP32 scoring,
at most 5x baseline CPU time, at most 4 GiB peak RAM and at most 64 MiB of
uncompressed inference assets. Training may use CPU/GPU, but only the
supplied training text; validation is for development and test must wait until
the method is frozen. No external text or pretrained weights were used in
the experiments reported here. The model does not retain state across
evaluation windows or call a network during inference.

The original course baseline measured **2.102014912 full-test BPB** in the
earlier frozen comparison. A previously frozen Stage10 model produced
**1.605136467 full-test BPB**, but that is a different checkpoint and must
not be presented as Stage143's test score. The current candidate's complete
validation BPB is reported above;
test comparison and relative improvement remain pending.

## 3. Model and training mechanism

The neural ancestor has width 288, eight blocks, eight attention heads and
context 256. Blocks 2/4/6/8 use kernel-seven gated causal depthwise
convolution, alternating with global causal self-attention. The model uses
RoPE, RMSNorm, SwiGLU, tied token/output embeddings and a within-window
prefix-copy distribution. The fixed [Stage54 configuration](code/configs/stage54_hybrid_conv_rdrop.json)
records dropout 0.1, embedding-row dropout 0.1, deep-supervision layers 4/6,
future-token offsets 2/3 and R-Drop coefficient 0.5. The key output
statistics are modified Kneser–Ney tables derived from the supplied train
text; a four-feature gate blends them with the neural/copy distribution.
The gate's learned statistics are train-derived, while its final feature
mask/scale was selected on validation and is disclosed as such.

This is a multi-stage lineage, not one 7,200-step run: Stage54 trains the
hybrid neural model; Stage56 continues its fixed average; later output-bias,
count-aware and distillation stages change the neural expert and gate before
the exact Stage105 predictor is packaged as Stage143. All checkpoints in the
lineage inherit their prior training cost. The detailed schedules, seeds,
target presentations and negative screens are retained in `code/docs/`,
`code/results/` and the code scripts. A hash-linked
[accepted-lineage audit](code/results/stage143-evidence/lineage-cost.json)
totals **255,225,110 documented primary training-target presentations** and
**7,754.94 seconds of recorded training time** across Stages54/56/61/63/
65/67/71/92. R-Drop uses two stochastic passes for many of these targets;
the count is not FLOPs or unique corpus positions. This is a **lower bound on
the accepted neural lineage**, excluding train-only count construction,
rejected/failed searches, validation, export and runtime audits. A complete
deduplicated project-wide search-cost ledger is still pending.

Stage143 changes only inference packaging relative to Stage105: it stores
one 31,805,041-byte ONNX feature graph and a 23,824,895-byte checkpoint
containing the head, copy, MKN and gate tensors, avoiding duplicate backbone
weights. There is no new training, coefficient fitting or validation selection
in the OpenVINO export. The model still returns finite, normalized causal
log-probabilities for the unchanged scorer.

## 4. Equal-target controls and ablations

The following completed runs use seed 17, batch 32, **7,200 updates** and
**58,982,400 primary next-token presentations**; the same sampled-window
sequence, optimizer settings and five-checkpoint averaging rule make the
paired comparisons interpretable. The optimization trajectories differ by
design. R-Drop processes each primary position twice stochastically;
this extra training compute is not a new independent target count.

| Model | Complete CPU FP32 validation BPB | Controlled question |
| --- | ---: | --- |
| Stage26, ordinary objective | 1.464993908 | Reference for training-only R-Drop |
| Stage47, same architecture + R-Drop | 1.450613003 | R-Drop gain 0.014381 BPB, no inference change |
| Stage54, R-Drop + alternating conv/attention, width 288 | 1.428594045 | Matched Stage47 comparison; gain 0.022019 BPB |

Stage47 versus Stage26 isolates the training objective; Stage54 versus
Stage47 tests the **combined** backbone reallocation (convolution placement
and width/FFN capacity), not the effect of convolution alone. These controls
do not share the later continuation/distillation budgets of Stage143 and
must not be presented as an equal-budget Stage143-versus-baseline comparison.
Source records: [Stage47](code/docs/STAGE47_RDROP_PLAN_20260922.md),
[Stage54](code/docs/STAGE54_HYBRID_CONV_RDROP_PLAN_20260923.md).

The later, separately budgeted path improves the development predictor:
Stage54 neural-only 1.428594045; Stage56 continued neural 1.420423862;
Stage71 neural+MKN 1.406959541; Stage85 resource-qualified calibrated
mixture 1.403024133; the Stage103/105 gated model 1.399686163; Stage143
packaging 1.399686162. These numbers are **validation model-selection
evidence**, not matched-target ablations or independent replications of the
final score. The Stage143 score still misses the aspirational 1.35 target.

## 5. Resource and reproducibility audit

The exact Stage143 checkpoint passed the unchanged evaluator on full Windows
CPU FP32 validation. In three alternating fresh-process comparisons, baseline
times were 23.7861/23.8256/23.3136 s and candidate times were
86.8676/86.0511/85.9281 s. The ratio of medians was 3.617702x. The maximum
candidate peak working set was 2,176,729,088 bytes; conservative assets were
55,810,412 bytes. These pass the three limits on the measured Windows host,
but do not guarantee the instructor's CPU timing. The [Stage143 qualification
record](code/docs/STAGE143_COMPACT_OPENVINO_QUALIFICATION_PLAN_20260925.md)
contains exact hashes and individual measurements. A fresh directory
extraction reproduced 1.399686162 BPB with the tracked assets.

| Frozen component for prospective submission | SHA-256 |
| --- | --- |
| `code/checkpoints/stage143-openvino-order6.pt` | `256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3` |
| `code/inference_assets/stage143-stage92-features.onnx` | `5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4` |

Reproduction from `code/`, in an environment installed per
[`code/README.md`](code/README.md):

```text
python scripts/verify_fixed_files.py
python -m unittest discover -s tests -v
python evaluate.py --checkpoint checkpoints/stage143-openvino-order6.pt --device cpu --precision fp32 --threads 4 --split validation --output results/stage143-reproduction.json
```

After the final freeze, use the same exact checkpoint/source bundle with
`--split test` once for the ranked score. **Do not run this command yet.**
The final code link must be immutable and must match the complete checkpoint
bundle, ONNX graph, dependencies and report. **PENDING FINAL HASH/LINK AUDIT.**

## 6. Negative results, limitations and AI disclosure

Stage145/147/150 changed attention allocation, width and depth but missed
their predeclared continuation gates. Stage155's larger neural model reached
1.409877 validation BPB and did not replace Stage143. Stages157/158
distilled an over-budget complementary teacher only to about 1.403 BPB.
Stage160's context-conditioned spelling residual, Stage161's
medium-frequency loss emphasis, Stage162's impossible sparse two-model
oracle, and Stage163's compact train-only nearest-neighbor expert also
failed their predeclared quality gates. These negative results constrain the
chosen method; they are not test scores. Exact details are in the linked
stage plans and evidence files, including target-only diagnostics that must
never be used as inference-time gates.

The validation score reflects many sequential decisions on one development
split and may be optimistic. The course CPU may differ from the Windows
laptop. The resource audit excludes third-party runtime wheel sizes but
counts every student-authored inference asset. The current candidate has
not yet been tested or submitted; exact final accuracy and ranking are
unknown. A student must inspect and understand the code and all claims.

OpenAI Codex provided substantive assistance with assignment analysis,
model/experiment design, implementation, testing, Windows execution,
diagnostics, documentation and this report draft. Earlier conceptual work
reused RoPE, SwiGLU, RMSNorm, R-Drop, modified Kneser–Ney, prefix-copy and
kNN-LM ideas. The implemented concepts' original publications are listed
below; citing an idea does not imply that the paper's implementation or
pretrained weights were copied. The [detailed disclosure](code/docs/AI_ASSISTANCE.md)
and experiment records distinguish reused ideas, AI contributions and
original project-specific implementations. Passing automated checks is
not a substitute for student review.

## 7. Method references

- Su et al. (2021), [RoFormer: Enhanced Transformer with Rotary Position
  Embedding](https://arxiv.org/abs/2104.09864) — RoPE.
- Shazeer (2020), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
  — SwiGLU.
- Zhang and Sennrich (2019), [Root Mean Square Layer
  Normalization](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
  — RMSNorm.
- Liang et al. (2021), [R-Drop: Regularized Dropout for Neural
  Networks](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)
  — stochastic consistency objective.
- Chen and Goodman (1999), [An Empirical Study of Smoothing Techniques for
  Language Modeling](https://www.sciencedirect.com/science/article/pii/S0885230899901286)
  — modified Kneser–Ney smoothing.
- Merity et al. (2016), [Pointer Sentinel Mixture
  Models](https://arxiv.org/abs/1609.07843) — within-context copy concept.

The later compact kNN-LM pilot was rejected and is documented in the
[Stage163 record](code/docs/STAGE163_COMPACT_KNN_LM_PILOT_20260925.md), which
cites its originating publication. It is not part of the Stage143 predictor.

## 8. Pending finalization checklist

- Decide and record the frozen method using validation only; stop all
  development decisions before reading Stage143 test results.
- Audit cumulative training/search cost and verify all report claims/hashes.
- Run one matching full-test CPU FP32 score after freeze and fill the score,
  target count, raw byte count and checkpoint SHA here.
- Render this report to PDF, visually inspect every page, and verify <=10 pages.
- Submit student ID and full-test BPB through the course website **before
  29 September 2026 (UTC+8)**; complete the generated GitHub issue.
- By the end of 30 September, submit immutable matching code and checkpoint
  links plus the final report; retain public-review reproduction evidence.
