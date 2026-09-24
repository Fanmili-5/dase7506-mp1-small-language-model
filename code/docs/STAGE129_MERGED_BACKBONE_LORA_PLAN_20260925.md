# Stage129: merged backbone-LoRA distillation of the heterogeneous teacher

The fixed Stage91 two-architecture neural teacher scores 1.38320 BPB, but its
two full networks exceed deployment resources. Full-parameter Stage92
distillation reaches 1.40171 with five-order MKN after calibration; a second
teacher-only continuation (Stage109) regresses. Output-only LoRA (Stage96/97)
also regresses. The remaining testable hypothesis is that *distributed*
low-rank corrections across the backbone can transfer useful teacher behavior
while preventing full-weight drift. The corrections can be exactly merged
into existing FP32 linear weights for inference: no adapter operation remains.

Initialize the Stage92 averaged neural exactly. Freeze all original tensors.
Attach rank-8, zero-output LoRA to the four internal linear projections in
each of its eight blocks (attention or convolution input/projection plus the
two SwiGLU projections), excluding tied vocabulary head and copy head. Train
only the 32 adapters on supplied training prefixes with the same fixed
Stage71/Stage76 teacher and 0.75 teacher cross-entropy + 0.25 hard-label NLL.
One run uses seed 129017, batch 24, 1,800 updates, AdamW peak LR 0.002,
cosine decay, and weight decay 0.01. Do not alter architecture, tokenizer,
split, teacher, or count statistics. Monitor fixed Stage94 calibration plus
five-order MKN weight 0.0625 at steps 0/300/.../1800. Save merged, ordinary
neural checkpoints at 900/1200/1500/1800 and prespecify their uniform
parameter average. The old Stage92 predictor must reproduce 1.401708 BPB at
step 0. A one-batch preflight must prove zero-delta equivalence, finite adapter
gradients, and <=3e-5 maximum probability discrepancy after merging.

The averaged merged candidate must reach **<1.4 complete-validation BPB**
before any checkpoint export into the fused submission graph or CPU resource
audit. Then independently reproduce it with the unchanged scorer and verify
three-run CPU <=5x baseline, peak RSS <=4GiB, and assets <=64MiB. If the
quality gate fails, archive the result without resource or test scoring.

This is a *single-model architecture-preserving training intervention*, not
seed selection. Its strongest objection is that Stage92 may already sit at
the capacity/teacher-transfer ceiling; the fixed trajectory and step-0
control expose that failure directly. Stage85 stays the qualified fallback.

## First preflight correction

The first invocation failed before it created a run directory or performed an
optimizer step. Its zero-delta and initial merge smoke tests passed, but the
subsequent full-validation control raised `AttributeError` because removing
parametrizations from a deep copy interfered with the live model's shared
dynamic `ParametrizedLinear` class. The failure is preserved in
`../results/stage129-evidence/preflight-failure.json`. The corrected merge
builds a fresh ordinary model and copies each effective adapted weight into
it, leaving the training model untouched. The regression test now evaluates
the original *again after merging* to guard this exact failure. Training
settings and gate thresholds are unchanged.
