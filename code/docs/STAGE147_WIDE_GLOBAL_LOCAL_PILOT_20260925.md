# Stage147: capacity within the accepted OpenVINO envelope

The sub-1.35 target remains 0.049686 BPB below the resource-qualified
Stage143 validation candidate. Stage145's six-attention allocation improved
the matched 2,400-step Stage54 control by just 0.004699 BPB; the Stage146
exact train-suffix mixture added only 0.003037 BPB in a non-deployable
diagnostic. Neither justifies a long or implementation-heavy continuation.

Stage59's 8x320 local-heavy model and Stage54's 8x288 alternating model were
trained under the same R-Drop recipe and ended close in neural-only quality.
They confound width with global/local allocation. A single 8x320 model with
Stage54's four-attention/four-convolution allocation isolates whether more
representational capacity helps without sacrificing the stronger mixing
pattern. The older PyTorch CPU gate disallowed this configuration, but the
exact Stage143 OpenVINO FP32 path now has measured headroom. That is an
inference-feasibility premise, not evidence of a BPB gain.

First export a random-weight feature graph with the frozen training/inference
config, force four-thread OpenVINO FP32, and compare parity, graph size and
eight interleaved batch-32 timings with the Stage143 feature graph. Pass only
if hidden error <=3e-4, graph plus a conservatively reserved 25 MB compact
head/count bundle and 0.2 MB source <=64 MiB, and median feature time <=1.25x
Stage143. This is an input-only preflight, not a complete model resource gate.

If it passes, train exactly 2,400 steps on the same seed-17 R-Drop recipe,
sampled train windows and first 2,400 updates of the original 7,200-step
learning-rate schedule as Stage54, changing only width 288 to 320. Compare the
complete GPU FP32 step-2,400 validation score with Stage54's archived
1.519950369 BPB. Require at least **0.015 BPB** improvement before committing
to a full 7,200-step schedule. A full run would still need an exact train-only
MKN mixture, independent CPU FP32 validation and three-repeat CPU/RAM/assets
qualification. Validation selects; test remains untouched.

Strongest objection: extra width may mostly memorize the 3.6M-token train
corpus. Stage143's sampled-train/validation NLL gap was 0.636 nats/token, so
capacity could worsen generalization. The matched early curve is a cheap
falsifier, not a guarantee that a positive 2,400-step result survives to the
final mixture or achieves 1.35.

The Stage145 exploratory pilot shortened the learning-rate schedule to 2,400
updates, so its earlier 0.004699-BPB margin over Stage54 is not a strict
architecture-only comparison. Stage147 fixes this before quality is observed.

## Input-only resource preflight

The random-weight width-320 graph used **9,722,241** neural parameters and
occupied **38,909,889 bytes**. With a conservative 25 MB reserved for the
compact count/head checkpoint and 0.2 MB for inference source, projected
student-authored assets were **64,109,889 bytes**, below 64 MiB. Eight
interleaved Windows CPU FP32 feature calls gave medians 1.464850 seconds for
the candidate and 1.260027 seconds for Stage143 (ratio **1.16255x**).
Maximum hidden-feature error versus eager PyTorch was **4.22e-6**. All fixed
pilot gates pass; this does not measure complete predictor scoring time or
trained quality. Raw samples and hashes are in
`../results/stage147-evidence/preflight.json`.
