# Stage141: mechanism triage and FP32 CPU backend pilot

## Problem-first framing

The frozen Stage105 predictor already reaches **1.399686 BPB** on complete
validation, but its three-repeat CPU ratio is **5.512x**, about 9.3% above
the allowed time. No extra modeling benefit is needed if an exact FP32 CPU
implementation can save enough runtime. The 62.7% matrix-multiply share in
the Stage105 profile makes a different CPU graph backend a plausible first
test. This is not a claim that a backend change *will* help.

Two-sentence pitch: The sub-1.4 Transformer/count predictor cannot be
submitted because its four-thread CPU implementation is too slow. Run the
same frozen FP32 feature graph through a different CPU compiler, accepting
it only if the complete feature path is materially faster and predictions
remain close enough for exact full-score verification.

## Divergent candidate ledger

| Mechanism | Prior evidence / main objection | Decision |
| --- | --- | --- |
| OpenVINO CPU FP32 on the existing feature ONNX graph | ORT was exact but 1.36% slower; OpenVINO has a different CPU compiler. Must force FP32/four threads and package a reproducible dependency. | **Pilot now** |
| End-to-end oneDNN layout without per-block conversions | Stage134 isolated FFN packing was slower; a fully layout-preserving graph is technically expensive. | Reserve |
| FFN importance pruning plus train-only repair | Stage119 10% pruning lost 0.016 BPB before repair; Stage113 block repair plateaued. | Reserve |
| New seven-block student from scratch | Faster graph but Stage113 warm-start student stopped at 1.422 BPB. | Low priority |
| Distill teacher into balanced 8x300 student | Stage140 finished at 1.40893 under fixed Stage94 mixture. | Rejected for this recipe |
| Heterogeneous expert choice per window | Stage138 hindsight oracle itself was 1.40344 BPB. | Rejected |
| Two-model probability ensemble | Stage91 gives 1.38162 BPB but runs two large networks and exceeds budget. | Infeasible directly |
| Exact local successor cache | Stage132 target-only signal 1.39898; Stage133 direct complete predictor adds 9.48% CPU atop an overbudget parent. | Reserve optimized integration |
| Simpler train-fitted count gate | Stage117 no-margin gate remained over 1.4; Stage128 quadratic gate regressed. | Low priority |
| Reweight count probabilities by token frequency | Stage81 cross-fit gain only 0.000084 BPB. | Rejected |
| Recalibration / output bias alone | Stage67 bias gained 0.00023; late scalar scans unlikely to close resource gap. | Low priority |
| More Stage92 hard-label or mixture-aware continuation | Stage120/125 worsened full validation. | Rejected for tested objectives |
| TorchScript or ONNX Runtime feature graph | Stage124/126 exact but 1.14%/1.36% slower. | Rejected |
| Pretransposed FFN weight layout | Stage139 exact but <0.2% benefit. | Rejected |
| Dynamic INT8 inference | Stage127 feature errors and speed gate failed; FP32 rule ambiguity. | Rejected |

The three live mechanisms are OpenVINO FP32, a fully layout-preserving
oneDNN graph, and structural FFN compression with repair. OpenVINO is the
least invasive and tests the exact sub-1.4 candidate's dominant bottleneck.
The strongest objection is that ONNX Runtime failed the same speed test and
OpenVINO's default may silently use BF16 on the i7 host. The pilot explicitly
requests FP32, records the compiled property, caps inference to four threads,
and uses no CPU pinning. Intel's documentation confirms direct ONNX loading
and those precision/thread properties. See:
https://docs.openvino.ai/nightly/openvino-workflow/model-preparation/convert-model-onnx.html
and
https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html

## Fixed pilot contract

Use the existing SHA-pinned Stage126b FP32 ONNX *feature* graph generated
from the Stage92 neural checkpoint. No weights, tokenizer or evaluation code
change. On the Windows scoring host, load it through OpenVINO CPU and request
inference precision `f32`, four inference threads, one stream, and no
affinity/pinning. Compare against eager PyTorch on two 32x256 input-only
validation batches and one single-window input; maximum hidden absolute
error must be <=3e-4. Time eight warmed interleaved full-feature calls per
engine on the first batch, including the NumPy/Torch boundary. Proceed to
integrating the complete Stage105 predictor only if median feature time
is **at least 15% lower** than eager and the parity/FP32/thread checks pass.
This 15% feature gate accounts for the roughly 9.3% full-predictor CPU
shortfall plus integration overhead. An input-only pilot is never a score or
resource qualification. Complete official validation and three-repeat
CPU/RAM/asset checks remain mandatory. Test is not scored.

## Input-only pilot result

OpenVINO `2026.4.0` was installed into the existing Windows virtual
environment with dependencies unchanged (`--no-deps`). The exact Stage126b
ONNX graph compiled for `CPU` with reported inference precision `float32`,
four inference threads, one stream and CPU pinning disabled. Maximum hidden
errors on two full input batches and one single window were respectively
`7.15e-6`, `6.17e-6` and `4.41e-6`, below the `3e-4` gate. Eight interleaved
warmed FP32 full-feature medians were **2.172431 s** eager PyTorch and
**1.251868 s** OpenVINO, a relative time of **0.576252** (42.37% faster).
The graph hash, compiled properties, all timing samples and source hash are
in `../results/stage141-evidence/input-only-cpu-fp32.json`.

This passes the 15% *feature* gate, so Stage142 may integrate the same
feature graph with Stage105's unchanged head, copy and count/gate path.
It is not yet a full predictor score, asset bundle, or CPU/RAM qualification.
The OpenVINO package requirement and a single-copy graph asset must be
included if that later predictor succeeds. No test scoring occurred.
