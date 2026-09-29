# Stage126: ONNX Runtime CPU feature-extractor pilot

Stage124's TorchScript path reproduced the neural features but was 1.14%
slower than eager PyTorch. That rejects this compiler path, not all CPU graph
backends. ONNX Runtime and ONNX are already present in the Windows virtual
environment, so Stage126 tests an exported, fixed Stage92 feature extractor
without changing weights, dataset, tokenizer, evaluator, or training.

Use the exact Stage92 checkpoint, export only `neural.features` for independent
256-token windows with a dynamic batch axis, and compare two 32-window batches
plus one single-window input against eager CPU FP32. Time six warmed 32-window
forwards, including NumPy/Torch boundary conversion, for both engines at four
CPU threads. Require maximum hidden-state error at most 3e-4 and at least 12%
median speed gain before attempting a full Stage105 predictor integration.
The 12% gate covers its roughly 10% CPU overage with a small integration
margin. If export, numerical equivalence, or speed fails, preserve the
negative pilot and do not promote. Any positive pilot still needs a single
checkpoint/graph bundle under 64 MiB and formal three-repeat CPU/RAM/asset
qualification; a model file with duplicate weights is not acceptable.
Validation inputs are read without labels and test is not scored.

## Exporter compatibility correction

The first source-pinned attempt stopped before timing: PyTorch's installed
ONNX exporter rejected `aten::rms_norm` for opset 17. This is an exporter
failure, not a measured speed result. Stage126b rewrites only the frozen
RMSNorm expression into square/mean/reciprocal-square-root/multiply operations
in a *copy* of the neural model. It first checks the portable copy against the
untouched eager model on two input batches, then applies the original ONNX
equivalence and 12% speed gates. The original attempt's error is recorded in
`../results/stage126-evidence/export-failure.json`; the rerun uses a new
output directory and source hash.

## Stage126b result

All 17 rewritten RMSNorm layers exactly matched the original PyTorch copy
on the pre-export checks. The 31,805,041-byte ONNX graph then matched eager
hidden states within 5.97e-6 across two full batches and one single-window
input. Six warmed CPU FP32 medians were **2.165081 s eager** and
**2.194512 s ONNX Runtime**, so ORT was **1.36% slower**, not at least 12%
faster. The graph is only a pilot artifact, not a deployable model; no
checkpoint or full scorer audit follows. This rejects ONNX as a speed-only
solution for the unchanged Stage92 feature extractor on this Windows machine.
The raw result is in `../results/stage126-evidence/result.json`; no test
scoring occurred.
