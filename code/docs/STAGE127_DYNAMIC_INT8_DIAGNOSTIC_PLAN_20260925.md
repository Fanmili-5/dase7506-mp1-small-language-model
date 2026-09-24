# Stage127: dynamic INT8 matrix-multiply diagnostic, not a submission candidate

Stage105 profiling assigns most CPU self-time to dense linear layers; exact
TorchScript and ONNX Runtime feature probes did not speed them up. Test whether
CPU dynamic INT8 packing can reduce feature latency on the actual Windows
scoring host. This is deliberately a **diagnostic** because the course asks
for FP32 ranked evaluation. An internally quantized model must not be claimed
rule-compliant without an explicit course interpretation; no checkpoint export,
full validation score, or test score is authorized by this plan.

Freeze the Stage92 neural checkpoint and two input-only 32x256 validation
batches. Compare (a) eager FP32, (b) only eight SwiGLU FFNs converted to
dynamic INT8, and (c) all linear layers converted to dynamic INT8. Run each
variant warm twice and time eight forward feature passes on the first batch,
with four CPU threads. Check finiteness and maximum/relative hidden-feature
error on both input batches. Record PyTorch and CPU backend versions and
source/checkpoint hashes. No labels, target probabilities, or selection score
are used. A variant warrants any further investigation only if median feature
time is at least 12% faster, finite, and relative RMS error is below 5%; even
then rule clarification is required before adopting INT8 for submission.

This gate is diagnostic, not a claim that 12% feature speed implies 5x full
evaluation, nor that feature error bounds BPB. The qualified Stage85 candidate
remains unchanged.
