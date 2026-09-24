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

## Result

On the fixed Windows host with PyTorch 2.7.1 and four CPU threads, the eight-run
median feature times were 2.083780 s FP32, 2.021619 s with FFNs in dynamic
INT8, and 2.005327 s with all linear layers in dynamic INT8. Thus the
quantized variants saved only 2.98% and 3.77%, respectively, versus the
predeclared 12% gate. Their hidden-feature relative RMS errors were 15.97–16.23%
and 17.46–17.90% on the two input-only batches, versus the 5% gate. Both
variants fail both tests. No complete validation, export, resource audit, or
test scoring was done. This also avoids needing a course interpretation of
internal INT8 arithmetic. Raw evidence is in
`../results/stage127-evidence/result.json`.
