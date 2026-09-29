# Stage139: exact transposed-weight CPU GEMM pilot

The Stage105 profile attributes 62.7% of self CPU time to matrix products,
with the first SwiGLU projection alone at 686.9 ms of a 2.677-second
profiled batch. Stage134 oneDNN packing was slower because every FFN paid
conversion overhead. Test a simpler representation change: store each
frozen FFN weight already transposed and contiguous, and call FP32 `addmm`
on flattened token rows. This keeps the same mathematical affine map,
activation, network, count tables and training weights. It does not override
the official four-thread setting or use quantization.

Pin the Stage92 averaged neural checkpoint. On the Windows scoring host,
compare eager FP32 against pretransposing only the eight SwiGLU input
projections, only the eight output projections, and all sixteen. Use two
input-only validation batches for full-feature parity (maximum absolute
error at most 3e-5), then two warmups and eight interleaved timed full
feature forwards on the first batch at four threads. Advance only if a
variant is at least 8% faster in median feature latency and passes parity;
then implement a portable single-copy checkpoint and independently score
full validation plus three-repeat official CPU/RAM/asset gates. A pilot is
not a qualification. No validation labels, test scores, or new training.

## Observed result

All three transposed-weight variants matched the eager model's complete
features **exactly** (maximum absolute error 0 on both input batches), but
none improved speed materially. Eight interleaved four-thread CPU runs gave
median full-feature times of **2.087724 s** eager, **2.084634 s** input-only,
**2.095787 s** output-only and **2.090770 s** both. Relative times were
0.99852, 1.00386 and 1.00146, far short of the 0.92 gate. The current
linear path already handles transposed operands efficiently on this host.
No new inference module, full validation score, or resource audit follows.
Raw input-only evidence is in `../results/stage139-evidence/`.
