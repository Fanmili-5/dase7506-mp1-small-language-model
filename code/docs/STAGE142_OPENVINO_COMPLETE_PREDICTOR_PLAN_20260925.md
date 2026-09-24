# Stage142: exact Stage105 prediction with OpenVINO FP32 features

Stage141 proved that the frozen Stage92 feature graph is 42.37% faster on
the Windows scoring host under explicit four-thread FP32 OpenVINO, with
maximum hidden error below `8e-6`. Stage105 has the same Stage92 neural
ancestry and is already **1.399686 BPB**, but misses 5x CPU. The next
test must evaluate its *complete* predictions, not assume feature speed
transfers to the gated neural/count mixture.

Pin the Stage105 exported checkpoint and Stage126b feature graph by SHA.
Load the unchanged Stage105 head, prefix-copy attention, order-six count
tables and trained four-feature gate. Replace only its neural `features`
call with the OpenVINO CPU graph at explicit `f32`, four inference threads,
one stream, no pinning. On two validation input batches, compare complete
FP32 output probabilities against the untouched model: maximum probability
error <=3e-6, log-probability error <=3e-4 and log-normalization error
<=1e-5. Then run the unchanged evaluator's complete validation scoring
function for both in the same process. Require exactly 376,599 targets,
the Stage105 reference within 2e-6 for eager, OpenVINO BPB within 2e-5
of eager and still below 1.4, and a >=20% reduction in measured complete
score time before attempting a compact deployment bundle.

This temporary in-memory substitution retains duplicate PyTorch backbone
weights alongside the ONNX graph, so it is deliberately *not* an eligible
64MiB checkpoint. A passing result only admits Stage143: replace the
duplicated backbone with an ONNX graph asset, retain only head/count/gate
weights, and run clean-load official validation plus three alternating
fresh-process CPU/RAM/asset measurements. Test is not scored.

## Complete validation result

The source-pinned Stage105 model with only its `features` method temporarily
replaced reproduced the complete 376,599-target validation score at
**1.399686162042 BPB** versus **1.399686163172 BPB** for untouched eager.
On the two input batches, maximum full-distribution probability error was
`2.62e-6`, log-probability error `1.25e-4`, and log-normalization error
`2.99e-6`; all prespecified guards passed. The unchanged evaluator's
complete scoring function took **86.7310 s** with OpenVINO and **126.0794 s**
eager, relative time **0.687908** (31.21% faster). OpenVINO reported FP32,
four threads, one stream and no CPU pinning. The complete record is at
`../results/stage142-evidence/validation-cpu-fp32.json`.

The quality/20%-speed gate passes. This in-memory pilot still duplicates
the PyTorch feature weights and is **not** asset-qualified. Proceed to
Stage143's single-copy bundle and fresh-process resource measurements.
No test scoring occurred.
