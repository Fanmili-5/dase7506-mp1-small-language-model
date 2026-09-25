# Stage170: Linux x86-64 resource recheck for the same Stage143 predictor

This is a portability/resource diagnostic, not an optimization, method
freeze, or test evaluation. Stage143's complete Windows CPU FP32 validation
is 1.399686162 BPB at 3.617702x baseline time, 2,176,729,088-byte peak
RSS and 55,810,412-byte inference assets. The separate Linux x86-64 CI
run reproduced the full-validation score to 1.7e-8 BPB but did not measure
a same-host baseline ratio or whole-process memory. The GitHub Actions
runner exposed only two logical CPUs; OpenVINO accepted one thread, so the
new same-host comparison must request **one** thread for both predictors.

The baseline checkpoint is the exact Windows resource-control file, now
copied into `../benchmark_controls/baseline-stage3-long-s17.pt`, SHA-256
`2e4b9e749796dc1d6d8a2e5aedc9482d961dcabd9740760e905f7ae117ba647d`.
It contains the unchanged course `model` architecture (vocab 2048, width
128, four heads, four blocks, context 256). Its longer training schedule
does not change the baseline scoring architecture; this is a time control,
not a new quality comparison or part of the final inference-asset bundle.
The Stage143 checkpoint, graph, evaluator,
tokenizer and all inference-source hashes remain unchanged.

On a fresh Ubuntu 24.04 x86-64 Actions runner, install the same pinned CPU
dependencies and verify fixed course files/tests. Run the unchanged
`benchmark_cpu.py` with three alternating fresh-process full-validation
scores per candidate, one FP32 thread, and no test access. Recompute the
median candidate/baseline time ratio, maximum candidate whole-process RSS,
and conservative uncompressed asset total. Pin the exact baseline and
Stage143 checkpoint/source hashes and require the known complete-validation
coverage (376,599 targets, 1,148,007 bytes) and score agreement within
2e-5 BPB. The local gate is CPU <=5x, RSS <=4 GiB, assets <=64 MiB.

If any gate fails, keep Stage143 as the Windows-qualified fallback but
report the Linux discrepancy explicitly; investigate runtime compatibility
before claiming portability. If all pass, report it as an **additional
same-host resource observation**, not proof of the instructor's hardware
result or of sub-1.35 performance. No architecture, weights, validation
selection or test scoring will change during this recheck.
