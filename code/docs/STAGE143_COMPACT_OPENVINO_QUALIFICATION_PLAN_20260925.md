# Stage143: single-copy OpenVINO Stage105 checkpoint qualification

Stage142 proved that replacing only the Stage105 feature call preserves its
complete 1.399686-BPB validation score and reduces complete scoring time
by 31.21% on the Windows host. The temporary model kept duplicate PyTorch
feature weights, so it cannot satisfy the 64MiB asset rule. Stage143
packages one frozen 31,805,041-byte ONNX feature graph plus a PyTorch
checkpoint containing only the vocabulary/copy head, MKN order-six count
tables, and trained dynamic gate. The same graph/weights/data are used;
no new training or validation-selected coefficients are introduced.

The `student_stage143_openvino_singlepass` model must implement the original
PyTorch scorer interface. OpenVINO CPU is forced to FP32, the requested
PyTorch thread count (four for qualification), one stream and no CPU
pinning. The graph path and SHA are code-pinned; no inference-time network,
future-token information, or cross-window state is allowed. Its own model
state must not include any of the old Transformer feature weights. Exact
smoke comparisons against the Stage105 source must pass maximum probability
error 3e-6, log-probability error 3e-4 and normalization error 1e-5 on two
deterministic input batches; causality and independent-row checks follow.

Export only after source-checkpoint and graph SHA checks. Use the unchanged
`evaluate.py --split validation --device cpu --precision fp32 --threads 4`
on the compact checkpoint and require complete coverage, BPB <1.4 and
agreement within 2e-5 of Stage105. Then run three alternating fresh-process
baseline/candidate measurements via `benchmark_cpu.py` with four threads.
Qualification requires median candidate CPU time <=5x median baseline,
peak process RAM <=4GiB and the sum of the compact checkpoint, ONNX graph,
own inference source, tokenizer and installation manifest <=64MiB. Record
all hashes and individual times. This is local Windows qualification;
clean-extract reproduction and release freeze are separate subsequent gates.
The test split remains unscored.

The OpenVINO dependency is a normal separately installed runtime, just as
PyTorch is; neither third-party wheel is counted as a student-authored
inference asset. The bundle must pin its version and install instructions.
The strongest objection is platform-dependent CPU performance. Passing on
Windows is necessary local evidence but does not guarantee the course Xeon
ratio; the final report must distinguish local qualification from remote
instructor verification.

## Observed Windows qualification (25 September 2026)

The exact compact checkpoint passed the unchanged full-validation scorer on
all 376,599 targets: **1.399686162042141 BPB**. Three alternating fresh-process
four-thread CPU runs gave baseline times 23.7861255, 23.8256361 and
23.3135692 seconds, and candidate times 86.8676388, 86.0511059 and
85.9281289 seconds. The ratio of medians was **3.617701668x**, below 5x.
Maximum candidate peak working set was **2,176,729,088 bytes**, below 4 GiB;
conservatively counted inference assets were **55,810,412 bytes**, below
64 MiB. The compact checkpoint is 23,824,895 bytes (SHA-256
`256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3`);
the ONNX graph is 31,805,041 bytes (SHA-256
`5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4`).

The resource probe's `parameters=629025` counts only live PyTorch parameters,
not weights stored in the ONNX graph. It must **not** be presented as the
effective full-model parameter count. The bytes above include both the graph
and checkpoint. Source, smoke, scorer and individual resource-run evidence are
under `code/results/stage143-evidence/`. This is local Windows evidence, not a
course-server guarantee or a test result.

## Clean-extract reproduction

An archive of commit `384fadf` was extracted into a fresh Windows directory,
without copying the original experiment run directory. With the existing
Python 3.12 environment containing the pinned dependencies, the repository's
unchanged `evaluate.py` scored the bundled checkpoint on full CPU FP32
validation at **1.399686162042141 BPB** over 376,599 targets. The checkpoint,
implementation, evaluator and tokenizer SHA-256 values matched the primary
score record. The independent output is
`code/results/stage143-evidence/clean-extract-validation.json`.
