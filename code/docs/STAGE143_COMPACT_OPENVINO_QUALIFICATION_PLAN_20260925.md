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
