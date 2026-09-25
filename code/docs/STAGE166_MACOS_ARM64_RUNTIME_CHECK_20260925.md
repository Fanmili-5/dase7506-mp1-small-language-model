# Stage166: independent macOS ARM64 runtime check (not a score)

On 25 September 2026, the Stage143 tracked checkpoint and ONNX graph were
checked on the local Apple Silicon Mac as an additional portability probe.
`scripts/verify_fixed_files.py` passed. The existing Python 3.12.13 virtual
environment had PyTorch 2.7.1; the separately pinned `openvino==2026.4.0`
installed successfully. The validation command was:

```text
python evaluate.py --checkpoint checkpoints/stage143-openvino-order6.pt \
  --device cpu --precision fp32 --threads 4 --split validation \
  --output results/stage143-macos-arm64-validation-20260925.json
```

It exited with code 134 before scoring any target, with
`bad err=11 in Xbyak::Error` and `Xbyak_aarch64::Error: internal error`.
A smaller probe confirmed that importing OpenVINO and constructing
`ov.Core()` succeeded, but querying `core.available_devices` caused the same
abort. The failure therefore occurs before loading or compiling the Stage143
feature graph, and cannot be attributed to its model arithmetic. Whether
the cause is this sandboxed Mac environment or an OpenVINO ARM64 runtime
issue remains unknown. One-thread graph-compilation probing also aborted
while discovering the CPU device.

There is **no macOS validation BPB or macOS resource qualification** from
this check. It does not change the independently reproduced Windows x86-64
CPU FP32 validation score and Windows resource measurements, nor does it
establish portability to the instructor's Linux Xeon. The final submission
must state the tested Windows environment and the pinned runtime rather than
claiming universal CPU support. No test split was accessed.
