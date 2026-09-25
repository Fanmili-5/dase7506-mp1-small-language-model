# Stage172 Linux FP32 feature-backend preflight (fixed before outcomes)

The unchanged Stage143 predictor scores 1.39968618 BPB on complete Linux
validation, but its one-thread Linux resource check takes 5.50255 times the
same-host baseline, above the course limit of five. Stage171 attributed about
84% of sampled inference time to its frozen FP32 OpenVINO feature graph.

This experiment compares that **same ONNX graph and validation inputs** with
OpenVINO 2026.4.0 and ONNX Runtime 1.26.0 CPUExecutionProvider, each using one
thread. Two passes of eight fixed batches alternate backend order. The graph,
weights, training data, evaluator and score selection remain unchanged.

The predeclared gate for building a full ONNX Runtime candidate is: finite
FP32 hidden states of shape `(batch, 256, 288)`, maximum absolute difference
from OpenVINO at most `1e-3`, and at least **12% feature-time reduction** in
the median paired pass. This 12% threshold is intentionally above the roughly
11% feature reduction estimated to cross the observed full-runtime limit;
the component-to-full extrapolation is uncertain. Passing this preflight
would authorize only a new full-validation candidate and fresh Linux/Windows
resource checks; it would not establish course eligibility. Failing it means
do not replace Stage143 with this backend.

No test split is scored in this preflight. Keep the existing Stage143
checkpoint and implementation unchanged.
