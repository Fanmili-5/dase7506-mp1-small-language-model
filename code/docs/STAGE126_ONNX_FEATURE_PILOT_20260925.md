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
