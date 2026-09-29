# Stage210: bounded FP32 inference-backend feasibility screen

Stage208's identical lexical tree formula took 3.103434 s/32x256 in naive
PyTorch. Stage209's level-wise propagation cut it to 0.642773 s on the
Windows four-thread CPU, but missed the fixed <=0.50 s screen. Stage209 was
not trained and has no validation/test score. The Stage209 GPU update,
normalization, causal isolation and projected asset checks passed.

Stage210 changes **no probability formula or training method**. Export only
the Stage209 lexical tree head to an FP32 ONNX graph and compile it with
OpenVINO CPU, four threads, one stream and disabled CPU pinning. Do not
quantize, change precision, topology, tree or tokenizer. With the same
synthetic 32x256 hidden states, require maximum eager-versus-OpenVINO
log-probability error <=3e-4 for batch 1 and 32; normalization <=1e-5;
eight warmed median head time <=0.40 s (stricter than the earlier 0.50 s
screen to leave full-predictor timing margin); and a conservative projection
of Stage143 assets plus the head graph and 0.5 MiB source reserve <=64 MiB.
The random graph is not a trained inference bundle or full CPU qualification.

Only if this backend gate passes may the fixed Stage209 model enter the
previously preregistered seed-17, 2,400-step same-target pilot. The
early-quality advancement gate remains a >=0.030-BPB gain versus Stage54's
1.5199503686120217. Any quality winner must subsequently export its own
trained FP32 graph, meet full CPU-FP32 validation <1.35 and all exact
5x/4-GiB/64-MiB resource rules. No test split is read before a separate
method freeze. Failure here stops the lexical hierarchy route; no nearby
backend/precision sweep follows.
