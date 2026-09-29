# Stage121: improved static qualified fallback

Stage92 + Stage94 calibration + fixed five-order count weight 0.0625 reached
1.401708 validation BPB without a dynamic confidence gate. Stage121 folds
the calibration into existing final-norm, output-bias and copy-head affine
parameters, then exports the same fixed-weight fused-count path that was
resource-qualified in Stage85. It adds no trained tensor or inference
operation. A smoke test compares the exact Stage94 mixture against the folded
predictor; complete CPU FP32 validation must reproduce Stage114's fixed
mixture. Three alternating fresh-process baseline/candidate CPU measurements,
peak RSS, and a conservative uncompressed asset sum determine whether it can
replace Stage85 as the qualified development leader. No test scoring occurs.

## Observed result

The exact folded export reproduced 1.401707663 CPU FP32 validation BPB
(checkpoint SHA-256
`e1905b57d020766f72bea0c4ccdf214fcbd6d1ec43482cb9fe33d5e1313e0f9e`).
The three-repeat Windows resource audit measured baseline median 22.978547 s
and candidate median 118.875586 s, a **5.173329x** ratio. Peak RSS
2,040,385,536 bytes and conservative assets 48,567,046 bytes passed, but CPU
did not. Stage121 is **not resource-qualified**; Stage85 remains the latest
qualified development candidate. Raw validation, resource, and final records
are preserved under `../results/stage121-evidence/`. The discrepancy with
Stage85's 4.900108x audit shows that the narrow prior timing margin is not
robust to run-to-run conditions. No test scoring was performed.
