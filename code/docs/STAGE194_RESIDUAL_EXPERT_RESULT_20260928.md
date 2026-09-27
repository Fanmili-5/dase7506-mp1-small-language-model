# Stage194 synthetic CPU preflight: stopped before training

The frozen Stage143 checkpoint hash matched the protected reference. With
synthetic 256-token windows on the Windows CPU (FP32, four threads), the
zero-initialized residual matched Stage143 to `1.9073486328125e-06` maximum
log-probability deviation. A future-token perturbation changed no earlier
log-probability in the measured prefix, and the largest probability-mass error
was `1.1920928955078125e-06`.

The residual state occupied 3,267,456 bytes; projected uncompressed assets
including the declared source reserve were 59,340,012 bytes, below the
67,108,864-byte limit. The interleaved median candidate/base CPU ratio was
**1.2005743093487935**, just above the predeclared **1.20** screen. Therefore
`feasibility_gate_passed=false`; Stage194 was not trained or validated. This
one synthetic timing check is not an official CPU-resource measurement and
does not estimate BPB. Raw evidence is in
`code/results/stage194-residual-preflight-v1.json`.

No training, validation, or test data was opened by this preflight script.
