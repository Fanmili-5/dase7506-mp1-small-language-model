# Stage195 result: feasible but below the validation advancement gate

The synthetic Windows CPU preflight passed. Its zero-init parity error was
`1.9073486328125e-06`, the measured future-token prefix perturbation error
was zero, projected uncompressed assets were 58,003,692 bytes, and its
interleaved median candidate/base CPU ratio was 1.097101395. These are
feasibility screens, not official score/resource qualification.

The fixed 1,200-step, batch-8 Windows CUDA pilot used only the supplied train
text for gradients and scored only the complete validation split (376,599
targets; 1,148,007 UTF-8 bytes). Both arms began at approximately 1.39968619
BPB. The teacher-mixture arm's best full-validation BPB was **1.398492575**
at step 1,200, an improvement of only **0.001193587** over Stage143. The
hard-label arm never improved over initialization; its endpoint was
1.402028049. The predeclared absolute advancement requirement was a gain of
at least 0.015 BPB, and the teacher-effect requirement was 0.010 BPB. Neither
passed. The residual architecture is not a replacement candidate and was not
run on the test split.

Interpretation is narrow: this fixed frozen-feature residual recipe did not
recover the complementary benefit of a second full model. It does not prove
all shared-backbone experts impossible. The Stage143 inference checkpoint and
OpenVINO graph remain unchanged. Raw evidence is in
`code/results/stage195-feature-residual-preflight-v1.json` and
`code/results/stage195-feature-residual-pilot-v1.json`.
