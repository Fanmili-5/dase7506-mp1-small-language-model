# Stage120: stronger hard-label continuation of the full Transformer

Stage92 distilled a heterogeneous teacher into one Transformer but left the
static order-six mixture at 1.401288 BPB. Stage109's teacher-only continuation
regressed, so Stage120 tests the opposite mechanism: equal weight on the same
frozen 0.55/0.45 Stage71/Stage76 teacher and real next-token training labels.
The student starts from the exact Stage92 average, retains all eight blocks,
and uses the fixed Stage94 inference calibration in its training loss and
validation monitor. The count expert is frozen and serves only the fixed
0.0625 validation mixture. Seed 120017, batch 24, 3,600 updates, peak LR
8e-6, AdamW and a four-checkpoint late average are fixed in advance.

This is a 22.12-million-new-target training test of whether stronger hard
supervision can improve the same deployment graph without gate overhead.
Only supplied train prefixes generate gradients. The complete order-six
validation monitor includes the unchanged Stage92 reference; no test scoring
occurs. An apparent BPB gain still requires exact fused export and the
three-repeat CPU/RAM/asset check before promotion.

## Observed result

Windows RTX 3070 Ti training completed all 3,600 updates and 22,118,400 new
targets. The fixed order-six validation monitor was best at step 0
(1.401288462 BPB); the endpoint was 1.401841357. The preregistered four-step
late parameter average (steps 1800/2400/3000/3600, SHA-256
`7b70fd562e159b8c95b0cb01bafec0231c2a84193ae19b1324530ac5ebdbbc6e`)
scored 1.401887300 BPB on the same validation windows. Thus stronger real-label
continuation did not improve Stage92. It is rejected without an export or
resource audit; no test evaluation was run. Raw metrics, checkpoint ancestry,
and the separate average score are in `../results/stage120-evidence/`.
