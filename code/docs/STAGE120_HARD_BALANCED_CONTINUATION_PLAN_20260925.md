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
