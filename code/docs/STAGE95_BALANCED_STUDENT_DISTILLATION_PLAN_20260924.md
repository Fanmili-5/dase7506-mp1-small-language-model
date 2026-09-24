# Stage95: balanced-student heterogeneous distillation

Stage92 proves that the heterogeneous teacher transfers useful information into
the 8x288 Stage71 graph, but the gain plateaus above 1.4. Stage95 tests a
capacity/allocation explanation: initialize a student from the resource-admitted
8x300 Stage76 balanced graph, add only a zero-initialized vocabulary bias, and
distill the same fixed 0.55/0.45 Stage71/Stage76 teacher.

The objective remains the successful Stage92 recipe: 0.75 full-distribution
teacher cross-entropy plus 0.25 training next-token NLL. One fixed seed-95017
trajectory uses batch 24, 2,400 updates, peak learning rate 1e-5, and averages
updates 1200/1500/1800/2100/2400. The old Stage79 calibration and MKN weight
0.075 are used only for monitoring; a positive student receives its own scalar
rescan. The student still has one deployable model graph and Stage72 established
resource headroom for this architecture. Test remains untouched.

## Result

The mismatched inherited calibration started at 1.4206787405 BPB and improved
monotonically to 1.4074187100 by update 2400. The fixed average scored
1.4076731598 BPB, far behind Stage92. Wider balanced capacity is not the active
bottleneck; the branch is rejected without qualification. Test was not scored.
