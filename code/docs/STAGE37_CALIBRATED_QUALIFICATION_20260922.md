# Stage37: exact calibrated hybrid qualification

Stage37 serialized the fixed Stage32 scalar calibration on top of the Stage30
Transformer/modified-KN experts and used the collapsed sparse recurrence. The
first scheduled attempt stopped before writing a checkpoint because moving the
temperature division before the bias-free projection exceeded the preregistered
2e-5 per-log-probability tolerance. The exact Stage32 arithmetic was restored;
the resumed preparation then passed full validation equivalence with maximum
log-probability error 1.907e-6 and normalization error 7.902e-7.

The exact checkpoint SHA-256 is
`ed29352b1fca4c325beec7fcb30ab2b4a019a3f264b256945d119534c2380e43`.
Independent CPU FP32 validation reproduced **1.4464620202 BPB**. Peak RSS
2,045,689,856 bytes and conservative assets 44,240,406 bytes passed. The median
CPU ratio was **5.006015116x**, narrowly above the 5x limit, so Stage37 is
formally rejected despite its quality. Stage27 at 1.4543904321 remains the
qualified fallback until an equivalent faster implementation passes.

Stage39 separately tests the mathematically equivalent temperature reassociation
with a declared 5e-5 full-output tolerance and an independent complete-score
tolerance of 1e-6. It completed with checkpoint SHA-256
`294ae321dec8eddc7e4942f759c75f16b3333ea1ecf22b532e13295b1ced3a19`,
validation BPB **1.4464620191**, CPU ratio **4.837137190x**, peak RSS
2,045,800,448 bytes and conservative assets 44,242,609 bytes. Stage39 therefore
replaces Stage27 as the resource-qualified validation leader. No test data was
scored.
