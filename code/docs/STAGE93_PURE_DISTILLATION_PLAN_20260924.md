# Stage93: pure heterogeneous-distillation continuation

Stage92 transfers part of the 1.38320 neural-ensemble advantage and improves the
single-model endpoint to 1.40173574 BPB, with the curve still descending. Its
objective nevertheless assigns 25% weight to hard-label NLL, which pulls the
student back toward the already saturated original training objective.

Stage93 starts from the exact Stage92 endpoint and keeps the same frozen
0.55/0.45 Stage71/Stage76 teacher. It removes hard-label loss entirely and
optimizes only full-distribution teacher cross-entropy for 1,500 updates at
peak learning rate 5e-6. Batch size remains 24 and the original seed 92017 is
continued by advancing the RNG past Stage92's 1,800 draws, so this is not a new
seed trial. The fixed average uses updates 600/900/1200/1500. Validation only
monitors the unchanged calibrated student plus count weight 0.075. Test remains
untouched.

## Result

Every pure-distillation checkpoint regressed from the 1.4017357355 start; the
fixed average scored 1.4025742414 BPB. The hard-label component is therefore a
necessary regularizer, and this route is closed. Test was not scored.
