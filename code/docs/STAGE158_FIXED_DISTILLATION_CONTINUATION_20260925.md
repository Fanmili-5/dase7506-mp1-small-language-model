# Stage158: one bounded continuation of the complementary-teacher student

Stage157 transferred 0.004728523 validation BPB in 900 updates, clearing
its predeclared 0.004 gate, but ended at 1.405148748 versus the qualified
Stage143's 1.399686162. The final 300 updates gained only 0.000232086 while
the learning rate fell from its 2e-5 peak to 2e-6. The testable alternative
is that the low terminal rate, not an absolute capacity ceiling, caused the
flattening. This is one finite experiment, not a schedule grid.

Start from the exact Stage157 pilot checkpoint SHA-256
`daa1bfa9ef598f1e5d7f00e7af2e8df8802a39c945284962f79392ecbc6687bf`.
Keep the same frozen Stage105/Stage155 50:50 teacher, Stage155 student shape,
0.75 full-distribution teacher CE + 0.25 hard train-label NLL, and physical
batch 16. Run 3,600 additional updates with fresh AdamW, peak LR 1e-5,
50-step warmup and cosine decay to 1e-6. Seed the dropout stream 158017;
recreate the pilot's seed-157017 training-window generator and advance its
first 900 draws so new train windows continue rather than repeat. The added
nominal training-target presentations are 14,745,600.

Score complete GPU FP32 validation at step 0 and every 300 updates.
Save the fixed last five checkpoints at updates 2400/2700/3000/3300/3600;
compare the endpoint and their parameter average once on complete validation.
Only a BPB below Stage143's 1.399686162 can advance to trained-graph
equivalence, complete CPU FP32 validation, three-repeat CPU/RAM/asset audit,
and clean-extract reproduction. Otherwise stop this teacher-transfer route
and preserve Stage143. No test scoring occurs. The 1.35 goal remains the
development target, not a score guaranteed by this continuation.

## Result and stop decision

The Windows job completed all 3,600 additional updates and **14,745,600**
additional primary train-target presentations in 718.84 training seconds.
The complete GPU FP32 validation score improved from 1.4051487477 at the
start to **1.4031053912 BPB** at step 3,600. The prespecified five-point
parameter average independently scored **1.4032149590 BPB** on all 376,599
targets (SHA-256
`603ab643ac274cb46206c80e7fdf8b52337c0e55c9bbc69100c5b3db1318fe55`).
The endpoint is better than the average but remains **0.0034192291 BPB worse**
than resource-qualified Stage143; it also misses the 1.35 target by
0.0531053912 BPB.

The fixed teacher-transfer route is stopped. The continuation did recover
another 0.0020433565 BPB after restarting the learning rate, so the Stage157
tail was not an absolute plateau; nevertheless it failed the predeclared
Stage143 quality gate. No compact graph export, CPU/RAM/asset qualification,
promotion or test scoring followed. Full source hashes, trajectory, average
score, task status and console are in `../results/stage158-evidence/`.
