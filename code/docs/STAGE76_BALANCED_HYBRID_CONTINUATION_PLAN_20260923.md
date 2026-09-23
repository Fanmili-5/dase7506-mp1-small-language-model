# Stage76: balanced-hybrid low-learning-rate continuation

Stage74's matched 7,200-update average beats Stage54 by 0.0033153 BPB, and
Stage75 shows that its improvement survives the unchanged MKN mixture.  Stage76
therefore applies the already successful Stage56 continuation recipe to the
exact Stage74 training average: 4,800 additional updates, batch 32, the
continued seed-17 sampling stream, a fresh AdamW optimizer, peak learning rate
2e-4, minimum 2e-5, R-Drop 0.5 and unchanged auxiliary heads.  A distinct fixed
dropout RNG seed 74017 identifies this architecture's continuation; it is not a
seed screen.

Selection is fixed before launch: uniformly average updates
3600/3900/4200/4500/4800, export the inference graph, and score once on
validation CPU FP32.  The branch advances only if the average improves Stage74;
the frozen MKN grid is rerun afterward.  Test remains untouched.

## Result

The exact start reproduced Stage74 at 1.4252788070 BPB.  The five fixed
averaging points at updates 3600/3900/4200/4500/4800 scored
1.4207736098/1.4198540995/1.4191823304/1.4190414921/1.4184223382 BPB.
The exported average independently scored **1.4184445504 BPB** on CPU FP32,
with SHA-256
`f4c499b63db9b39ce8062b4f07eff313ec23c491952e1b4c92f22d873b23bfdf`.
This improves Stage74 by **0.0068342364 BPB** and is 0.0019793112 below the
matched Stage56 average.  The continuation therefore passes its quality gate
and advances to a fresh frozen-MKN scan.  Test was not scored.
