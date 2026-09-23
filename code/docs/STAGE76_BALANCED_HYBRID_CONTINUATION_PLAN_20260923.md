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
