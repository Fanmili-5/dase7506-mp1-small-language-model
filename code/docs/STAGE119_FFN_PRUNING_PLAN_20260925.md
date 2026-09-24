# Stage119: train-importance FFN channel pruning screen

Stage108's train-prefix probe showed that direct removal of low-energy SwiGLU
channels changes hidden features, but it did not measure actual BPB. Stage119
uses the same 96 training-prefix windows and fixed seed to rank each of eight
FFN blocks' channels by activation energy times output-column norm. It then
masks the least important 10%, 20%, or 25% in every block, without training,
and scores the same Stage92 + Stage94 + order-six MKN mixture on complete
validation. The unchanged 1.401288-BPB reference must reproduce. This is a
structural-screen diagnostic only: an accepted ratio would need physical
smaller-matrix export, train-only repair, dynamic gate transfer, and exact CPU,
RAM, asset qualification. Test is not scored.

## Result

The unchanged reference reproduced at **1.401288462 BPB** on all 376,599
validation targets. Masking 10%, 20%, or 25% of selected FFN channels in all
eight blocks scored **1.417569**, **1.444275**, and **1.462414 BPB**,
respectively. The 20% option loses 0.04299 BPB before repair, essentially
the same size of loss as deleting one complete block in Stage112. Stage113's
fixed-budget repair of that block recovered only about half the loss, so a
new costly repair run is not justified by this screen. This does not prove
all structured pruning fails, but rules out treating a simple 20% mask as
a cheap near-exact CPU fix. Full per-block channel IDs and scores are in
`code/results/stage119-evidence/result.json`. Test was not scored.
