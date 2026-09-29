# Stage115: exact five-order gate export and CPU preflight

The Stage114 setting at 1.400224946 BPB uses the Stage92 neural, train-fitted
Stage100 four-feature gate, and Stage25 five-order MKN. This stage instantiates
the same fused inference path as Stage105, with a five-order count component,
serializes one self-contained checkpoint, and uses the unchanged CPU FP32
evaluator on the full validation set. It then measures the same baseline and
candidate in fresh processes once, including peak RAM and inference assets.
One repeat is an early feasibility check, not a final resource qualification.
No test evaluation is performed.

## Result

The exact exported checkpoint (SHA-256
`902e4b21c9ddf3afda2258ae032fe852517716cccfd5341567b2fabc21910e76`)
reproduced **1.4002249463 BPB** on complete validation. The single fresh-
process CPU comparison took 124.065 versus 23.997 seconds, a **5.169956×**
ratio. Peak RSS was 2,050,453,504 bytes and conservative inference assets
were 48,585,687 bytes; memory and assets pass, CPU does not. This is one
repeat and is only a preflight, but the 0.17× gap is too large to call
qualified. The candidate remains below the quality target as well. Preserve
Stage85 as the actual qualified leader. Raw records are in
`code/results/stage115-evidence/`. Test was not scored.
