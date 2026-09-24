# Stage116: cached count-row maxima

The Stage115 five-order gate spends CPU time recomputing the maximum next-
token mass of a training-derived count row for every inference prefix. Stage116
materializes those maxima once as checkpoint buffers and folds four affine
gate coefficients into one vector plus intercept; all source probabilities
and the gate equation are unchanged. The first 32×256 validation-input batch
matched Stage115 with zero observed probability and log-probability error.

Ten repeated CPU FP32 forwards had medians 2.68158 seconds for Stage115 and
2.64485 seconds for Stage116, a **1.37%** gain. This is below the roughly
3.3% saving needed to close Stage115's one-repeat 5.170× CPU preflight gap,
and is only a batch microbenchmark. No checkpoint is promoted or full CPU
qualification claimed. Test was not scored. Raw timing samples are in
`code/results/stage116-evidence/probe.json`.
