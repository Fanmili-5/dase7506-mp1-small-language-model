# Stage118: sparse count-addition kernels

The fused Stage115 predictor adds training-derived count masses to a dense
2,048-way neural distribution. Stage118 compared its existing advanced-index
addition with flattened `index_add_` and `scatter_add_` on the same first
32×256 validation-input batch. All three produced identical observed
probabilities and log probabilities. Their ten-repeat median CPU FP32 forward
times were 2.58654, 2.73972, and 2.69022 seconds, respectively. The
alternatives were 5.9% and 4.0% slower. Retain the existing addition;
no full-resource test or new candidate is warranted. Raw samples are in
`code/results/stage118-evidence/probe.json`. Test was not scored.
