# Stage105: single-pass gated MKN inference

Stage104 traverses the count tables twice: once to obtain backoff and maximum
sparse mass for the gate, and once to add the count distribution. Stage105
collects both in one traversal, leaving the Stage103 neural weights, count
tables and training-derived gate coefficients fixed. It first checks the
complete probability distribution on deterministic windows against Stage103,
then runs the entire official CPU FP32 validation and a three-repeat CPU/RAM
resource audit. No test scoring is performed.

## Result and next bottleneck

The complete Windows validation remained **1.399686163172 BPB**. Three
alternating CPU runs yielded candidate times 124.075, 127.544 and 127.386
seconds; the baseline median was 23.110 seconds. The resulting **5.512127×**
ratio fails the fixed 5× gate. Peak RSS was 2,064,015,360 bytes and
conservative inference assets 53,270,487 bytes, both passing. Checkpoint
SHA-256: `7597f7519b4bce5dd3617f495223466cde699267d06e2ae74141b50a46160fa2`.
The raw independent evaluator and resource records are in
`code/results/stage105-evidence/`. Test was not scored.

A single-batch CPU FP32 operator profile (`scripts/profile_stage105_cpu.py`)
places 62.7% of self CPU time in matrix multiplication, 9.4% in attention,
2.0% in the dynamic-gate top-2 selection, 1.4% in sparse maximum-mass
reduction, and 0.1% in count-key binary searches. The first MLP projection
alone uses 686.9 ms of a 2.677-second profiled forward. Further sparse
lookup-only optimization is unlikely to close the approximately 10-second
validation-time gap; the next experiment should target the neural/gate
compute path and independently requalify any changed predictor.
