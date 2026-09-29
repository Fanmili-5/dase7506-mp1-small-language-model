# Stage69: frozen-backbone low-rank output adaptation

The byte-composed continuation and vocabulary intercept are reproducibly
positive but together improve the neural model by only 0.00101 BPB. Stage69
tests a more expressive zero-start mechanism: a rank-16 residual on the
vocabulary projection, trained while the entire Stage67 backbone, copy route,
embedding matrix and output bias remain frozen.

The low-rank residual starts at exactly zero (Gaussian A, zero B), so the initial
predictor is identical to Stage67. It receives five deterministic full passes
over supplied training text in independent 256-token windows. The fixed
candidate is the epoch-3/4/5 parameter average. At export, the residual is
materialized into one independent output matrix; inference still performs one
vocabulary matrix multiplication, while the input embedding remains unchanged.
This adds about 2.25 MiB of assets but no new deployed matmul.

Only validation is scored. The mechanism advances to MKN/resource qualification
only if the fixed exported average improves Stage67 materially. Test remains
untouched.

## Result

The zero-residual start reproduced Stage67 at **1.4150978831 BPB**.  Training
loss improved monotonically across all five full passes, but validation
immediately regressed: epochs 1--5 scored
1.4207569637/1.4213344936/1.4214285184/1.4217941369/1.4218029379 BPB.
The prespecified epoch-3/4/5 average independently scored
**1.4194831103 BPB** on CPU FP32 (checkpoint SHA-256
`92833a24f76fab41b8a590c931f1288ca96aecf37182c2a02831ed5b062aa1aa`).
This is 0.0043852495 worse than Stage67, so the expressive output-only
adaptation is rejected as overfitting.  It does not advance to MKN scanning or
resource qualification and does not change the Stage68 leader.  Test was not
scored.
