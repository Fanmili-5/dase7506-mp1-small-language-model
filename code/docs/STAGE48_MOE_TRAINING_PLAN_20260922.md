# Stage48: matched-target sparse MoE training

Stage46 admitted the full MoE/count inference graph at 4.87245x CPU, 2.052 GB
peak RSS and 52.62 MB assets. Stage48 now tests its quality with one fixed,
matched-target run. The seed, sampled-window generator, batch 32, 7,200 updates,
AdamW schedule, row dropout, deep supervision, +2/+3 prediction and last-five
averaging match Stage26. Only the dense SwiGLU683 modules become two-expert
top-1 SwiGLU512 modules.

All tokens are processed; there is no capacity dropping. A 0.01 load-balancing
term and 0.001 router z-loss are fixed before training. Routing uses only the
current causal hidden state. Training reports both router losses, targets, time
and peak GPU memory. Export removes training-only prediction heads but retains
the learned routers and experts, with exact output equivalence checked.

The neural average must improve Stage26 by at least 0.003 BPB before MKN
calibration and final trained-checkpoint resource qualification. Passing the
random-weight resource gate was not a quality claim. No test split is scored.

The first launch stopped on its first BF16 forward before completing any update:
the sparse `index_copy` destination was FP32 while autocast made expert outputs
BF16. The failed run and logs are retained. The fix explicitly casts each
expert result back to the residual dtype and adds a CPU BF16-autocast regression
test. The replacement launch uses a new run directory; experimental settings
are unchanged.

The corrected `-b` launch was manually stopped after its PowerShell transcript
failed to expose live JSON progress. Recovered `progress.json` showed that the
job had actually reached step 1,500 in 316.75 training seconds; this was a
monitoring error, not a training hang. Its validation BPB at steps
300/600/900/1200/1500 was 1.96699/1.76710/1.69017/1.64783/1.61698. The partial
run and stale running-status receipt are retained rather than overwritten.

The replacement training kernel evaluates both experts with large dense GEMMs
and gathers only the routed output. A dense-training-versus-sparse-eval test
checks their numerical semantics, while CPU evaluation continues to execute
only the chosen expert. The `-c` run restarts from the same seed and settings;
minor floating-point trajectory differences from full versus indexed GEMMs are
possible and are disclosed.

## Result

The completed `-c` run processed the fixed 58,982,400 primary targets in
1,890.39 GPU training seconds. The 7,200-step endpoint was 1.4783006303 BPB;
the prespecified last-five average scored **1.4745912215 BPB** on independent
CPU FP32 validation (checkpoint SHA256
`c3ba2e4e024937b1b95a4b214dea255e1948637708fc11c14e167d4661833138`).
This is 0.0095973132 worse than the matched Stage26 average, 1.4649939083.

Router balance remained near its ideal value (1.0166 at the final logged step),
so expert collapse is not the explanation. The likely trade-off is unfavorable:
each token sees only a 512-wide expert instead of the dense 683-wide FFN, while
the small corpus does not provide enough data for conditional expert capacity
to compensate. Stage48 is rejected without MKN mixing or a trained-checkpoint
resource rerun. Peak allocated/reserved GPU memory was 2.325/2.496 GB. No test
split was scored.
