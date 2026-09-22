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
