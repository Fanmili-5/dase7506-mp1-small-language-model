# Stage63: late primary-emphasis continuation

Stage61 still improves at its final raw checkpoint, but each stochastic pass
continues to optimize deep next-token supervision at weight 0.2 and +2/+3 token
prediction at weight 0.2.  Those training-only tasks helped shape the backbone;
late in training they may divert capacity from the official next-token BPB.

Stage63 starts from the fixed Stage61 five-checkpoint training average and
continues the same seed-17 sampled-window stream after 16,800 prior updates.
Over the first 600 updates, both auxiliary weights decrease linearly from 0.2
to 0.05 and then remain fixed.  R-Drop stays at 0.5.  The run uses 3,600
updates, a fresh AdamW optimizer, peak learning rate 0.00005, 50-update warmup,
and cosine decay to 0.000005.  Architecture, data, batch size and deployed
inference graph are unchanged; dropout uses documented seed 63017.

Selection is fixed before launch: average updates
2,400/2,700/3,000/3,300/3,600, export through the exact existing path, and
independently score validation on CPU FP32.  MKN is reconsidered only if the
neural average improves Stage61.  Test remains untouched.
