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

## Result

The smaller learning rate and auxiliary anneal limited the initial disturbance:
validation moved from 1.4178313777 at update 0 to 1.4190538448 at update 600,
then recovered below the start by update 1,800.  The five fixed averaging
points at updates 2,400/2,700/3,000/3,300/3,600 scored
1.4169926791/1.4164341258/1.4160881818/1.4160161750/1.4160604035 BPB.

The exported average independently scored **1.4161061665454577 BPB** on CPU
FP32, improving Stage61 by **0.0017251918951285 BPB** with the identical
deployed graph.  Its checkpoint SHA-256 is
`878c728f43c9e2d359e165b2dfdc1e3baeed79efa4bee7e33696906378dc0b25`.
The result advances to a fresh fixed MKN scan; no test split was scored.
