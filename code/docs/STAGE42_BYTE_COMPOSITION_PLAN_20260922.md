# Stage42: training-only byte-composed BPE embeddings

WikiText-2 has only about 3.6 million training tokens, while the fixed BPE
vocabulary contains 2,048 independently learned rows. Stage42 tests whether
sharing spelling structure improves rare-token generalization. Each BPE token is
represented by a deterministic bag of its reversible byte-level symbols plus
its first and last byte. A zero-initialized 768-to-256 shared projection is
added to the ordinary tied token/output embedding during training.

Zero initialization consumes no random numbers and makes the initial model and
outputs exactly equal to Stage26. The Transformer, content-copy head, row
dropout, deep supervision, +2/+3 multi-token objective, optimizer, seed,
schedule, update count and primary target count remain fixed. The only extra
trainable parameters are 196,608 shared byte-projection values.

At export, the learned byte contribution is added once to all 2,048 token rows;
the projection and feature matrix are removed. The deployed module, parameter
count, tensor shapes, assets and arithmetic are then exactly the ordinary
Stage26 `student_structured` graph. Consequently any quality gain has no
inference-time CPU or RAM cost. Selection uses the same fixed average of updates
6,000/6,300/6,600/6,900/7,200 and validation only. A gain of at least .003 BPB
triggers MKN/calibration rebuilding; otherwise the mechanism is rejected. No
test split is scored.
