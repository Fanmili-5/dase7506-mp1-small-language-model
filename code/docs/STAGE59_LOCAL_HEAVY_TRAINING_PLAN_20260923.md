# Stage59: train the resource-admitted local-heavy architecture

Stage58 admits the full width-320, six-convolution/two-attention inference graph
at 4.7406x baseline CPU time.  Stage59 now runs the same seed-17 R-Drop protocol
as Stage54: batch 32, 7,200 updates, AdamW, identical learning-rate schedule,
row dropout, deep supervision, +2/+3 future prediction and R-Drop alpha 0.5.

Only the backbone allocation changes.  Selection is the prespecified average
of updates 6,000/6,300/6,600/6,900/7,200 followed by exact export and independent
CPU FP32 validation.  The neural average must beat Stage54's 1.4285940452 BPB
before any MKN mixture or continuation.  No test split is scored.

## Result

The local-heavy model converged faster: it beat Stage54 by 0.02095 BPB at
update 300 and remained slightly ahead through much of training.  That early
advantage did not persist.  Its fixed averaging points at updates
6,000/6,300/6,600/6,900/7,200 were
1.4388810491/1.4366506033/1.4351759596/1.4329516109/1.4327642334 BPB,
all worse than the matched Stage54 points.

The exported average scored **1.429628574507109 BPB** on independent CPU FP32,
which is 0.0010345293 worse than Stage54.  The candidate is rejected without
MKN mixing or continuation.  Its checkpoint SHA-256 is
`c46bb3f2b494cffef52b63baf43ae165aca2c13868420bf72539085c790745c3`.
No test split was scored.
