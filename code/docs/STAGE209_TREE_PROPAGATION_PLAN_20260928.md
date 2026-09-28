# Stage209: resource repair of the Stage208 lexical hierarchy (pre-outcome)

Stage208 failed its input-only CPU gate: its 11 repeated full-vocabulary
gathers take 3.103434 seconds per 32x256 batch. It was not trained, and no
validation/test result was read. The failure is in *how the same tree
probabilities are enumerated*, not in the tree's mathematical definition.

Keep exactly the same lexical sort, balanced binary topology, trainable
decision weights, initial gate, Stage54 backbone/config, R-Drop objective,
seed, sampler, 7,200-step LR prefix and 2,400-step pilot budget. Compute
the identical leaf log probabilities by propagating partial log probability
from one root through levels with 1, 2, 4, ..., 2,048 active branches,
then permute leaves back to fixed BPE IDs. This takes work proportional to
the number of tree edges, not 11 times the full vocabulary. Do not change
tokenizer, split, benchmark or any probability formula.

Fixed input-only gate before training:

1. For the same random weights/hidden states, new versus Stage208 lexical
   log probabilities must agree within 2e-6. The whole predictor must
   normalize within 1e-5 and preserve future-position and independent-row
   isolation within 1e-5.
2. Eight warmed four-thread synthetic 32x256 lexical-head calls must have
   median <=0.50 seconds; conservative assets <=64 MiB; one batch-32 BF16
   R-Drop optimizer update on Windows must have finite nonzero gradients,
   <=7 GiB peak allocated memory and reserved memory below GPU total.
3. Only if both pass, run Stage54's same first 2,400 seed-17 updates and
   complete GPU-FP32 validation. The archived matched control is
   1.5199503686120217 BPB; require >=0.030 BPB early gain to justify a
   full run. No seed, topology, gate initialization or LR sweep follows a
   failure. A passing pilot is not a final predictor: it must still meet
   full CPU-FP32 validation <1.35 and the actual 5x/4-GiB/64-MiB limits.

The already-scored frozen Stage143 test is not consulted for this method.
There will be no new test scoring before a separately frozen deployable
candidate exists. Stage143 hashes and bundle stay intact.
