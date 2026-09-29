# Stage215: train-derived semantic input for the Stage54 Transformer

## Problem and fixed mechanism

Stage143's complete validation BPB is 1.399686, still 0.049686 above the
student's <1.35 goal. The Stage214 audit shows that the larger-backbone
complement is spread across target spellings; another lexical output head is
not a strong use of the deadline. We need broader contextual generalization
without a second inference backbone, because the Stage143/155 mixture cannot
meet the CPU/asset limits. This is a problem-first architecture pilot, not a
claim that train co-occurrence will work.

Stage215 adds one fixed train-derived semantic input stream to Stage54's
eight-layer alternating attention/convolution Transformer. Construct a
2,048-by-2,048 symmetric weighted co-occurrence table from *only* supplied
training-token IDs, with distances 1–5 and weight 1/d. Positive PMI followed
by a deterministic rank-64 low-rank decomposition produces a 2,048-by-64
semantic basis. It is stored in the checkpoint, not recomputed from any
evaluation text. At each input position, add a trainable 64-to-288 linear
projection of that token's semantic vector to the ordinary token embedding.
Initialize the projection to exactly zero, so step-zero predictions and all
existing Stage54 weights must match. The fixed basis cannot inspect future
tokens of an evaluation window; it is a train-only learned asset like a
checkpoint weight. The output head, prefix copy, R-Drop, deep supervision,
future-token auxiliary heads and optimizer are otherwise unchanged.

Stage149's fixed pairwise co-occurrence probability expert worsened
validation, and Stage65's byte-composed input gained only 0.000777 BPB.
Those results lower the prior of success but do not test this jointly trained
low-rank *input* representation. The strongest objection is that a Transformer
trained for 58.98 million target presentations may overwrite or ignore the
semantic input; a large training loss reduction alone would not count.

## Gates fixed before results

1. Build the basis with source and train-data hashes. It must have finite
   values, exact shape [2048,64], no validation/test reads, and no changes to
   the fixed tokenizer. Unit tests check zero-start Stage54 parity, causal
   future-token isolation, row independence, normalized output and nonzero
   semantic-projection gradients. A synthetic four-thread CPU timing screen
   must have median candidate/control <=1.20 and conservative projected
   Stage143 assets plus new basis/projection/source reserve <=64 MiB.
2. If preflight passes, use seed 17, batch 32 and the first 2,400 steps of
   Stage54's exact 7,200-step learning-rate schedule. Sample the same training
   windows and process exactly 19,660,800 primary targets. Score *complete*
   validation at 300-step intervals; the fixed step-2,400 endpoint alone is
   the advancement decision. Stage54's matched endpoint is 1.519950368612.
3. Continue to a full 7,200-step run only if Stage215 reaches at most
   **1.489950368612** at step 2,400, a material >=0.030-BPB improvement.
   If not, stop without changing semantic rank, context span, scaling, seed
   or learning rate. A passing pilot still requires a fixed average, complete
   CPU FP32 validation <1.35, three-repeat <=5x/4-GiB/64-MiB qualification
   and clean-extract reproduction before method freeze. No new test scoring
   occurs during development.

The protected Stage143 checkpoint, graph and evidence are never overwritten.
All pilot outputs use new run directories and disclose training/search costs.
