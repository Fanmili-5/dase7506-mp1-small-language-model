# Stage163: compact train-only hidden-state retrieval pilot

Stage143's complete validation remains 1.399686162 BPB, 0.049686 above
the desired 1.35. Larger/deeper networks, exact token-suffix lookup, a
spelling residual, train-only loss reweighting, and a complementary second
Transformer have not closed this gap. Stage162 shows the existing two-model
ensemble itself bottoms out near 1.376 under an impossible window oracle.
The new hypothesis is that **semantic context similarity**, unlike exact
token suffix equality, can recover training continuations relevant to the
medium-frequency, unseen-prefix targets. The mechanism follows the
train-only key/value and probability-interpolation concept of Khandelwal
et al., *Generalization through Memorization: Nearest Neighbor Language
Models* (ICLR 2020, https://arxiv.org/abs/1911.00172), but is redesigned
for this course's 64-MiB and 5x-CPU limits. It uses no external text or
weights. A later implementation must acknowledge the reused concept.

Construct 100,000 deterministic keys, evenly spaced over the 3,613,343
training-token stream's next-token positions, using the frozen Stage105
neural hidden states from independent 256-token windows. Values are the
corresponding supplied train next-token IDs. Fit a 64-dimensional PCA basis
on those *training* keys only, normalize projected rows, and round them to
signed int8. The raw key budget is 6,400,000 bytes, values 200,000 bytes,
PCA/centering under 100 kB; together these leave several MiB under the
Stage143 55,810,412-byte measured inference bundle before index overhead.
The pilot computes exact top-32 cosine neighbors of these **quantized**
keys for every independent validation prefix using GPU brute force. This
is a quality diagnostic, not a CPU/asset qualification and not a deployable
ANN index. For the true validation target, read out the top-32 softmax
retrieval probability and evaluate the fully normalized mixture
`(1-lambda)*Stage143 + lambda*kNN` at fixed lambda values
`{0, .05, .10, .20, .30}` and neighbor temperatures `{.05,.10,.20}`.
The label is used only to read out target probability *after* retrieval;
it never enters keys, search, neighbor weights or inference selection.

Verify train/validation token hashes, frozen Stage105 checkpoint SHA,
Stage143 cached target-probability SHA, all 376,599 targets, raw byte count,
finite hidden/query rows and baseline BPB. Record PCA eigenvalues, quantized
array bytes, retrieval hit coverage, complete BPB for all 15 cells, source
hashes, elapsed time and GPU memory. A best full-validation gain of at least
**0.020 BPB** over Stage143 is required before building an approximate CPU
index or any submission candidate. Such a gain does not itself imply the
final 1.35 target, budget compliance, or test performance. If the gate
fails, stop the retrieval route. Test is untouched.

The strongest objections are severe datastore sparsity at 100k keys and
CPU query cost. The fixed pilot screens quality first; success would still
require a separately verified CPU index with exact independent-window
causality, full 2,048-way normalization, <=5x time, <=4 GiB RSS and <=64 MiB
of all uncompressed assets. Avoid using validation labels to select keys,
PCA, an ANN index or an adaptive retrieval gate.

## Observed pilot and decision

The fixed Windows GPU diagnostic completed over all 376,599 validation
targets. The top-32 compressed-neighbor list contained the true next token
on **61.6669%** of positions, but this coverage alone did not make a good
probability model. The Stage143 zero-mixture control reproduced
**1.399686162 BPB**. Every nonzero mixture in the predeclared 3x4 grid
worsened it: the least harmful, temperature 0.05 and weight 0.05, scored
**1.401071171 BPB**. The best cell over the full grid was weight zero.
The four saved pilot asset files occupy 6,675,392 bytes, yielding a
62,485,804-byte Stage143-plus-pilot projection before index overhead;
this is **not** CPU or asset qualification. The required >=0.020 BPB
quality gain failed, so no approximate CPU index, full-distribution
inference integration, or test scoring follows. This rejects the specified
100k-key/64D/32-neighbor configuration, not all possible retrieval models.
The complete fixed grid, hashes, timing, GPU memory and job status are in
`../results/stage163-evidence/`. The source and train-derived array hashes
permit regenerating the diagnostic; rejected key arrays remain in the
Windows run directory rather than the final inference bundle.
