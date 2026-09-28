# Stage208 result: lexical hierarchy fails the CPU preflight

The Windows RTX 3070 Ti synthetic input-only preflight completed without
opening any train, validation or test text. Three structural unit tests also
passed on Mac and Windows. The 2,048-way predictor normalized within
4.77e-7, hierarchy within numerical zero; modifying a future input token
changed no preceding log probabilities, and independent rows agreed within
4.77e-7. The 32x256 BF16 R-Drop synthetic optimizer step had finite loss and
nonzero lexical gradients, 5,813,714,944 peak allocated GPU bytes and
6,115,295,232 peak reserved bytes. Conservative inference assets were
58,972,524 bytes, below 64 MiB.

The fixed resource gate **failed**: eight warmed four-thread 32x256
hierarchy-only CPU calls had median **3.103434 seconds**, exceeding the
predeclared **0.50-second** cap by 6.21x. This is just the added head, not
the full predictor. The exact Stage208 implementation therefore cannot meet
the practical CPU envelope and is stopped **before any training or
validation selection**. No Stage208 test score exists. Stage143 remains the
protected qualified candidate, still above the student's acceptance levels.

Raw result with source, tokenizer and config hashes:
[`../results/stage208-preflight-a.json`](../results/stage208-preflight-a.json).
This negative result concerns the naive full-vocabulary, 11-index-select
implementation. It is not a measurement of a different tree-propagation
algorithm or a general proof that lexical hierarchies cannot work.
