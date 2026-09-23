# Stage61: second low-learning-rate continuation

Stage56's raw validation BPB still improved from 1.4226356 at continuation
update 3,600 to 1.4206119 at update 4,800, and its fixed average improved the
deployed neural expert to 1.4204239.  This stage continues the same hybrid-conv
Transformer and the same seed-17 sampled-window trajectory from the Stage56
five-checkpoint training average.

The only planned optimization change is a gentler fresh AdamW schedule: 4,800
updates, peak learning rate 0.00008, 50-update warmup, and cosine decay to
0.000008.  Architecture, data, R-Drop, auxiliary objectives, batch size and
inference graph remain fixed.  The sampler is advanced past all 12,000 earlier
updates, and the dropout stream is reset once to documented seed 56017; there
is no seed search.

Selection is fixed before launch: average updates 3,600/3,900/4,200/4,500/4,800,
export through the existing exact-equivalence path, and independently score
validation on CPU FP32.  MKN is rescanned only if this neural expert improves.
Test remains untouched.
