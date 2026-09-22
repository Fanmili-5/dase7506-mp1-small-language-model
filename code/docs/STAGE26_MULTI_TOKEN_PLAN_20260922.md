# Stage26: training-only multi-token prediction

Stage22 deep supervision improved the neural validation BPB from 1.4823791185
to 1.4702546457 at identical inference cost. Stage26 tests a second representation
learning signal while keeping that exact deployed architecture.

The eight-layer width-256 prefix-copy Transformer, embedding-row dropout,
intermediate layers 4/6, optimizer, seed, batch size, 7,200 updates, learning-rate
schedule and 58,982,400 primary next-token targets match Stage22. From each final
causal hidden state, two training-only heads additionally predict offsets +2 and
+3. Each head has a separate RMSNorm and identity-initialized 256x256 projection
into the tied vocabulary head. Their averaged loss receives weight 0.2; the
existing intermediate loss retains weight 0.2.

The model never receives future tokens as input. Future tokens are labels from
the supplied training text, and 117,964,800 future-label presentations are
reported separately from the unchanged primary target count. The auxiliary
modules initialize without random draws, preserving the Stage22 base
initialization and post-construction RNG state. One extremely rare end-of-corpus
start can be clamped so all future labels exist; its count is recorded.

Selection is the same fixed average of updates 6,000, 6,300, 6,600, 6,900 and
7,200. Export removes all intermediate and future heads and verifies exact output
parity against `student_structured`. Only validation is scored. If neural quality
improves, count mixing and exact resource qualification are separate later gates.
No test scoring, external data, pretrained weights, seed search or adaptive
training extension is allowed.

Substantive AI assistance covers method selection, implementation, tests,
orchestration and analysis; it must be disclosed and understood by the student.
