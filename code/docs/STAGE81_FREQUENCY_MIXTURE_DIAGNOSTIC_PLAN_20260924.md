# Stage81: frequency-group mixture diagnostic

After Stage79, only about 0.00303 BPB remains to the 1.4 development target.
A single count weight assumes that the MKN expert is equally useful for every
candidate token, although count and neural errors can vary strongly with token
frequency.  Stage81 tests a deployable candidate-dependent mixture without
using the unknown next token as an input.

The 2,048 vocabulary entries are ranked only by supplied-training frequency and
partitioned into 32 fixed equal-size groups.  Nested 4/8/16/32-group variants
assign one count weight to every candidate vocabulary entry, form the complete
unnormalized mixture, and renormalize across all entries.  At inference this
computes weights for every possible token before the outcome is known, so it
remains causal and target-independent.

For diagnosis, opposite contiguous validation halves fit and evaluate the
bounded group weights.  The combined out-of-half BPB is compared with the exact
Stage79 scalar reference.  Validation-fitted vectors are never exported.  A
material gain is only evidence to fit the same low-dimensional mechanism using
the supplied training text; otherwise the branch stops.  Test remains
untouched.
