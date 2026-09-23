# Stage73: train-only order-6 MKN extension

The accepted Stage25 modified Kneser--Ney expert stops at order five, so its
longest table uses four observed tokens to predict the next token.  Stage73
keeps every frozen Stage25 unigram/order-2--5 tensor unchanged and appends one
raw order-6 table.  It retains only six-grams observed at least twice in the
supplied training text, applies modified count-bucket discounts, and backs all
pruned mass into the unchanged order-5 distribution.

This is a bounded use of the remaining asset budget, not an unpruned retrieval
database.  Because six token IDs need 66 bits, the builder sorts exact
`(five-token-context, next-token)` pairs rather than using a lossy 64-bit hash.
The first gate is an exact target-probability validation screen against the
same Stage67 neural and weight grid for both original and extended counts.  No
checkpoint is promoted and no resource claim is made unless the extension
improves the original count expert materially.  Test remains untouched.
