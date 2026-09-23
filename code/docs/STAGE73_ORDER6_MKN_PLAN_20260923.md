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

## Result

The first scan attempt exposed a 64-bit lookup defect rather than a model
failure: although the builder preserved a five-token context in 55 bits,
appending the target to that raw code requires 66 bits.  The corrected exact
lookup keys each edge by `(CSR row number, target)`; a regression test with
token IDs near 2047 verifies equality with the complete order-6 distribution.

The corrected train-only extension contains the exact frozen Stage25 lower
tables plus the new order-6 table and has checkpoint SHA-256
`b901f11766c005733f769fa5582e82f586db31a41edaefe93122aae1ed471953`.
Against the fixed Stage67 neural expert, order five reproduced
1.4101617730 BPB at weight 0.0625.  Order six selected weight 0.075 and scored
**1.4097481255 BPB**, a gain of only **0.0004136475 BPB**.  This is a real but
weak quality signal, too small to justify adding another sparse inference table
and resource qualification.  Stage73 therefore exports no candidate.  Raw
build and scan evidence is in `results/stage73-evidence/`; test was not scored.
