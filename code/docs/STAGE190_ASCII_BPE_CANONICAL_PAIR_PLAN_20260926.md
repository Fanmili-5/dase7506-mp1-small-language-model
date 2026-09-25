# Stage190: canonical ASCII-letter BPE pair feasibility screen

This is a distinct, fixed-tokenizer follow-up to the failed Stage189 **direct
merge-pair** mask. Stage189 removed only about 0.00134 mean probability mass
and gained 0.000590 BPB. A broader BPE property is worth one bounded screen:
when two adjacent final tokens both decode to ASCII letters only, they should
belong to the same ByteLevel pre-tokenization word. If encoding the two token
symbols concatenated does **not** return those two IDs, the adjacent pair may
be noncanonical even though it is not a direct merge-list entry. The rival is
that BPE segmentation of a substring may depend on its wider word context,
or ByteLevel pre-tokenization may introduce exceptions; a data-incidence
check is required and still is not a universal proof.

## Fixed protocol before outcomes

Use the unchanged course tokenizer JSON (SHA-256
`020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e`).
Select vocabulary symbols for which Python `str.isascii()` and
`str.isalpha()` are both true, including only nonempty strings. For **every
ordered pair** of these IDs, concatenate the symbol strings and encode the
result with the unchanged tokenizer. Mask the pair exactly when its encoded
ID sequence differs from `[left_id, right_id]`. No train/validation answer
affects construction of this 2,048×2,048 Boolean mask; no other token class
is masked. Record the symbol count, pair count, mask count and source hashes.

Tokenize the supplied **train and validation** files only. Stop if even one
actual adjacent target pair is masked in either split. This is an empirical
safety gate, not a proof about test; do not open or tokenize test. If it
passes, score exactly one full-validation Stage143 diagnostic at the same
predeclared `alpha=0.9` as Stage189:

`P'(v)=P(v)*(1-alpha*m(previous_id,v)) / (1-alpha*sum_u P(u)*m(previous_id,u))`.

The rule is causal and has positive support even if a masked pair occurs in
unseen data. Verify the unchanged Stage143 score, full 376,599-target/1,472-
window coverage, finite normalized distributions and both fixed halves.
Advance to **separate** inference implementation and fresh three-repeat
CPU/RAM/asset qualification only if complete validation improves by at least
**0.015 BPB**, both halves improve, and no true train/validation pair is
masked. Otherwise stop this exact pure-ASCII canonical-pair route; do not
change alpha, token class or rule after inspecting validation. Stage143's
checkpoint, graph, tokenizer, scorer and test split stay unchanged throughout.
