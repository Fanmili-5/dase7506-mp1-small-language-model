# Stage189: tokenizer-implied adjacent-pair support screen

## Fixed question and boundaries (before reading outcomes)

The Stage143 predictor assigns a normalized distribution over all 2,048 BPE
tokens at every causal position. Some pairs of adjacent vocabulary symbols
appear as an explicit merge rule in the **fixed supplied tokenizer**. If a
previous token and proposed next token form such a pair *inside the same
ByteLevel pre-tokenization segment*, the pair should merge and cannot be the
final canonical adjacent tokenization. This could leave probability mass on
tokenizer-impossible successors. The question is whether a conservative,
input-only support adjustment has enough mass to matter for BPB.

The important rival is that a merge pair can span a pre-tokenization boundary
in an actual token stream; a raw merge-list mask could then suppress a legal
target. A second rival is that the neural model has already learned to assign
almost no mass to these pairs. Therefore the first gate is a **train and
validation pair-incidence audit**, not a new predictor. Build the 2,048×2,048
Boolean mask solely from the 1,792 explicit merge pairs in the fixed tokenizer
JSON, mapping each merge's left/right symbols to vocabulary IDs. Verify every
merged concatenation is in the vocabulary and no duplicate left/right pair
exists. Tokenize the supplied **train and validation** text with the unchanged
course tokenizer, then count all observed adjacent pairs, globally and within
independent 256-token windows. Do not open, tokenize or score test.

**Stop** if any actual train or validation target pair is masked. Zero
observations would be an empirical safety check, not a universal proof; a
passing mask would still need a pre-tokenization-boundary argument before a
hard zero could be deployed. If zero, run one fixed full-validation Stage143
diagnostic: for each position, let `P` be the unchanged normalized
distribution, `m(v)` the mask row keyed only by the current input token, and
`alpha=0.9` fixed here. Score the normalized
`P'(v)=P(v)*(1-alpha*m(v)) / (1-alpha*sum_u P(u)*m(u))` on all 376,599 targets.
The factor remains positive even for a masked true token, so a rare unseen
boundary case would not make log loss infinite. Verify `alpha=0` reproduces
Stage143, causality, normalization, and complete coverage. No validation
label changes the mask or alpha.

Advance only if the complete score improves by **at least 0.015 BPB**, both
fixed 736-window halves improve, and no true pair was masked in train or
validation. Advancement would license a separate inference implementation,
three fresh CPU/RAM/asset checks and clean-extract reproduction; it would not
promote a diagnostic result or authorize test. The 2,048×2,048 byte mask is
4,194,304 bytes before packaging, inside Stage143's current 11,298,452-byte
asset headroom, but this estimate is not a resource measurement. A failing
incidence or score gate closes this fixed direct-merge mask; it does not rule
out all tokenizer-state models.
