# Stage223: train-derived sparse bigram input representation (pre-outcome)

## Observation and question

Stage143's complete validation BPB is 1.399686162. Matched pilots of
shared-depth, extra local attention, positional bias, lexical output
hierarchies and longer auxiliary targets did not deliver the early gain
needed for <1.35. The Stage214 diagnostic found that known two-model
complementarity is distributed across positions and token groups. Those
facts motivate, but do not demonstrate, a different *input* mechanism.

The supplied training stream has 3,613,343 tokenizer IDs and 298,024
distinct adjacent token pairs. Its top 16,384 pairs cover 65.3814% of
training pair positions; this is a train-only observation, not a claim about
validation benefit. No external text or model is used.

## Candidate and rival

Candidate: a compact learned representation of common **input token
bigrams** may help the Transformer form phrase/context features before
prediction, repairing errors that a fixed output n-gram mixture cannot.
At position `t`, the extra input is only `(x[t-1], x[t])`; position zero
receives a zero vector. Select the 16,384 most frequent pairs in the
supplied training token stream, sorting tied frequencies by numeric pair
key. Other pairs map to zero. Train one 16-dimensional embedding per
selected pair and project it into the unchanged 288-dimensional backbone.
No train-derived target probability, future token or cross-window state
enters this representation. The tokenizer/evaluator remain unchanged.

Rival: the existing seven-token causal convolution, attention and MKN
output counts already extract the useful local collocations, so the extra
embedding merely memorizes the small training corpus. Under this rival,
same-target validation gain is negligible or negative. The new table is
not assumed to solve the requested score gap.

## Fixed discriminating procedure

Build and hash the pair-rank table from training text alone. It is an
int16 lookup over all `2048^2` possible pair keys; the selected rank is
stored with the checkpoint. Preserve Stage54's width, depth, attention,
convolution, output/copy heads, R-Drop, batch32, seed17, sample stream,
AdamW and first 2,400 learning rates of its 7,200-step schedule.
Only the input representation changes. At step 2,400 compare the complete
validation BPB over 376,599 targets with Stage54's fixed same-step
1.519950368612022. Intermediate 300-step scores are diagnostic; do not
choose an earlier checkpoint or change the table size/dimension after
seeing them.

Before quality training, require: same training/inference state after
stripping training-only heads; exact causal prefix and independent-row
behavior; normalized finite log probabilities; a two-update BF16 batch32
GPU smoke test below 8 GiB; and a synthetic OpenVINO feature graph whose
FP32 output matches eager to <=3e-4. Reserve 25,200,000 bytes for the
head/count/runtime assets and require graph plus reserve <=64 MiB.
Measure an interleaved eight-pair four-thread feature-time ratio to the
Stage143 graph; the Stage143 measured full scorer timings imply a
projected total CPU ratio <=4.5 is required to admit the pilot. These
are input-only screens, not formal release qualification. If either
preflight fails, stop this exact route.

Advance to a separately planned full 7,200-step training run only if
the fixed pilot endpoint improves by at least **0.030 BPB**. A passing
pilot still must achieve complete-validation <1.35 and actual CPU<=5x,
peak RSS<=4 GiB, inference assets<=64 MiB before any method freeze.
No test scoring occurs in this experiment. Stage143 stays protected.

The accountable student must review and understand this AI-assisted
candidate and its eventual result. This is a candidate hypothesis, not
evidence that token-pair embeddings improve the assignment.
