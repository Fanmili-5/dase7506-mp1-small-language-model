# Stage203: train-derived class-context correction (pre-outcome plan)

This is a fixed, complete-validation-only diagnostic. It does not alter or
unfreeze Stage143, load test text, fit parameters to validation labels, or
report a new submission score.

## Why this mechanism is distinct

Stage143's order-six MKN models exact BPE-token histories; longer exact
histories and word-prefix successors gave only milliscale gains. Stage151
located high validation NLL on medium-frequency target tokens absent from
the current causal window. A class-context model pools *different tokens*
with similar train-learned output embeddings, potentially generalizing across
unseen exact contexts. The correction changes only probability assigned to
each of 64 disjoint next-token classes, while preserving the frozen
Stage143 relative probabilities **within each class**. This is neither a
scalar neural/count gate nor another exact-token n-gram table.

## One fixed train-only construction

1. Load the SHA-pinned frozen Stage143 checkpoint and its 2,048 by 288 tied
   output matrix. L2-normalize the rows. With NumPy RNG seed **203017**,
   sample 64 unique initial centroid rows weighted by supplied-train token
   frequency plus one. Run exactly 20 cosine Lloyd iterations; retain an
   empty centroid unchanged if encountered. Reject the construction if any
   class is empty at the end. No validation embedding or label enters fitting.
2. Encode only the supplied training text with the unchanged BPE-2048
   tokenizer. Map each train token into one of the fixed 64 classes. Count
   all directed class bigrams and trigrams. Form normalized class unigram
   probabilities with add-one smoothing. Use hierarchical Dirichlet
   interpolation: bigram pseudo-count mass **32** times the unigram;
   trigram pseudo-count mass **64** times the bigram. These numbers are
   fixed here, not validation-tuned.
3. At independent validation-window position 0, let class model Q be the
   bigram after the current input class; from position 1 onward use the
   trigram after the preceding and current input classes. Stage143's
   unchanged normalized distribution is P. Let M(c) be P's total mass in
   class c. With a fixed geometric correction **gamma=0.25**, define
   `R(v) proportional to P(v) * [Q(class(v))/M(class(v))]^gamma`.
   The normalizer is the 64-class sum of the corresponding class masses.
   This is a fully normalized causal 2,048-token predictor; it does not
   access the target until NLL readout. Reset class history between the
   evaluator's independent 256-token windows.

## Fixed checks and advancement threshold

Verify train/validation/tokenizer/checkpoint/graph hashes, 64 nonempty
clusters, normalized unigram/bigram/trigram rows, positive finite masses,
and synthetic full-distribution normalization/causality. The unchanged
Stage143 full-validation BPB must reproduce 1.399686162 within 2e-5 over
376,599 targets / 1,148,007 bytes before computing the candidate score.
Report gain in each fixed contiguous half of the 1,472 windows. Advance to
an actual portable CPU implementation and formal 5x/4GiB/64MiB gates only
if the fixed gamma=0.25 candidate improves complete validation by at least
**0.030 BPB**, both halves improve, and the dense 64^3 FP32 trigram plus
other extra assets project within the current 11,298,452-byte asset margin.
If it fails, stop without a cluster-count/smoothing/gamma search or test.

Even a passing diagnostic cannot be submitted: it would still need a
serialized train-only table, exact CPU-FP32 score, three-repeat resources,
new method freeze, and a separate test authorization. The strongest
objection is that Stage143's neural and MKN components may already predict
the 64-class transition almost perfectly; this single diagnostic will
measure whether a material residual class signal exists.
