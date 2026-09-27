# Stage191: train-only ASCII word-prefix successor diagnostic

Status: pre-outcome exploratory plan, 27 September 2026 (UTC+8). The analyst
has already seen many complete-validation results while developing Stage143;
this is not an independent confirmatory study. The student owns the coursework.
Only the supplied public train/validation text and fixed tokenizer may be
processed locally. No external text, validation fitting, test read/scoring or
change to the protected Stage143 predictor is authorized by this plan.

## Observation, question and rivals

Stage143 is resource-qualified on Windows at 1.399686162 complete-validation
BPB, above the aspirational 1.35 target. Its train-derived order-six token
MKN is useful, while exact longer *preceding-token* suffix lookup gained only
0.003037 BPB in Stage146. A different context equivalence may pool within-word
continuations across preceding words: the current ASCII letter prefix since a
ByteLevel space-marked token. This is a candidate predictive mechanism, not a
claim that morphology or a larger model has been solved.

The rivals are (a) token MKN/neural copy already captures the same signal,
(b) word prefixes have little validation coverage, and (c) sparse train
successors overfit and damage probabilities for unseen continuations. The
result cannot isolate a linguistic mechanism; it only tests this fixed expert.

## Frozen measurement and analysis

1. Verify the existing Stage143 validation-target log-probability array against
   its JSON SHA-256, checkpoint SHA-256 and baseline BPB. Verify the provided
   tokenizer and train/validation text against `data/manifest.json`. Never
   open `wikitext_test.txt` or call eager `common.load_data()`.
2. Token ID spellings come only from the fixed tokenizer. After consuming a
   token whose spelling is `Ġ` followed by ASCII letters, start a case-sensitive
   prefix with those letters. A subsequent pure ASCII-letter token extends it.
   Any other token clears it. Prefixes longer than 32 letters are inactive.
   For train positions, count the *next token ID* after each active prefix,
   including a boundary or punctuation token. No next-token label is used to
   construct the prefix itself. Train text is one supplied stream.
3. At validation, reset prefix state at each independent 256-token input
   window. Query the train table after consuming the current input token.
   Activate the expert only if the exact prefix has at least **3** train
   successor observations. Its normalized distribution is unsmoothed train
   successor relative frequency. If inactive, retain Stage143 unchanged.
4. Primary fixed mixture weight is **0.10**:
   `p_mix(v) = 0.9*p_stage143(v) + 0.1*p_prefix(v)` for active contexts.
   This is a valid full distribution because both components normalize.
   Calculate target probabilities solely for NLL readout. Weights 0, 0.05 and
   0.20 are reported as sensitivity cells, not selected for advancement.
   The unit is each validation next-token target; BPB divides aggregate NLL
   by the unchanged raw UTF-8 validation byte count. Report all 376,599
   targets, active/correct coverage, and two fixed 736-window halves.
5. A positive advancement screen requires the primary 0.10 cell to improve
   complete validation by **at least 0.015 BPB** and each half by at least
   **2,000 nats** of aggregate NLL. The half threshold avoids inventing a
   per-half raw-byte denominator at a token-window boundary. A smaller or
   one-sided gain stops this route. Even passing
   does not make a deployable score: build a causal inference implementation,
   estimate serialized table bytes against the remaining 11.3 MB headroom,
   and remeasure three CPU/RAM/asset runs before promotion. No test scoring.

The fixed zero-weight cell must reproduce Stage143 to 2e-5 BPB. Any count
using validation successors, cross-window prefix state, changed tokenizer, or
target-dependent mixture weight invalidates the diagnostic. Report all fixed
cells and errors, not just the best result. AI assistance in the hypothesis,
script and interpretation is disclosed in the project README.
