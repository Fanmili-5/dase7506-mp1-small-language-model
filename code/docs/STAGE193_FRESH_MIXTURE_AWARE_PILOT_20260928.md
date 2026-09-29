# Stage193: fresh mixture-aware training of one deployable Transformer

This is post-freeze exploratory **train/validation-only** research. It cannot
change the already frozen Stage143 test record, and course permission to
replace a tested candidate has not been established. The experiment does not
open, score, or select on the test split.

## Hypothesis and matched comparison

Stage71's low-learning-rate mixture-aware continuation improved a fixed
neural/count predictor by only 0.003202 BPB, but it started after the neural
expert had already learned alone. Test the distinct hypothesis that teaching
the neural backbone to complement a frozen train-only count expert **from
random initialization** creates a larger gain. Both arms use the identical
Stage54 eight-block alternating attention/convolution Transformer, seed 17,
batch 32, independent 256-token train windows, auxiliary losses, R-Drop,
AdamW and the first 2,400 learning rates of Stage54's 7,200-step schedule.
Both use the same fixed Stage25 count checkpoint and 0.0625 count weight for
their complete-validation endpoint score. The count model is never trained.

The control uses Stage54's original neural-only primary NLL. The intervention
replaces only each stochastic pass's primary NLL with the NLL of the
normalized 93.75% neural / 6.25% count distribution. Deep/future-token
auxiliary losses and the neural-to-neural symmetric KL are unchanged. The
count target probability is computed only for training labels, from a frozen
model fitted solely on the supplied train text. The deployed architecture
would be the same single neural backbone plus count expert as before, not a
second neural model.

## Fixed gates and stop rule

1. Verify the fixed count/config hashes, the train/validation files and
   tokenizer; do not read or hash the test file. Unit-test the objective
   substitution and run one batch-32 CUDA forward/backward with finite
   gradients and <=6.5 GiB peak allocated memory. Failure stops the pilot.
2. Run one 2,400-update control and one 2,400-update intervention from the
   same initialization and sampled-window stream. Each presents 19,660,800
   primary train targets. Use the fixed endpoint, not the best intermediate
   checkpoint. Score the identical complete validation split at endpoint,
   using the exact frozen 0.0625 probability mixture for **both** arms.
3. Advance only if the intervention improves the matched control by at
   least **0.020 BPB** and has finite, normalized causal predictions. No
   weight, schedule, seed or step search follows a failed gate. A passing
   pilot licenses only a separately fixed full-training plan, followed by
   exact compact export, full CPU FP32 validation, three fresh resource
   repetitions and clean-extract verification. It does not authorize a
   second test run.

The 0.020 gate is deliberately material relative to the existing ~0.05-BPB
validation gap. This pilot may fail: Stage71's small gain and Stage169's
failed distillation are strong rival evidence. Stage143 remains untouched.
