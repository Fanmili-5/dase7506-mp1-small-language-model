# Stage39: resource-qualified calibrated hybrid

Stage39 changes only an algebraic association in the bias-free vocabulary
projection: instead of computing `head(hidden) / temperature`, it computes
`head(hidden / temperature)`. The rest of the Stage37 calibrated Transformer,
prefix-copy, unigram prior and collapsed modified-Kneser-Ney recurrence is
unchanged.

Full validation equivalence against the exact Stage37 checkpoint passed before
resource measurement. The maximum absolute log-probability difference was
3.43323e-5, the maximum normalization error was 7.90227e-7, and complete-score
BPB differed by 1.08e-9. The optimized checkpoint SHA-256 is
`294ae321dec8eddc7e4942f759c75f16b3333ea1ecf22b532e13295b1ced3a19`.

Independent CPU FP32 validation scored **1.4464620191 BPB**. Three alternating
baseline/candidate repetitions gave a median candidate-to-baseline ratio of
**4.837137190x**. Peak process RSS was 2,045,800,448 bytes and conservative
uncompressed inference assets were 44,242,609 bytes. All limits pass, so this is
the current resource-qualified validation leader. The large checkpoint is
archived outside Git; the JSON receipts and scheduled-job logs are under
`results/stage39-evidence/` and `results/stage39-job-logs/`.

This result is still 0.0464620191 BPB above the sub-1.4 development target. It is
a safe fallback, not a reason to freeze the method. No test split was scored.
