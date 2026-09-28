# Stage223 post-launch related-work correction

After committing the Stage223 fixed plan and starting its Windows pilot,
Codex found two already completed local experiments that should have been
consulted before describing the input-pair path as a new mechanism:

- Stage204 used a hashed previous/current-token embedding. Its fixed
  same-target 2,400-step gain was 0.010645 BPB, below its 0.030 gate.
- Stage205 used shared low-rank token-pair factors. Its corresponding gain
  was 0.010380 BPB, likewise below the 0.030 gate.

Stage223's train-frequency-selected top-16,384 pair table is a different
representation and has a distinct train-derived map, but it belongs to
the **same broader adjacent-token-input family**. The original Stage223
plan and its thresholds are not rewritten. Since the fixed pilot had
already begun, it will run to its predeclared 2,400-step endpoint and use
that endpoint alone. The Stage204/205 negative evidence lowers the prior
plausibility of a large durable Stage223 gain; it is not a license to
switch checkpoints, change table size, or run more pair variants. If
Stage223 misses its 0.030 gate, stop the whole measured pair-input family
for this deadline. This audit used no new test data or score.
