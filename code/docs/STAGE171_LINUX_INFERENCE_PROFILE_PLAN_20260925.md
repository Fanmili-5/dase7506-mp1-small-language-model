# Stage171: locate the Linux single-thread CPU bottleneck before changing inference

Stage170 found an exact-score but over-limit Linux resource ratio:
67.250413-second candidate median versus 12.221671-second baseline median,
or 5.502555x. The target for a portable 5x pass on that host is at most
61.108354 seconds. This is a 6.142059-second (9.13%) reduction if the
baseline stays fixed. Windows four-thread and Linux one-thread performance
must remain separate observations.

Before editing the frozen Stage143 implementation, time its existing
OpenVINO feature graph, sparse MKN collection, MKN addition, remaining
neural/copy/gate work, and evaluator validity checks on the first eight
independent full-sized validation batches. Use one CPU FP32 thread, warm
once, and record source/checkpoint/graph hashes, row and target coverage,
component call counts and wall times. This is a **partial-validation
runtime profile**, not a BPB selection or a complete resource score.

Use the result to choose one algebraically equivalent inference-only
optimization with a plausible >=9.13% whole-score-loop benefit. Require
full-distribution parity, normalization and causality checks before full
validation, then fresh three-repeat Windows and Linux resource gates.
Do not modify training, tokenizer, evaluator or weights, and do not score
test while searching. If no component has enough cost to justify a safe
optimization, report the Linux risk rather than claiming a universal pass.
