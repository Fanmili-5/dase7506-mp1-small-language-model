# Stage 14 resource preflight — not trained-model results

All measurements use random-initialization checkpoints, solely to establish
whether each architecture fits the inference budget before GPU training.
Do not report their BPB as learned-model quality or a leaderboard score.

| Candidate | Median CPU time / baseline | Maximum process peak RSS (bytes) |
|---|---:|---:|
| A: 4 layers, width 320 | 3.254584548 | 1,979,662,336 |
| B: 6x256 with causal prefix copy | 3.903508390 | 1,977,778,176 |
| C: 6x256 with two-component softmax | 4.098455662 | 1,978,777,600 |

Each JSON contains three fresh CPU FP32 validation repetitions per model and
baseline, alternating order. Source/checkpoint hashes, coverage, NLL-to-BPB
arithmetic, median times, peak memory and pass flags have been independently
checked by `scripts/audit_architecture_screen.py`'s resource validator.
This preflight validation has not yet audited trained checkpoints or selected a
winning model. All three candidates were admitted to the already declared
7,200-update screen; the incumbent remains Stage 12.

The corresponding runner additionally counts checkpoint and inference source /
tokenizer assets, totaling 22,437,976 / 21,277,144 / 21,673,320 bytes for A/B/C.
A final selected trained checkpoint must be measured and counted again.
