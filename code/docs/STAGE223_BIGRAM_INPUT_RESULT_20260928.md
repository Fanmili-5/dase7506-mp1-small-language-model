# Stage223: fixed pair-input pilot stopped

The predeclared Windows seed-17 pilot completed all 2,400 updates and
19,660,800 primary next-token target presentations. Its fixed endpoint was
**1.5067547843836253 BPB** over all 376,599 validation targets and
1,148,007 bytes. The same-step Stage54 control was **1.5199503686120217**,
so the gain was **0.0131955842283964 BPB**, below the predeclared **0.030**
continuation gate. The terminal task status was `completed`, with no error;
the pilot checkpoint SHA-256 was
`5d199717b9421d00e1210b7dea42211e6c9e7a358758c44f995b91123884b44b`.

The preflight only projected resource feasibility from a synthetic graph. It
did not qualify a trained, complete CPU-FP32 scorer. No full 7,200-step run,
new test score, or candidate replacement follows this result. Stage143 remains
the protected resource-qualified candidate: validation **1.399686162** and
already frozen test **1.415657617** BPB, above the student's 1.35 target and
1.38 minimum.

The [post-launch prior audit](STAGE223_POSTLAUNCH_PRIOR_AUDIT_20260928.md)
acknowledged that Stage204 and Stage205 had already tested distinct forms of
the same broad adjacent-token-input idea, with gains of 0.010645 and 0.010380.
Stage223's 0.013196 gain is real at its fixed endpoint but does not justify
another pair representation, table-size, seed, or schedule search. Stop this
measured pair-input family for the deadline. These observations do not prove
that every possible pair model fails.

Evidence: [metrics](../results/stage223-bigram-pilot-metrics.json),
[fixed run record](../results/stage223-bigram-pilot-run.json), and
[terminal status](../results/stage223-bigram-pilot-status.json). The run
record pins source and data hashes. No test data were opened by this pilot.
