# Stage43: exact within-window successor cache

Stage41 showed that diffuse cosine retrieval over untrained final hidden states
hurts. Stage43 tests a narrower rival explanation: repeated exact phrases within
one article may be useful when the predictor abstains everywhere else.

For each position, the diagnostic incrementally records only previously observed
context-to-successor pairs from the same independent input window. It queries the
longest exact token context up to order eight and enables the cache only when
that order meets a fixed threshold. A predecessor ending at `j<t` contributes
only its observed successor `x_(j+1)`, so no future token or cross-window state
is available. The scan covers six minimum-order thresholds and eight mixture
weights on validation.

Targets are used only to score each fixed normalized cache distribution without
materializing all 2,048 output entries. They do not change the table, match
order, counts or gate. This is a ceiling/screen, not a submission model or CPU
claim. A material gain requires a full normalized implementation, causality
tests and independent resource qualification. No test split is scored.

## Result

The best fixed setting required an exact context match of order at least two and
mixed the local successor expert at weight .10. It covered 45,770 of 376,599
targets (12.1535%) and scored **1.4633075274 BPB**, improving the Stage26 base by
0.0016864506. The signal is positive but smaller than the .003 advancement
standard used for the neighboring architecture experiments, and a deployable
implementation would add arithmetic to an already tight CPU budget. Stage43 is
therefore retained as a diagnostic rather than implemented as a third inference
expert. Evidence and job logs are under `results/stage43-evidence/` and
`results/stage43-job-logs/`; no test split was scored.
