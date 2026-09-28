# Stage221 fixed preflight result

The predeclared Stage221 random-input preflight completed on the Windows RTX
3070 Ti. Inference graph parity, causal/independent-row checks, 64-MiB
projected assets, and synthetic two-update GPU memory checks passed. The
eight-pair median feature times were 1.2569681 s (Stage143 reference) and
1.60621185 s (shared-depth candidate), a ratio of **1.277846**. This is
above Stage221's fixed **1.25** gate. Consequently Stage221 did **not**
admit training and no Stage221 quality pilot was launched. The exact result
is in `results/stage221-evidence/preflight.json`.

The assignment CPU cap is different: total scorer CPU must be at most 5x
the baseline. Using Stage143's measured 86.0511059 s candidate and
23.7861255 s baseline over the scorer's 46 feature calls, an input-only
projection replaces each reference feature call by the observed additional
0.34924375 s. It predicts 102.1163184 s, or **4.29310433x** baseline.
This is not an actual course resource measurement and cannot override the
failed Stage221 gate. The Stage221 route remains stopped. A separately
predeclared experiment can evaluate the same fixed architecture under a
course-aligned CPU projection, retaining this negative result in the record.
