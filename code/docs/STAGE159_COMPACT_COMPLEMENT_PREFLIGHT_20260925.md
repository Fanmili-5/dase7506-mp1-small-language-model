# Stage159: independent compact Transformer complement under the spare budget

Stage158 showed that a full Stage155-capacity student captures only part of
the Stage143/Stage155 teacher's complementary errors: it stopped at
1.403105391 BPB versus Stage143's 1.399686162. The teacher's 1.376182664
cannot itself be deployed: two large graphs exceed the asset and CPU limits.

Failure-boundary, composition and simplicity checks suggest one different
structural test. Retain **the entire exact Stage143 predictor** and add one
*independently trained, very small* 4-layer width-128 Transformer expert.
Independent features may preserve the error diversity that direct student
distillation lost, while the compact graph could fit Stage143's remaining
**11,298,452-byte** asset margin and roughly 33-second CPU-time margin.
This is a potential incremental leaderboard improvement, not a credible
standalone route from 1.3997 to below 1.35: even the much stronger
Stage143/Stage155 teacher is 1.3762. We will reject it quickly if the
resource or complementarity gate fails.

The candidate keeps the fixed BPE vocabulary and causal 256-token windows,
but has four blocks with attention at 1/3 and gated convolution at 2/4,
width 128, four heads, and the existing prefix-copy/R-Drop training recipe.
No training text outside the supplied corpus and no test split is used.

First export a **random-weight input-only** FP32 OpenVINO feature graph and
check exact eager/graph hidden parity <=3e-4. Time eight interleaved
four-thread batch-32 feature calls against the frozen Stage143 graph;
require median candidate feature time <=0.35 times the reference. Conservatively
reserve 1,500,000 bytes for the compact head/new source and require
Stage143's 55,810,412 assets + new graph + reserve <=64 MiB. These are
preflight thresholds only, not complete CPU/RAM eligibility. If any fail,
do not train. If they pass, predeclare a 2,400-step matched-schedule pilot
and use complete validation target probabilities at fixed mixture weights
0, 0.1 and 0.2 to test whether the small expert actually complements
Stage143; a full predictor and resource audit would still be required.

## Input-only preflight result and fixed pilot gate

The random-weight graph is **4,068,213 bytes**. Its maximum trained-independent
eager/OpenVINO hidden discrepancy was **1.43e-6**; eight-pair four-thread
feature medians were 0.194057 s for the compact expert and 1.254391 s for
the Stage143 reference, a **0.154702x** ratio. Stage143's measured assets
plus this graph and the 1.5-MB head/source reserve total **61,378,625 bytes**,
below 64 MiB. All predeclared input-only gates pass. Raw timings and hashes
are in `../results/stage159-evidence/preflight.json`. They are not a complete
predictor or CPU/RAM qualification.

Authorize one seed-17 2,400-update pilot with Stage54's *first 2,400 of 7,200*
learning rates, sampler, effective batch 32, context 256 and R-Drop/auxiliary
weights; only the compact architecture changes. At its endpoint, score the
complete GPU FP32 validation target stream and the three fixed Stage143/
compact probability weights **0 / 0.1 / 0.2**. Require the best nonzero
mixture to improve Stage143 by at least **0.005 BPB** on all 376,599 targets
before paying for a full train or integrated inference path. The diagnostic
may gather true validation target probabilities only for loss measurement;
the predictor must never use the true target to choose a mixture or gate.
If the threshold fails, preserve the negative result and stop. No test scoring.

## Pilot result and rejection

The scheduled Windows job completed all 2,400 updates and **19,660,800**
primary train-target presentations in 313.24 training seconds. Its endpoint
complete GPU FP32 validation was **1.714011162 BPB** over 376,599 targets.
The fixed target-probability diagnostic reproduced Stage143 at 1.399686162
and gave **1.403317245** at compact weight 0.1 and **1.412771929** at 0.2.
The best nonzero cell **worsened** BPB by 0.003631083, far from the
predeclared 0.005 improvement gate. This diagnostic is not a deployable
predictor; the absence of a positive validation mixture signal already
rejects a full run. No trained graph export, integrated CPU resource audit,
promotion or test scoring followed. The raw run, fixed grid, source hashes
and task logs are in `../results/stage159-evidence/`.

The result distinguishes feasibility from utility: a ~1-M-parameter
independent expert fits the spare time/asset budget in input-only tests but
is too weak at the matched pilot point to supply useful complementary mass
at the predeclared weights. It does not prove that all compact experts or
longer schedules fail; those would be new experiments, not this one.
