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
