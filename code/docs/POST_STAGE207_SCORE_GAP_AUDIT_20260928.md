# After Stage207: score-gap and design-space audit

This is a **decision audit**, not a new model, validation score, freeze or
submission. The student wants complete-validation BPB <1.35 and considers
complete-test BPB <1.38 the minimum acceptable result. The frozen Stage143
checkpoint (`256e0e3c...b9963c6da3`) remains the best qualified candidate:
complete Windows CPU FP32 validation **1.399686162042141**, already-frozen
complete test **1.415657616535609**, CPU-time ratio **3.617702x**, lifetime
peak RSS **2,176,729,088 bytes**, and inference assets **55,810,412 bytes**.
Thus the observed validation gap to 1.35 is **0.049686162 BPB**. The test
gap to the student's 1.38 floor is **0.035657617 BPB**; that already-seen
test number is an acceptance status, **not a target for selecting a new
method**. The Windows resource pass does not erase the separate Linux
one-thread 5.502555x observation.

## Problem-first decomposition

The high-loss medium-frequency/no-prefix group in Stage151 is real but
defined with the true target. Stage191's train-only ASCII word-prefix table
was active on 300,020 validation targets. A later read-only decomposition
of the same frozen Stage143 validation target-probability stream found that
the 35,977 active positions whose actual successor was unseen in that train
prefix table have mean target NLL 5.794766 and contribute 0.261994 BPB.
Those membership labels are unavailable at inference. Stage207 added the
observed prefix legally as a causal neural feature, but regressed by
0.001636 BPB in both fixed validation halves. This rejects its exact frozen
residual, not all word/character models.

The over-budget Stage143/Stage155 50:50 teacher scores 1.376182664 on
complete validation, still 0.026183 above 1.35. Its advantage is spread
throughout independent 256-token windows, not confined to first-position
failures. Recomputed from hash-pinned Stage162 target arrays:

| Causal position in window | Targets | Mean mixture gain, nats/target | Contribution to total BPB gain |
| --- | ---: | ---: | ---: |
| 0–15 | 23,552 | 0.033365 | 0.000988 |
| 16–63 | 70,615 | 0.048163 | 0.004274 |
| 64–127 | 94,144 | 0.049234 | 0.005825 |
| 128–255 | 188,288 | 0.052477 | 0.012417 |

This is retrospective target-probability arithmetic, **not** a deployable
ensemble or evidence that a longer-memory model will achieve the gain.
Sources: `../results/stage162-evidence/stage143-target-logp.npy` SHA-256
`75dc47298d03f807dd47e1a44a8d1bdb31d128631225acad273a6e528d78a6e3`
and `stage155-target-logp.npy` SHA-256
`0cbe8ee4aec517ba756fc5e167a95aea2b82e088b5f870e29f77337a9a499603`.
The formula is `logaddexp(logp143, logp155)-log(2)`; sum its per-target
improvement over Stage143 within each position bin, then divide by
`ln(2)*1,148,007` validation bytes. The four gains sum to 0.0235035 BPB,
matching the fixed mixture. No test target array is used.

## Candidate filter after the recent pilots

| Mechanism class | Most relevant completed evidence | Decision now |
| --- | --- | --- |
| Wider/deeper single Transformer | Stage155 full average 1.409877, worse than Stage143 despite matched early gain | No nearby width/depth sweep |
| Heterogeneous two-network probability mixture | 1.376183 validation but >64 MiB and projected 7.26x CPU; still >1.35 | Not deployable as is |
| Shared-trunk heterogeneous paths | Stage197 full average 1.426863 | Stop that architecture |
| Distill the complementary teacher | Stage158 1.403105; Stage169 endpoint 1.399440 with only 0.000246 gain | No repeat of the same transfer |
| Sparse MoE FFN | Stage48 full 1.474591 versus matched dense 1.464994 | No nearby router/expert-width sweep |
| Parallel/reordered local-global core | Stage199 early gain 0.005609; Stage202 regressed | Below material early gate |
| Rank-16 causal linear-memory replacement | Stage206 early 1.535373 versus matched 1.519950 | Stop rank/layer tuning of this mechanism |
| Adjacent-token input features | Stage204/205 early gains only 0.010645/0.010380 | Stop pair-feature family |
| Contextual spelling/observed word-prefix residual | Stage160 1.402680; Stage207 1.401323 versus 1.399686 | Stop frozen lexical-head variants |
| Sparse/count/word-prefix experts | Exact suffix <=0.0031; Stage188 1.414884; Stage191 fixed mixture worsened; Stage203 1.426626 | No nearby count-calibration grid |
| Causal expert-gate features | Stage176 out-of-half 1.400045; Stage198 gain 0.000830 | Oracle 1.300 is label-aware and unusable |
| Regularization/optimizer tweaks | Stage168 SAM, Stage179 dropout20, Stage180 Muon, Stage200 margin missed gates | No seed or rate search |

The filter is not a proof that <1.35 is impossible. It says that **none of
the measured nearby variants has produced a deployable gain of the needed
order**. The remaining materially different concepts are jointly trained
word/character hierarchy, a fully normalized autoregressive byte-to-BPE
decoder, or an efficient heterogeneous exact-attention core. Each is
unmeasured; each has a substantial implementation or CPU/asset risk, and
the failed Stage160/197/206/207 tests lower its prior plausibility. It
would be misleading to launch one as if it were already a projected
<1.35 candidate.

## Decision and release boundary

Do not run another near-neighbor seed, rank, head, layer, mixture-weight or
dropout scan. If the student supplies a concrete new mechanism or chooses
one of the three high-risk directions, fix its train-only causal contract,
input-only CPU/asset screen, matched-target pilot and material-quality
gate **before** training. A passing pilot still needs full CPU FP32
validation and all three resource measurements; only a frozen method may
touch test. Keep Stage143 hashes and fallback package intact. Do not publish
the student's ID/score or submit a below-1.38 fallback without their
explicit decision. The task remains unfinished.
