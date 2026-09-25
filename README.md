# MP1 — Small Language Model Challenge

This repository is a reproducible working scaffold for DASE7506 Project 1. The
course baseline and fixed scoring pipeline are preserved under `code/`; all new
training utilities, model variants, logs, and documentation are additive.

## Current status

**Current resource-qualified development candidate: Stage143 single-copy
OpenVINO + train-only order-six MKN, 1.399686162 CPU FP32 validation BPB.**
An independent Linux x86-64 one-thread CPU FP32 run reproduced
1.399686179 validation BPB; it was a score check, not a Linux resource gate.
Its exact checkpoint passed three alternating fresh-process Windows CPU
measurements at **3.617702x** baseline median time, **2,176,729,088 bytes**
maximum peak RSS and **55,810,412 bytes** of conservative inference assets.
The checkpoint and ONNX graph are tracked in this repository with SHA-256
`256e0e3cd32e23c8ef2551a80ab3b39fd094401c9034853849da35b9963c6da3`
and `5da1de435c86b40c97718e8a5bbc990af9f2eab431cb8a907b36fbc3bcce5ef4`.
See `code/docs/STAGE143_COMPACT_OPENVINO_QUALIFICATION_PLAN_20260925.md`,
`code/results/stage143-evidence/`, and the reproduction instructions in
`code/README.md`. A concise Chinese
[implementation explainer and self-check](code/docs/STAGE143_IMPLEMENTATION_EXPLAINER_20260926.md)
is available for the student's own review; it is not evidence that the student
has already understood the implementation. This is a **validation-only local
Windows qualification**;
the new candidate has not been frozen or scored on test, and passing on the
course CPU is not yet established.
An additional Linux x86-64 one-thread, three-repeat same-host benchmark
found **5.502555x** baseline CPU time, **above** the 5x limit on that host.
Its RAM and asset gates passed. This is a portability risk that must be
resolved or explicitly disclosed; Windows qualification is not a universal
CPU guarantee. See `code/docs/STAGE170_LINUX_RESOURCE_RECHECK_PLAN_20260925.md`.
The attempted independent Linux **four-thread** recheck could not be measured
on that CI host: it exposes two logical CPUs and OpenVINO caps the feature
model at one reported thread, so Stage143's strict thread check stops before
candidate timing. This is not a four-thread failure or pass; see
`code/docs/STAGE173_LINUX_FOUR_THREAD_RESOURCE_PLAN_20260925.md`.
The fixed-graph ONNX Runtime backend probe improved Linux one-thread feature
time by only **5.31%**, below its predeclared 12% advancement gate; it was not
integrated. See `code/docs/STAGE172_LINUX_BACKEND_PREFLIGHT_PLAN_20260925.md`.
Stage187's read-only Linux node profile attributed **9.884/14.790 seconds**
of eight-batch feature runtime to distributed FP32 dense projections. No
single algebraically equivalent local rewrite cleared its plausibility gate;
the Stage143 inference files remain unchanged. See
`code/docs/STAGE187_LINUX_NODE_PROFILE_20260926.md`.
The read-only [pre-test readiness audit](code/results/stage143-evidence/pretest-readiness.json)
rechecked the checkpoint, graph and all counted source-file hashes, three
complete-validation score records, and the three-repeat Windows resource
measurements. It is reproducible with
`cd code && python scripts/audit_stage143_pretest_readiness.py` and does
**not** freeze the method or authorize test scoring by itself.
An independently extracted **pre-test candidate bundle** now contains the
exact 18 counted checkpoint/inference files (55,810,412 uncompressed bytes).
Its SHA-256 is `e3e2765dccfa5859140bcb4ba33fb5a606d1a80e3087f5197ac1b66c32c43da3`.
The transfer hash matched on Windows, and evaluating solely from the fresh
extraction with the fixed course data reproduced **1.399686162** BPB on all
376,599 validation targets. The [bundle audit](code/results/stage143-pretest-bundle-evidence/audit.json)
and packaging script are tracked; the ZIP is a local staging artifact under
`output/pretest/`, **not a frozen or submitted release**. Its manifest pins
code commit `898f37b5c31eb65095bc8bb4d84e3c6a3b9cc4e0`.

`REPORT.pdf` is an **eight-page historical report for the earlier Stage10
checkpoint** (1.605136467 full-test BPB), created before Stage143 existed.
It does not document or qualify Stage143 and must not be submitted as
Stage143's final report. A matching report and full-test score are pending
candidate freeze. The tracked `REPORT_STAGE143_DRAFT.md` has an ignored,
visually checked **three-page development PDF preview** covering experiments
through Stage186. The preview is explicitly watermarked **NOT FOR SUBMISSION**
and contains no Stage143 test score.
The fail-closed `code/scripts/render_stage143_final_report.py` is prepared
for use only after a committed method freeze and matching CPU FP32 full-test
record; it has not yet rendered or replaced `REPORT.pdf`.

The later Stage144 train-suffix coverage diagnostic found little long-history
coverage; Stage145's 2,400-step six-attention pilot beat its matched Stage54
control by only 0.004699 BPB, below its predeclared 0.015 advancement gate;
that pilot also used a shorter learning-rate cycle than the control, so this
is an exploratory rather than strictly architecture-only difference.
Neither produced a qualified replacement or a test score. See the Stage144
triage and Stage145 evidence under `code/docs/` and `code/results/`.
Stage146's exact train-suffix target-probability diagnostic gained at most
0.003037 BPB, below its predeclared 0.01 gate, so no suffix index was built.
Stage147 preserved Stage54's original learning-rate schedule and tested an
8-layer, width-320 four-attention/four-convolution backbone. Its 2,400-step
pilot reached **1.508604645 BPB** against Stage54's **1.519950369** at the
same step; the 0.011346 gain missed its predeclared 0.015 continuation gate.
No long run, CPU qualification or test scoring followed. The sub-1.35 target
is still unmet; see `code/docs/STAGE147_WIDE_GLOBAL_LOCAL_PILOT_20260925.md`.
Stage148's train-only skip-context diagnostic showed a best target-only gain
of **0.003159 BPB**, below its predeclared 0.010 gate. It was not turned into
an inference predictor; see `code/docs/STAGE148_SKIP_CONTEXT_DIAGNOSTIC_20260925.md`.
Stage149's distant co-occurrence expert worsened validation BPB at every
prespecified nonzero mixture weight and was rejected before implementation;
see `code/docs/STAGE149_DISTANT_COOCCURRENCE_DIAGNOSTIC_20260925.md`.
Stage150's ten-block alternating Transformer passed an input-only OpenVINO
resource preflight, but its matched 2,400-step validation result was
**1.508442880 BPB** versus Stage54's **1.519950369**. The 0.011507 gain missed
its predeclared 0.015 full-training gate. It was not promoted or test-scored;
see `code/docs/STAGE150_DEPTH10_HYBRID_PLAN_20260925.md`.
Stage151 located 57,247 validation targets whose train frequency is 100–999
and whose ID is absent from the current window prefix; their mean target NLL
is 4.700063 nats. This is error allocation, not a permitted inference gate.
See `code/docs/STAGE151_ERROR_ALLOCATION_AND_NEXT_MECHANISM_20260925.md`.
Stage152's train-only byte-suffix lookup improved the target-only validation
diagnostic by at most **0.000704 BPB**, far below its 0.010 gate, so it was
not deployed; see `code/docs/STAGE152_BYTE_SUFFIX_EXPERT_DIAGNOSTIC_20260925.md`.
Stage153's two-layer upper branch missed its predeclared feature-speed gate
by a narrow margin and was not trained. Stage154 reduced the extra path to
one heterogeneous final block and passed the input-only resource preflight,
but its matched 2,400-step GPU FP32 validation endpoint was **1.513952503
BPB** versus Stage54's **1.519950369**. The **0.005998** gain missed its
predeclared 0.015 continuation gate. No Stage154 long run, complete CPU
qualification or test scoring followed; see the Stage153/154 plans and
evidence under `code/docs/` and `code/results/`.
Stage155 reallocates the count-expert asset budget to a single 10-layer,
width-320 neural model. Input-only OpenVINO and batch-32 GPU memory screens
passed. Its matched 2,400-step GPU FP32 validation endpoint was **1.497499861
BPB**, **0.022450508** below Stage54 at the same step, clearing its
predeclared 0.020 continuation gate. The fresh 7,200-step run completed:
its endpoint was **1.413182592 BPB**, and the prespecified five-checkpoint
average was **1.409877270 BPB** on complete GPU FP32 validation. Both are
worse than Stage143's 1.399686162, so Stage155 was rejected before compact
export or CPU qualification. No new test result exists.
See `code/docs/STAGE155_NEURAL_BUDGET_REALLOCATION_PLAN_20260925.md`.
Stage156's fixed diagnostic showed that Stage143 and Stage155 have
complementary errors: a non-deployable 50:50 probability mixture scores
**1.376182664 BPB** on complete validation. This is not a qualified or test
score because simultaneous deployment exceeds the asset budget. It cleared
the predeclared gate for considering train-only distillation into one model;
see `code/docs/STAGE156_COMPLEMENTARITY_DIAGNOSTIC_PLAN_20260925.md`.
Stage157 trained a single Stage155-capacity student against this fixed
teacher using only training prefixes. A 900-update pilot improved complete
GPU FP32 validation from 1.409877270 to **1.405148748 BPB**, clearing its
predeclared 0.004 transfer gate, but it remains worse than Stage143 and has
not been CPU-qualified or test-scored. See
`code/docs/STAGE157_TRAIN_ONLY_COMPLEMENTARITY_TRANSFER_PLAN_20260925.md`.
Stage158's one fixed low-LR continuation further improved the student to
**1.403105391 BPB** at step 3,600; the prespecified average was
**1.403214959**. Both remain above Stage143, so the teacher-transfer route
was stopped before CPU qualification or test scoring. See
`code/docs/STAGE158_FIXED_DISTILLATION_CONTINUATION_20260925.md`.
Stage159 tested a compact independent Transformer as a second expert.
Its input-only graph fit the CPU/asset preflight, but its 2,400-step
validation was **1.714011162 BPB** and the fixed 10%/20% mixtures with
Stage143 both worsened BPB. The quality gate failed, so no integrated
two-model predictor or test score was produced; see
`code/docs/STAGE159_COMPACT_COMPLEMENT_PREFLIGHT_20260925.md`.
Stage160 tested a zero-start, context-conditioned ByteLevel-token spelling
residual on the frozen Stage143-equivalent backbone. Its fixed 2,400-update
pilot reproduced **1.399686195 BPB** at step zero but ended at
**1.402679625 BPB**; every intermediate validation was worse than step zero.
It failed the predeclared >=0.005 improvement gate, so no CPU export or test
score followed. See the Stage160 plan and evidence under `code/docs/` and
`code/results/`.
Stage161 kept the same Stage54 architecture and compute budget but gave
training-only extra weight to medium-frequency targets absent from the input
prefix. Initial model predictions matched exactly. Its matched 2,400-step
endpoint was **1.521144191 BPB**, worse than the Stage54 control's
**1.519950369 BPB**, so the predeclared continuation gate failed. No new
inference candidate or test score resulted; see the Stage161 plan/evidence.
Stage162 then tested an intentionally unattainable hindsight upper bound for
selectively using Stage155 in only 30% of independent windows. Even with
validation labels to choose the best windows, complete BPB was **1.388114**;
unbounded oracle selection reached only **1.376080**. Since this misses the
predeclared <=1.33 feasibility gate, no quantized dual-model router was built
or test-scored. See the Stage162 plan and evidence.
Stage163 tested a compact train-only hidden-state nearest-neighbor model,
inspired by kNN-LM, using 100,000 compressed keys. On complete validation,
every prespecified nonzero retrieval mixture worsened Stage143; the best
nonzero setting scored **1.401071171 BPB**. Its quality gate failed, so no
CPU index or test score was produced. See the Stage163 plan and evidence.
Stage164 kept Stage54's initial weights, parameter count, sampled windows
and 2,400-step training schedule while increasing the four causal-convolution
dilations to 1/2/4/8. Its complete-validation endpoint was **1.535790478
BPB**, worse than the matched Stage54 control's **1.519950369 BPB**; all
eight intermediate comparisons were also worse. The predeclared quality
gate failed, so Stage164 was not extended, exported or test-scored. See
`code/docs/STAGE164_DILATED_LOCAL_MIXING_PILOT_20260925.md` and its evidence.
Stage165 then tested training-only 0.05 positionwise input-embedding masking
against the same matched Stage54 control. Its 2,400-step endpoint was
**1.535982945 BPB**, worse than **1.519950369 BPB**; all eight paired
checks were worse. The quality gate failed; no long run, model promotion or
test scoring followed. See the Stage165 plan and evidence.
Stage168 used efficient sharpness-aware optimization on the unchanged
Stage54 Transformer/R-Drop recipe. Its fixed 2,400-step endpoint was
**1.522014969 BPB**, worse than the same-budget **1.519950369 BPB** control,
and training took 1.889x as long. The predeclared gate failed, so no long
run, inference promotion or test evaluation followed. See the Stage168 plan
and evidence.
Stage169 then distilled the same fixed Stage105/Stage155 complementary teacher
into a Stage103/105-equivalent *strong* student, with an identical-start,
identical-train-window hard-label control. Its 900-step complete-validation
endpoint was **1.399440139 BPB**, only **0.000246023** better than Stage143,
while the hard-only control worsened to **1.403915928 BPB**. The prespecified
0.005 gain gate failed, so no continuation, compact export, resource promotion
or test score followed. See the Stage169 plan and evidence under `code/docs/`
and `code/results/`.
Stage174 tested training-only next-token first/last ByteLevel supervision on
the matched Stage54 hybrid model. Its fixed 2,400-step complete-validation
endpoint was **1.520019189 BPB**, slightly worse than the same-target control
**1.519950369**; all eight interim points were also worse. Its predeclared
0.020 continuation gate failed, so no long run, inference asset, resource
promotion or test score followed. See
`code/docs/STAGE174_BYTE_AUXILIARY_SUPERVISION_PLAN_20260925.md` and the raw
records under `code/results/stage174-evidence/`.
Stage175 measured a **hindsight-only**, non-deployable ceiling for the
unchanged Stage143 neural/count experts. Its per-target answer-aware oracle
reached **1.300186476 BPB** on complete validation, while the original
model reproduced **1.399686162 BPB**. A legal gate would need to capture
about half of this oracle gap to reach 1.35; the diagnostic is not a score,
checkpoint or authorization to fit a gate on validation labels. See
`code/docs/STAGE175_CURRENT_EXPERT_ORACLE_PLAN_20260925.md`.
Stage176 then tested whether a 294-feature causal residual MLP gate could
capture that apparent opportunity. Its fixed two-direction, out-of-half
validation diagnostic scored **1.400044877 BPB**, worse than the unchanged
Stage143; one direction improved only slightly and the other regressed. The
predeclared <=1.35 gate failed. No validation-fitted coefficients were
deployed and no test score was made; see
`code/docs/STAGE176_CAUSAL_FEATURE_GATE_FEASIBILITY_20260925.md`.
Stage177 tested a more capable Transformer block that runs local convolution
and global attention in parallel. Structural parity and the projected
61,372,843-byte asset gate passed, but its Windows input-only OpenVINO
feature graph was **1.438569x** the Stage143 feature time, beyond the fixed
1.25x pilot-admission threshold. It was stopped **before training**; there
is no Stage177 validation BPB or deployable checkpoint. See
`code/docs/STAGE177_PARALLEL_GLOBAL_LOCAL_MIXER_PLAN_20260926.md`.
Stage178 reviewed output-layer alternatives before another architecture run:
the matched two-component softmax mixture, untied output matrix, frozen
output residual and contextual spelling head had all failed their respective
quality comparisons. It is a triage record, not a new model or score; see
`code/docs/STAGE178_OUTPUT_HEAD_TRIAGE_20260926.md`.
Stage179 kept the Stage54 architecture, seed, data, schedule and 58,982,400
primary training targets, changing only main dropout from 0.10 to 0.20.
Its prespecified last-five average scored **1.444744045 CPU FP32 validation
BPB**, worse than the same-target Stage54 control **1.428594045**. The
predeclared 0.015 advancement gate failed; no continuation or test score
followed. The full record and auditor are under `code/docs/`, `code/results/`
and `code/scripts/` as Stage179. Stage143 remains unchanged.
Stage180 is an isolated, validation-only optimizer pilot on the same Stage54
model and seed. It replaces AdamW on hidden block matrices with Muon while
retaining AdamW elsewhere, and changes no inference code. Before the run, the
step-2,400 continuation gate was fixed at BPB <= 1.499950369 (at least 0.020
below the matched Stage54 endpoint). It scored **1.534018833** on the complete
validation split, worse than the matched AdamW control **1.519950369** by
0.014068464 BPB; the gate failed, so no long run or inference promotion was
made. The pilot took 972.36 versus 639.37 training seconds through step 2,400
on the same laptop. No Stage180 test score or resource qualification is
claimed. The Muon
algorithm is adapted from [Keller Jordan's MIT-licensed implementation](https://github.com/KellerJordan/Muon);
its license is preserved in `code/third_party/Muon-LICENSE.txt`. See the
prespecified run record and independent auditor under `code/results/stage180-evidence/`.
Stage181's intermediate-layer readout screen and Stage182's matched four-head
pilot missed their fixed advancement gates. Stage183 then tested a fully
normalized geometric correction to the existing neural/MKN mixture, with
all 376,599 validation targets. Its best fixed cell reached **1.398247514
BPB**, only **0.001438648** below Stage143 and far short of its 0.015-BPB
deployment gate. It is a diagnostic, not a resource-qualified model or test
score; see `code/docs/STAGE183_EXPERT_FUSION_DIAGNOSTIC_20260926.md`.
Stage184 then tested one training-only 0.10 block drop-path intervention
against Stage54 at the fixed 2,400-step endpoint. It scored **1.537355105
BPB**, worse than the matched **1.519950369**, and missed the predeclared
0.015-BPB advancement gate. There was no long run or test scoring; see
`code/docs/STAGE184_DROP_PATH_PILOT_20260926.md`.
Stage185 ranks remaining architecture/count hypotheses against the measured
quality gap and the submission deadline; it records a release-readiness
decision, not a new model or test score. See
`code/docs/STAGE185_DESIGN_SPACE_TRIAGE_20260926.md`.
Stage186 tested one quarter-width attention path beside each local block.
Its input-only Windows OpenVINO resource preflight passed, but the matched
2,400-step complete-validation gain was only **0.005196 BPB** versus the
predeclared **0.020** continuation gate. There was no long run, complete CPU
qualification or test score; see
`code/docs/STAGE186_NARROW_PARALLEL_ATTENTION_PLAN_20260926.md`.
Stage188 kept the same neural model and count-table support but rebuilt the
count probabilities from the supplied train text with Witten–Bell smoothing.
It scored **1.414884002 CPU FP32 complete-validation BPB**, worse than
Stage143's 1.399686162, so it failed its fixed 0.015-BPB improvement gate.
No gate refit, formal resource qualification or test score followed; see
`code/docs/STAGE188_WITTEN_BELL_COUNT_SWAP_20260926.md`.

**Historical resource-qualified development candidate: Stage85 calibrated
mixture-aware hybrid, 1.403024133 CPU FP32 validation BPB.**  It retains the
Stage71 neural and frozen train-only MKN experts, applies the closed Stage79
scalar calibration, folds temperature into the final norm, scatters prefix-copy
mass directly into the vocabulary tensor, and omits a redundant final dense
division while bounding FP32 normalization drift.  The exact checkpoint passes
the three-repeat Windows gate at **4.900108x** baseline CPU time,
**2,041,077,760 bytes** peak RSS and **48,570,230 bytes** of conservative
inference assets.  Checkpoint SHA-256 is
`4d8d5a8356c49287985f0efc02f7dc5dddf33de875cbb4c15eddc394d836cffb`.
No test scoring occurred.  See `code/results/stage85-evidence/` and
`code/docs/STAGE85_RESIDUAL_NORMALIZATION_PLAN_20260924.md`.  Stage71 remains
the uncalibrated qualified reference.

The later Stage92 neural-model diagnostic plus
the Stage94 scalar rescan reached **1.401707688 BPB**. It is not resource-qualified
or submission-frozen. Stage90/91 also establish a **1.381623389** over-budget
two-neural ensemble ceiling; Stage92 distillation transfers part, but not all,
of that complementarity. Stages93 and 95 disfavor pure distillation and a wider
balanced student respectively. Stage96's rank-16 output-LoRA recipe also failed
to improve validation. Stage97 independently loaded the primary teacher and
replicated Stage96 within 1e-9 BPB, confirming its teacher was fixed despite
the less clear original code. Stage98's low-overhead validation cross-fit
diagnostic reached **1.400016286 BPB**, but its gate was fitted using validation
labels and is not an exportable candidate. No post-v1 candidate used test for
development selection.
Stage99's train-only order-6 MKN improved the fixed Stage92 mixture to
**1.401288435 BPB**. Stage100's direct train-fitted gate transferred poorly;
Stage101's bounded slope attenuation reached **1.400398787 BPB** with order
six. Stage102's train-learned four-feature gate, with feature mask and slope
scale selected on validation, reached **1.399686163 BPB**. Stage103 exported
this exact predictor and independently reproduced the score, but its CPU ratio
was **6.064540×**, above the fixed 5× limit; RSS and assets passed. It is not
the current qualified candidate. See `code/results/stage102-evidence/`,
`code/results/stage103-evidence/`, and the corresponding Stage102/103 plans.
Stage104 fused copy/count inference reached **5.484419×**; Stage105 merged
sparse feature extraction and count addition into one traversal but reached
**5.512127×** in a separate three-repeat audit. Both kept validation BPB at
1.399686 and passed RAM/asset checks, but neither clears the CPU gate. See
`code/results/stage104-evidence/` and `code/results/stage105-evidence/`.
Stage109's teacher-only continuation of Stage92 regressed to **1.402744180**
BPB after 14.75 million new training targets, so it was not exported.
Stages112–113 tested physical removal and train-only repair of original
Transformer block 5. The repaired seven-block average scored **1.422508959**
validation BPB, so it was not promoted. Stage114's cheaper order-5 count gate
screen reached **1.400224946**; exact Stage115 export reproduced that score but
its one-repeat CPU preflight was **5.169956×**, still over the limit. Test
was not scored for these new candidates. Stage117's train-only no-margin gate
refit improved the cheap variant only to **1.400478201**; the best full gate
remained at **1.400187297**. Exact-output CPU probes found that row-max
precomputation saves only **1.37%**, alternative sparse-add kernels are slower,
and 10–25% FFN channel masks cause substantial quality loss. None was
promoted. See `code/results/stage112-evidence/` through
`code/results/stage119-evidence/`.
Stage120's 22.12-million-target hard-label continuation regressed: its fixed
order-six monitor went from 1.401288462 to 1.401841357 BPB, and its planned
late weight average scored 1.401887300. Stage121's exact static Stage92/order-5
export reproduced **1.401707663** CPU FP32 validation BPB, but its three-repeat
CPU ratio was **5.173329x**, so it failed the time gate despite passing RAM
and assets. Stage85 was the qualified development candidate at that stage.
See `code/results/stage120-evidence/`, `code/results/stage121-evidence/`, and
their experiment documents. No new-candidate test scoring occurred.
Stage122 folded the dynamic gate's fixed affine calibration and cached count
row maxima. Its one-batch CPU probe was 3.196% faster at equivalent outputs,
but the formal three-repeat audit still measured **5.167132x** CPU time for
**1.400224947** validation BPB. It also fails the CPU gate, so no candidate
was promoted. See `code/results/stage122-evidence/`.
Stage123 tested nine prespecified neural temperature/prior combinations under
the same train-fitted order-five dynamic gate. The original 1.125/0.0625
calibration remained best at **1.400224946 BPB**; none reached 1.4. No
checkpoint was exported. See `code/results/stage123-evidence/`.
Stage124 tested a frozen/optimized TorchScript version of the same Stage92
neural feature extractor without changing outputs. Its six-run CPU median was
1.14% *slower* than eager execution, far short of the preregistered 12%
feature-speed gate needed to revisit the sub-1.4 Stage105 predictor. No
checkpoint was exported. See `code/results/stage124-evidence/`.
Stage125 tested a different train-only objective that directly distills the
final neural/count mixture from the Stage91 three-expert teacher. After a
numerical preflight correction and **9.216 million** new training targets,
its best validation score remained the unchanged start (1.401707689 BPB);
the endpoint and fixed average were **1.401806218** and **1.401822708**.
It was not promoted. See `code/results/stage125-evidence/`.
Stage126's ONNX Runtime feature pilot reproduced the Stage92 hidden states
within 6e-6 after an equivalent RMSNorm export rewrite, but its six-run CPU
median was 1.36% slower than eager PyTorch. It did not pass the predeclared
12% speed gate and was not integrated or promoted. See
`code/results/stage126-evidence/`.

**Historical resource-qualified development candidate: Stage27 collapsed
modified-Kneser-Ney hybrid, 1.454390432 CPU FP32 validation BPB.** It combines
the Stage22 deep-supervised Transformer with a train-only order-5 modified
Kneser-Ney expert at fixed validation-selected weight 0.125. Stage27 preserved
all learned tensors/statistics and verified full-validation equivalence after
collapsing the sparse recurrence. Three-repeat Windows CPU ratio **4.970076x**
passes the 5x limit; peak RSS is 2,044,981,248 bytes and conservative inference
assets are 44,210,786 bytes, within 4 GiB/64 MiB. The time margin is only 0.60%
on this machine and is not a portability guarantee. No test scoring occurred.
See `code/results/stage27-evidence/` and
`code/docs/STAGE27_MKN_QUALIFICATION_20260922.md`.

Stage24 is the previous qualified candidate at 1.461804908 BPB, 4.920595x CPU,
2,024,538,112-byte peak RSS and 38,153,954-byte assets. It uses the older
absolute-discount count expert at weight 0.075 and remains the safer timing
fallback. See `code/results/stage24-evidence/` and
`code/docs/STAGE24_STAGE22_QUALIFICATION_20260922.md`.

Stage21 is the previous qualified candidate: collapsed-backoff hybrid,
1.473335240 CPU FP32 validation BPB. Its three-repeat Windows CPU ratio was
4.866243x, peak RSS 2,025,037,824 bytes and total inference assets 38,153,250
bytes. The Stage19 hybrid's weights, count tables, .10 mixture ratio and
training ancestry were unchanged. See `code/results/stage21-audit.json`,
`code/docs/STAGE21_COLLAPSED_BACKOFF_20260922.md` and the Chinese explanation
`code/docs/HYBRID_INFERENCE_EXPLAINED_ZH.md`.

**Qualified fallback: Stage18 H, 1.482379119 CPU FP32 validation BPB**;
three-repeat Windows CPU ratio4.802831x, peak RSS1,989,242,880 bytes and total
inference assets29,678,074 bytes. Its unchanged train-count mixture improves
validation to **1.473335164**, but that original implementation fails CPU time
at **5.437347x** (RAM and assets pass). Stage21 above is the qualified optimized
implementation, not a retroactive change to this result. No new gradient training or test
evaluation. Audit checks raw scoring/resource records, source hashes and exact
equality of all hybrid tensors to H plus the original train-only count tables.
Both checkpoint files are backed up locally with matching SHA256 receipts.
See `code/results/stage19-audit.json` and `code/docs/STAGE19_COMPLEMENTARITY_20260922.md`.

Stage20 preserves that hybrid's tensors, counts and .10 mixture weight while
fusing inference operations. Full-validation numerical equivalence passed;
official CPU FP32 BPB **1.473335240**. Three-repeat Windows CPU ratio improved
to **5.046067x**, but still **fails** the unchanged 5x cap. Peak RSS2,022,481,920
bytes and total inference assets38,147,998 bytes pass. This is an inference
speed improvement, not a learned-quality gain or qualified replacement. Exact
Windows checkpoint and raw evidence are collected and audited; no new training
or test scoring. See `code/results/stage20-audit.json` and
`code/docs/STAGE20_EQUIVALENT_INFERENCE_20260922.md`.

Stage15 completed and passed its full Windows artifact audit (2026-09-21).
**Previous qualified reference: eight-layer prefix-copy F**, which achieved
**1.485094298 CPU FP32 validation BPB**, with a final three-repeat CPU ratio
**4.877905x**, peak RSS **1,988,993,024 bytes**, and conservative inference assets
**29,681,660 bytes**. Timing margin below the 5x cap is narrow. The audit checked
raw scores/resources, source hashes and recomputed the exact last-five weight
averages on Windows. Its real averaged checkpoint is also collected locally.
See `code/results/stage15-audit.json` and `code/results/stage15-evidence/`.

The precision-matched no-copy D averaged 1.547137283. Dual-copy E failed its
resource preflight (6.1833x) and was not trained. Stage14 B remains the archived
qualified 1.499334244 reference. The six-layer 3x-duration G averaged 1.496658491:
only .002676 better than B, below the preregistered .003 resource-retest trigger
and worse than F. Stage15 consumed 294,912,000 new gradient targets. F was the
selected follow-up at Stage15 completion, later superseded as described above.

Stage17 finished on 2026-09-21 at 14:24 UTC. The large teacher **failed** its
fixed quality gate: average CPU validation BPB **1.665982500**. Its best periodic
GPU FP32 validation was 1.521367639 at step4200, then deteriorated while training
loss fell. No CE/KD students were trained. Search cost:117,964,800 gradient targets.
See `code/results/stage17-completed-evidence/`; this rejects the teacher recipe,
not distillation in general. At Stage17 completion the qualified validation
reference remained F's1.485094298; it has now been superseded by H after Stage19.

Stage18 tests two training-only regularizers independently on F: input embedding
row dropout .10 and SwiGLU hidden dropout .20. Same 7200-update target budget,
seed and original trainer as F. Export restores the identical original inference
graph with unchanged weights. Plan: `code/docs/STAGE18_GENERALIZATION_20260921.md`.
Windows job `MP1-stage18-20260921-a` completed at 15:39 UTC on September21.
H average CPU validation BPB **1.482379119**, I **1.482562139**; each used
58,982,400 gradient targets. Both gains fall below the original .003 resource
retest trigger, so neither was resource-qualified by Stage18. H was Stage18's
lowest measured validation score; Stage19 subsequently qualified it. These are small
single-seed gains, not evidence of robust improvement or test performance.
Stage19 separately checked H's resources and a fixed train-count mixture grid
(0/.05/.10), with CPU component profiling and no new training or test scoring.
Plan: `code/docs/STAGE19_COMPLEMENTARITY_20260922.md`.
LSTM/CNN prototypes are untrained and paused. The completed train-count/B hybrid
scored 1.487889082 on validation but has not passed a resource gate; it is a
reserve experiment, not a qualified replacement.

No replacement candidate has received a final test evaluation. The final
course submission will identify one immutable code version and its matching
predictor checkpoint bundle. Historical freezes remain reproducibility records,
not a restriction on replacing the submission candidate. Do not use historical
test results to tune or select a replacement.

### Earlier development milestones

- Framework and configurable student model: implemented.
- Course contract tests and fixed-file verification: implemented.
- Mac CPU and Windows CUDA smoke tests: passed.
- Windows/CUDA setup and independent one-off experiment jobs: verified.
- First equal-target GPU screen: completed on 2026-09-19 (seed 17).
- Three-seed replication and seed-17 component ladder: completed on 2026-09-19.
- Three-seed simplified-model replication and equal-budget 4,800-step comparison:
  completed on 2026-09-19.
- 3.05M-parameter quality/resource gate and 4,800-step capacity run: completed
  on 2026-09-19.
- 3.9M/4.1M/5.25M capacity screen, 5.25M resource gate, and 4,800-step long run:
  completed on 2026-09-19.
- 5.25M matched 3,600-step dropout 0/0.05/0.10 screen: completed on 2026-09-19.
- Dropout 0.15/0.20 boundary screen: completed on 2026-09-19; 0.10 retained.
- Fresh 4,800-step dropout-0.10 duration screen: completed at 1.575606 CPU FP32
  validation BPB; the curve selected its 4,800-step endpoint.
- Frozen 4,800-step recipe replication at seeds 23 and 42: completed; three-seed
  CPU FP32 validation mean 1.579745 and sample standard deviation 0.003591.
- Historical v1 method and checkpoint: seed-17 4,800-step checkpoint selected by the
  preregistered validation rule and frozen by hashes before test evaluation.
- Formal resource gate: passed at 3.485x baseline CPU time, 1.972 GB peak RSS,
  and 21.14 MB of core inference assets.
- Historical v1 full-test result: 1.605136467 BPB versus 2.102014912 for the initial
  baseline. This v1 predictor remains immutable; subsequent development uses
  validation only and requires a separate freeze before any new test call.

Post-v1 development continues on validation only. A two-model probability
mixture reaches 1.559980227 validation BPB on Windows (1.559980123 on Mac),
compared with 1.575605774 for v1. Three fresh Windows CPU runs measure a 4.644x
median-time ratio, 1.985 GB maximum peak RSS, and a 25.21 MB checkpoint. This
passes the measured Windows limits with limited time headroom. The one-repeat
Mac probe failed at 7.046x and remains recorded; passing on Windows does not
establish portability to every CPU. The ensemble is a development candidate.
Stage 12 has now completed one fresh 7,200-step run and all three predeclared
checkpoint averages. Its best candidate averages the last five checkpoints and
reaches **1.547832503 CPU FP32 validation BPB**, compared with 1.551211927 for
the new single endpoint and 1.575605888 for the Windows v1 control. This keeps
single-model inference and improves validation by 0.027773385 BPB. All candidate
checkpoint hashes have been checked against their evaluation JSONs. This is
single-seed development evidence. Its exact last-5 checkpoint passes a fresh
three-repeat Windows resource gate: 3.643x CPU time, 1.979 GB maximum RSS, and
23.24 MB total inference assets. Fixed-recipe seed-23/42 replication was
preregistered but is now deferred in favor of mechanism-led architecture
screening; see `code/docs/ARCHITECTURE_REDESIGN_20260921.md`.
No new test evaluation has been performed.

Stage 14 implements three mechanism-led candidates: 4-layer/320-wide backbone,
causal within-window prefix copy, and a two-component mixture-of-softmax output.
The heads are in `code/student_structured.py`, with a hash-pinned dependency on
the unchanged `student.py`; both source files must accompany their checkpoints.
Resource-gated training is orchestrated by `scripts/run_architecture_screen.py`.
The single-seed screen is now complete. With the same 7,200 updates and fixed
last-five averaging rule, CPU FP32 validation BPB is **1.557700785 (A)**,
**1.499334244 (B)**, and **1.596398956 (C)**, versus the Stage-12 control
**1.547832503**. B is the new validation leader, improving by 0.048498259 BPB;
A and C are not advanced under this recipe. B's exact averaged checkpoint passes
three final CPU repetitions at 3.831x baseline time, 1.976 GB peak RSS and
23.38 MB inference assets. This is single-seed validation evidence, not a new
test score or a frozen submission. Replication and a precision-matched copy
ablation remain pending. The structured heads train in FP32 inside a BF16
backbone, while the historical control also autocasts its output projection;
therefore the gain is for the full variant, not yet an isolated copy-only effect.
See the architecture plan and `code/results/stage14-audit.json` for evidence.

All three random-initialization resource preflights have passed on Windows:
3.255x CPU time for A (wide/shallow), 3.904x for B (prefix copy), and 4.098x for
C (two-component output); each uses about 1.98 GB peak process RSS. Raw evidence
is in `code/results/stage14-preflight/`. These measurements do not establish
trained-model quality or replace the final exact-checkpoint resource gate.
The bounded job completed on 21 September. A completed screen can be audited with
`python scripts/audit_architecture_screen.py --run-dir runs/stage14-architecture-s17
--output results/stage14-audit.json` (one command, run inside `code/`). The audit
requires the original checkpoints and averaging snapshots and rejects partial
results, mismatched scores, changed sources, or incompatible training budgets.

The first CPU FP32 validation results are baseline 2.072081282, RoPE-only
1.922025745, SwiGLU-only 2.011474415, and modern bundle 1.835656528 BPB. Every run
processed 9,830,400 training targets. These are single-seed validation screening
results, not test scores or leaderboard claims. Replication with seeds 17, 23,
and 42 gives mean validation BPB 2.074152 (baseline), 1.921055 (RoPE-only), and
1.840009 (modern). Both candidates beat their paired baseline in all three seeds.
The seed-17 component ladder finds RoPE+SwiGLU at 1.838138, close to the full
modern bundle; not every added component helps. The simplified candidate's
three-seed mean is 1.837202 BPB (sample standard deviation 0.004797). In the
equal-budget 4,800-step seed-17 comparison, baseline reaches 1.754263, modern
1.698374, and RoPE+SwiGLU 1.700166 CPU FP32 validation BPB. Modern is therefore
the primary capacity-search candidate, but its 0.001791 advantage over the
simplified model is too small to claim robust superiority. No final method is
selected. Run `scripts/summarize_stage3.py` inside `code/` to audit the complete
predeclared Stage-3 evidence.

The 3,049,920-parameter scaled modern candidate reaches 1.733712 BPB in the
same 1,200-step screen and 1.660364 at the equal-budget 4,800-step endpoint. Its
validation-selected 4,200-step checkpoint scores 1.658828 on CPU FP32. A repeated
resource check measures 2.477x baseline CPU time, 1.956 GB peak RSS, and roughly
12.35 MB of core inference assets, all inside the course limits. These remain
validation-only development results. Run `scripts/summarize_stage4.py` to audit
the capacity result, selected checkpoint, hashes, curves, and resource gate.

The 5.25M width-256/depth-6 candidate scores 1.690024 at 1,200 steps. In a fresh
4,800-step run, its endpoint is 1.673348 while its validation-selected 3,000-step
checkpoint scores 1.656612 on CPU FP32, 0.002216 below the 3.05M selected control.
The later curve deteriorates, so this remains a small single-seed candidate gain,
not a final selection. Its repeated CPU ratio is 3.645x baseline, peak RSS is
1.972 GB, and core inference assets are 21.14 MB. Run
`scripts/summarize_stage5.py` to audit all hashes, curves, scores, and limits.

At a matched 3,600-step budget, dropout 0, 0.05, and 0.10 score 1.647222,
1.602406, and 1.597758 validation BPB respectively after CPU FP32 verification.
The 0.10 result is the current validation leader, but it is still a single-seed
hyperparameter-selection result rather than a frozen final model. Run
`scripts/summarize_stage6.py` to audit the complete curves and hashes.

Extending that frozen dropout-0.10 recipe to a fresh 4,800-step cosine run
improves CPU FP32 validation BPB to 1.575606, with the best checkpoint at the
endpoint. This is a 0.022153 BPB improvement over the 3,600-step control. The
recipe is now frozen for seed-23/42 replication; no further seed-17 tuning is
permitted. Run `scripts/summarize_stage8.py` to audit the curve, hashes, cost,
and preregistered decision.

Exact-recipe replication gives CPU FP32 validation BPB 1.575606, 1.582028, and
1.581602 at seeds 17, 23, and 42 respectively (mean 1.579745; sample standard
deviation 0.003591). The preregistered lowest-validation rule selects seed 17.
After the pre-test freeze, the official CPU FP32 full-test score is 1.605136467
BPB; the initial baseline scores 2.102014912. This test result was not used to
change the predictor. See `scripts/summarize_stage9.py` and the frozen manifest
in `outputs/final-candidate-20260919/` in the project workspace.

## AI assistance

OpenAI Codex substantially assisted with assignment analysis, model and training
code, tests, experiment planning, Windows deployment, and execution/analysis of
the validation experiments through Stage188,
including the Stage166–173 portability
and runtime checks, the OpenVINO inference
port, compact checkpoint packaging, resource audit, rejected retrieval and
multiscale-convolution/token-masking pilots, the Stage186 parallel-block pilot,
the Stage187 Linux operation profile,
the Stage188 train-only count-estimator comparison,
and report drafting. Human review and
understanding of the implementation are required before submission; they are
not implied by passing automated tests. See `code/docs/AI_ASSISTANCE.md` for the
disclosure and pending review responsibilities.

## Repository map

| Path | Purpose |
|---|---|
| `GUIDE.md` | Supplied assignment specification. |
| `code/model.py` | Untouched official GPT baseline. |
| `code/common.py`, `code/evaluate.py` | Untouched fixed data/evaluation contract. |
| `code/student.py` | Configurable student architecture. |
| `code/configs/student_*.json` | Single-change ablations and exploratory variants. |
| `code/train.py` | Untouched supplied trainer. |
| `code/train_experiment.py` | Reproducible trainer with resume and richer logs. |
| `code/scripts/` | Mac checks, Windows CUDA setup, staged runs, verification. |
| `code/docs/EXPERIMENT_PLAN.md` | Development protocol and freeze policy. |
| `code/docs/WINDOWS_HANDOFF.md` | Exact Mac-to-Windows transfer and CUDA checklist. |
| `code/docs/AI_ASSISTANCE.md` | Required substantive AI-assistance disclosure. |
| `REPORT.pdf` | Historical Stage10 report; **not** the final Stage143 report. |

## Windows RTX 3070 Ti quick start

From PowerShell in `code/`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows_cuda.ps1
.\scripts\smoke_windows_cuda.ps1
```

After the smoke test passes, inspect the generated validation metrics. The first
equal-target experiment batch can then be launched explicitly:

```powershell
.\scripts\run_stage1_windows.ps1
```

Do not run test evaluation during architecture or hyperparameter development.

## Manual commands

Run all tests:

```bash
python -m unittest discover -s tests -v
python scripts/verify_fixed_files.py
```

Train one validation experiment:

```bash
python train_experiment.py \
  --implementation student \
  --config configs/student_rope.json \
  --device cuda \
  --precision auto \
  --steps 1200 \
  --micro-batch-size 32 \
  --run-dir runs/rope-s17
```

Resume the same interrupted run with exactly the same arguments plus `--resume`.
Resume checks model/trainer hashes, precision, PyTorch version and the training
plan. Legacy smoke states from the initial scaffold are intentionally rejected.
`checkpoint.pt` is always the final equal-target endpoint; `checkpoint-best.pt`
is the lowest observed validation-BPB checkpoint, with its true target count.
Do not substitute the best checkpoint into an equal-target endpoint comparison.
Recorded resumed costs accumulate through saved state; work lost after the last
checkpoint in an unexpected crash is not included and must be disclosed separately.

Collect completed run metadata:

```bash
python scripts/collect_results.py
```

## Scientific and submission safeguards

- Training uses only the supplied training text.
- Model and hyperparameter selection use validation only.
- Every formal comparison records seed, processed targets, parameters, timings,
  hashes, optimizer settings, and checkpoint size.
- The baseline comparison and main ablation use the same number of processed
  targets.
- The final CPU FP32 model must remain within 5× baseline scoring time, 4 GiB
  evaluation RAM, and 64 MiB uncompressed inference assets.
- The final method is frozen by source/config/checkpoint hashes before any test
  result is inspected.

See `code/README.md` for the supplied course instructions and exact evaluator
contract.
