# AI assistance disclosure

## Stage190 canonical ASCII-letter pair diagnostic (26 September 2026)

Codex designed, preregistered, implemented and ran a fixed-tokenizer
follow-up to Stage189. It constructed a 35,039-pair noncanonical mask from
all ordered pairs of pure ASCII-letter BPE symbols, checked zero true-pair
incidence on train/validation, and measured one fixed 0.9 adjustment on
complete Windows CPU FP32 validation. The 0.000581-BPB gain missed the
declared 0.015 advancement gate, so Codex stopped without inference
deployment or test scoring. The student must review the mask's causal
construction, empirical-not-universal safety status and negative decision.

## Stage189 tokenizer-pair support diagnostic (26 September 2026)

Codex used a failure-analysis/composition brainstorming framework to propose
a fixed-tokenizer BPE-merge support test. It preregistered the incidence and
0.015-BPB advancement gates, implemented the train/validation-only scripts,
ran them on the Windows host, and interpreted the complete-validation result.
The 1,792 direct merge pairs never occurred as true adjacent targets in the
supplied train or validation token streams. The fixed 0.9 downweight improved
Stage143 by just 0.000590 BPB, below the gate, so Codex did not deploy a new
inference rule or score test. This is substantive AI-led experimental work;
the student should understand why a merge pair need not be universally
impossible across pre-tokenization boundaries and why empirical zero
incidence alone is not a proof for unseen text.

## Stage183 frozen-expert fusion diagnostic (26 September 2026)

Codex proposed, preregistered, implemented and ran a normalized geometric
fusion screen for the unchanged Stage143 neural and MKN experts. The fixed
five-cell grid used complete validation targets only to measure NLL, not to
train an inference parameter. Its best cell improved by 0.001439 BPB,
below the declared 0.015 gate, so Codex stopped without deployment, new
resource qualification or test scoring. The student must review the causal
input-only formula, the zero-count floor, the complete normalization and
the negative decision. This was substantive AI-led experimental work.

## Stages181–182 architecture diagnostics (26 September 2026)

Codex designed, implemented, tested and interpreted a validation-only
intermediate-layer readout screen for the retained Stage143 Transformer.
The readout weights and layer choices were fixed before scoring. Its best
32,768-target gain was about 0.0041 bits per target, below the declared
0.020 gate, so no validation-fitted layer mixture or new inference asset
was promoted. Codex then designed and ran an isolated Stage54-matched
attention-head comparison (eight heads versus four at the same width),
including input-only OpenVINO parity, size and timing preflight before
training. The four-head endpoint was 1.518811476 BPB versus 1.519950369
for the eight-head control, below the declared 0.015-BPB advancement gain.
The pilot stopped without long training, Stage143 replacement or test
evaluation. These are substantive AI-led diagnostics, not proof that other
attention configurations cannot improve the task.

## Stage143 freeze and final-release tooling (26 September 2026)

Codex implemented and tested a read-only method-freeze preflight, a
freeze-gated full-test entry point and a fail-closed final-release packager.
The entry point also supports a SHA-pinned Git-less Windows copy after the
freeze is committed on the Mac. These tools check the recorded inference file
set and hashes, source identity, test-score metadata, report replacement and
Git cleanliness where available. The packager also checks that the fixed
scorer's per-window loss sidecar sums to its reported full-test NLL; synthetic
metadata and loss-sidecar tests exercise rejection paths. Passing these
structural gates is not a full-test result, an originality claim, or a
substitute for the student's review of the implementation. No Stage143
method freeze, complete-test score or final submission bundle had been
created when this disclosure section was written.

## Stage180 hidden-matrix Muon pilot (26 September 2026)

Codex proposed and implemented an isolated, matched 2,400-step optimizer
comparison, using the same Stage54 model, seed, training data, loss, batch,
context and 7,200-step schedule horizon. Hidden block matrices use a small
single-GPU adaptation of Keller Jordan's MIT-licensed Muon implementation
(https://github.com/KellerJordan/Muon); embeddings, norms and auxiliary/output
parameters retain AdamW. The source and license are disclosed in
`code/muon_pilot.py` and `code/third_party/Muon-LICENSE.txt`. Codex wrote the
tests, fixed a 0.020-BPB improvement gate before execution, deployed the
Windows validation-only job and audited the complete-validation outcome.
The pilot scored 1.534018833 BPB versus the matched AdamW control 1.519950369,
so the route stopped without a long run, inference promotion or test score. This is
substantive AI assistance and reused algorithmic work. The student must
understand the optimizer partition, source attribution, comparison and result
before submission.

## Stage168 sharpness-aware optimization pilot (25 September 2026)

Codex used structured failure analysis and idea screening to select one
fixed, training-only SAM experiment for the retained causal Transformer
family. It adapted the general optimizer concept and quarter-batch ascent
from Bahri, Mobahi and Tay (ACL 2022), while writing project-specific code,
tests and a matched Windows runner. Codex preregistered, executed and
audited the 2,400-step comparison. The endpoint was 1.522014969 BPB versus
1.519950369 for the control, so the route stopped without a new inference
candidate or test score. The published T5 fine-tuning results were not
treated as evidence of a gain in this from-scratch task. This is substantive
AI-led design, implementation and interpretation; the student must review
the optimizer, unequal training compute, evidence and negative conclusion.

## Stage164–167 pilots and portability checks (25 September 2026)

Codex proposed, implemented, preregistered, ran and interpreted the matched
Stage164 multiscale-convolution and Stage165 training-only input-embedding
masking pilots. Both regressed against their same-schedule control and were
stopped without inference promotion or test scoring. Codex then installed
the pinned OpenVINO dependency in a local Mac environment, diagnosed a
pre-scoring ARM64 device-discovery abort, built the Linux x86-64 CI check,
identified a runner CPU-thread mismatch, and reproduced Stage143's full
validation BPB with the unchanged checkpoint and scorer. These were
substantive AI-led development and verification steps. The Linux run is a
score reproduction, not a second resource qualification; the student must
review the portability caveats before submission.

## Stage163 compact train-only kNN-LM pilot (25 September 2026)

Codex adapted the train-only datastore/interpolated-neighbor concept from
Khandelwal et al., ICLR 2020 (https://arxiv.org/abs/1911.00172), to a
prespecified 100k-key, 64D-quantized, GPU exact-search validation screen.
Codex designed, implemented, executed and interpreted the experiment; all
nonzero mixture weights worsened Stage143, so the route stopped without a
CPU index or test score. The reused conceptual method and substantive AI
assistance must remain credited in the student's final report/README if
discussed. This is a negative diagnostic, not a deployable predictor.

## Stage162 sparse second-expert oracle (25 September 2026)

Codex designed, preregistered, implemented and audited a validation-only
hindsight upper bound for calling the rejected Stage155 Transformer on a
subset of independent windows. At a 30% window budget the impossible oracle
reached only 1.388114 BPB, and even unbounded selection reached 1.376080.
The predeclared 1.33 gate failed; Codex stopped without a quantized router,
CPU claim or test score. The oracle uses validation labels after prediction
and must never be used as an inference rule. This is substantive AI-led
diagnostic work.

## Stage161 train-only frequency-focus loss (25 September 2026)

Codex used prior error allocation to design, implement, preregister, execute
and analyze a matched Stage54 pilot that increased training weight for
medium-frequency targets absent from the current input prefix. Initial
inference predictions matched the control exactly, but the 2,400-step
complete validation endpoint was 1.521144191 BPB, worse than the matched
1.519950369 control. Codex stopped before full training or test scoring.
This is substantive AI-led experimental work; the student should inspect
the train-only target mask, normalized loss and negative result.

## Stage160 context-conditioned morphology residual (25 September 2026)

Codex proposed, implemented, preregistered and ran a zero-start spelling
residual on a frozen Transformer/count/copy predictor. The synthetic CPU
preflight passed, but the fixed training pilot regressed from 1.399686195
to 1.402679625 complete validation BPB. Codex stopped at the prespecified
quality gate without promoting or test-scoring the candidate. This was
substantive AI-led design, code, execution and analysis; the student should
review the frozen-base boundary and negative result.

## Stage159 compact complementary expert (25 September 2026)

Codex used failure analysis, composition and simplicity checks to propose a
small independent Transformer within Stage143's spare resource budget. It
implemented and ran the input-only OpenVINO parity/speed/asset preflight,
then a fixed matched-schedule training and validation-only mixture pilot.
Although the preflight passed, both nonzero mixtures worsened BPB; Codex
rejected the full-run/deployment route at its predeclared quality gate.
This was substantive AI-led design, code and analysis, not a test score.

## Stage158 bounded teacher-transfer continuation (25 September 2026)

Codex preregistered, implemented, ran and audited one fixed low-learning-rate
continuation from Stage157, with the teacher frozen and training sampler
advanced beyond the pilot. The endpoint improved to 1.403105391 validation
BPB but failed to beat Stage143; the predeclared average was 1.403214959.
Codex stopped this route without CPU promotion or new test scoring. The
student must understand the additional training cost and why the measured
gain did not justify a deployable candidate.

## Stage157 train-only transfer pilot (25 September 2026)

Codex designed a fixed Stage143/Stage155 teacher-to-single-student transfer
experiment and implemented the parity, memory and speed preflight plus a
bounded train-only pilot. The preflight passed, and the 900-step pilot gained
0.004728523 validation BPB without reaching Stage143. This is substantive
AI-generated experiment design, code, execution and analysis. The student
must review the teacher/student separation, training-only labels,
normalization, validation gate and eventual asset boundary.

## Stage156 complementary-error diagnostic (25 September 2026)

Codex predeclared, implemented and ran a hash-pinned complete-validation
diagnostic comparing the retained Stage143 model to the rejected Stage155
average. Their fixed equal-probability mixture scored 1.376182664 BPB, which
passed a threshold for considering train-only distillation. This diagnostic
uses validation target probabilities and is not a deployable, resource-audited
predictor or test result. The student must understand that distinction and
review any later distillation design before submission.

## Stage155 neural-budget reallocation (25 September 2026)

Codex analyzed the limited measured benefit and resource cost of the count
expert, proposed a larger single-neural allocation, wrote the prespecified
feasibility and quality gates, implemented the Windows OpenVINO/GPU screens,
and ran the matched seed-17 pilot and fresh 7,200-step long run. The 2,400-step
validation gain of 0.022450508 BPB authorized the long run; the fixed
last-five average scored 1.409877270 BPB and failed to beat Stage143's
1.399686162. Codex preserved the negative evidence and did not promote,
CPU-qualify or test-score Stage155.
This is substantive AI-led design, code and analysis. The student must
understand the model, the removal of the count expert, the comparison and the
remaining CPU/asset risks before submission.

## Stage146–154 mechanism and architecture screens (25 September 2026)

Codex analyzed Stage143's validation errors, tested bounded train-only
suffix and byte-context diagnostics, implemented and ran matched width,
depth and shared-trunk architecture pilots, and applied prespecified
continuation gates. Stage153 failed its input-only timing screen; Stage154
passed that screen but gained only 0.005998 BPB at the matched 2,400-step
endpoint and was not continued. These were substantive AI-designed and
AI-executed experiments. The student must inspect the source, resource
assumptions, validation evidence and negative decisions before submission.
No new test score was generated.

## Stage144–145 target revision (25 September 2026)

After the student lowered the development target to 1.35 validation BPB,
Codex designed and ran a train-only exact-suffix coverage diagnostic and a
resource-screened six-attention/two-convolution architecture pilot. It
reconnected the existing private Windows tunnel, implemented the experiment
scripts, ran the 2,400-step matched-budget GPU job, compared the complete
validation trajectory with the archived Stage54 control, and rejected the
long run under its predeclared 0.015-BPB advancement gate. This is substantive
AI-led experiment design, coding, execution and interpretation. The student
must review the code, gate rationale, evidence and resource constraints.
Codex then audited Stage143 train/validation error groups and ran a separate,
predeclared validation-target-probability screen for train-only long-suffix
retrieval. The best diagnostic gain was 0.003037 BPB, so the route was rejected
without building a deployable index. No new test result was generated.

## Latest Stage141–143 work (25 September 2026)

Codex screened speed mechanisms for the existing Stage105 predictor, exported
and checked its frozen neural feature extractor as an ONNX graph, implemented
the OpenVINO CPU inference path and compact checkpoint packaging, and ran the
complete Windows validation scorer plus three alternating fresh-process
resource repetitions. It selected Stage143 because the locally measured
1.399686162 BPB, 3.617702x CPU time, 2,176,729,088-byte peak RSS and
55,810,412-byte conservative asset sum satisfy the assignment limits on this
machine. Earlier failed routes and exact evidence remain in the repository.
The student must still inspect and understand the graph/export equivalence,
train-only count tables, copy/gate behavior, resource measurements and
portability risk before submission. This is not a new test score.

OpenAI Codex provided substantive assistance in this project. Its contributions
included analysis of the assignment constraints, design and implementation of the
configurable experiment framework, drafting of model variants and training tools,
test and reproducibility infrastructure, Windows/CUDA setup scripts, and early
documentation and experiment-planning support. On 2026-09-19, Codex also ran the
Windows GPU smoke test and the four-run seed-17 equal-target screen, checked CPU
FP32 validation scores, backed up and hash-verified the checkpoints, and drafted
the initial results analysis. These are development results, not a final model
or leaderboard submission.

Codex subsequently added and executed six seed-replication runs and three
conditional component-ladder runs, developed artifact/recipe/hash auditing and
paired-seed summary tests, and drafted the stage-2 interpretation. Human review
of these implementations, results, and conditional claims is still required.

Codex then preregistered, implemented, remotely executed, backed up, and audited
the stage-3 experiments: two additional simplified-model seeds and three fresh
4,800-step equal-budget runs. It summarized the complete validation curves,
recommended modern as the primary capacity-search candidate while retaining the
simplified fallback, and prepared the next quality-and-resource gate. No test
split or leaderboard result was used for these choices.

Codex also designed and executed the stage-4 capacity gate, including a
3,049,920-parameter short screen, repeated fresh-process CPU time/RAM benchmark,
fresh 4,800-step comparison, and CPU verification of the validation-selected
4,200-step checkpoint. It distinguished equal-budget endpoint evidence from
validation-based checkpoint selection and prepared the next width/depth screen.

Codex then designed and executed the stage-5 width/depth capacity screen,
repeated CPU time/RAM gate, fresh 4,800-step wide-model run, and independent CPU
verification of its validation-selected 3,000-step checkpoint. It implemented a
machine-audited result summary, preserved endpoint-versus-early-stop distinctions,
and preregistered the subsequent matched dropout screen. These decisions used
the supplied validation split only; the test split and leaderboard were not
queried.

Codex next executed and audited the matched 3,600-step dropout 0/0.05/0.10
screen, including independent CPU FP32 endpoint and selected-checkpoint checks.
It identified dropout 0.10 as the single-seed validation leader and
preregistered a bounded 0.15/0.20 follow-up. This remains development evidence,
not a final or test-set claim.

Codex executed the bounded stronger-dropout follow-up, retained both negative
results, and selected dropout 0.10 for one fresh duration comparison before
cross-seed replication. It did not use test results for this decision.

Codex then executed and audited the fresh 4,800-step dropout-0.10 duration run,
verified its endpoint and selected checkpoint independently on CPU FP32, and
preregistered exact seed-23/42 replication of the frozen recipe. It stopped
seed-17 tuning after the duration decision. No test or leaderboard result was
used.

Codex executed the frozen seed-23/42 replication, audited all three seeds,
applied the preregistered validation-selection rule, repeated the formal CPU
time/RAM/asset gate, and generated the pre-test hash manifest. Only after this
freeze, Codex ran the initial baseline and frozen candidate once on the complete
test split, archived the exact outputs, and prepared the checkpoint bundle and
reporting materials. It did not tune or reselect the predictor after observing
the test result.

Human implementation review and explanation are still pending. Before
submission, the student must review the generated implementation, understand the
chosen experiments, and interpret the results. The student remains responsible
for understanding the submitted method, verifying correctness, reporting
failures and costs, and ensuring that all submitted claims are supported by
reproducible evidence. Automated test success is not evidence that this human
review has happened.

This disclosure must be updated before submission to name any additional AI tools
and to describe material changes made after the initial framework was created.

On 21 September, Codex implemented and tested a resource-aware two-model
probability ensemble, validation-only weight scanner, checkpoint packager,
periodic checkpoint preservation, and same-trajectory parameter averaging. It
ran the bounded ensemble screen on validation and recorded both the quality gain
and a provisional Mac CPU time-gate failure. It then preregistered the Windows
resource recheck and a fresh 7,200-step, validation-only trajectory-averaging
experiment. These post-v1 experiments must not use the already observed v1 test
score for model or hyperparameter selection.

Codex subsequently restored the private Windows tunnel, backed up deployed
source, ran all 25 tests and the fixed-file verification on Windows, and executed
the three-repeat ensemble resource gate. It recorded the Windows pass alongside
the Mac failure and launched the preregistered Stage-12 training recipe. It also
added an audit script that checks every candidate checkpoint against the
corresponding complete CPU FP32 validation result before selecting a candidate.

Codex completed the Stage-12 seed-17 run, generated the three predeclared weight
averages, CPU-scored all five candidates, backed up the artifacts, and verified
each checkpoint hash against its score. It selected the last-5 average on
validation, retained every candidate result, and preregistered seed-23/42
replication with the averaging window fixed. No new test score was used or
generated in this stage.

Codex subsequently verified the exact Stage-12 average with three fresh Windows
CPU measurements (3.643x baseline time), collected training/validation error
diagnostics with dropout disabled, and drafted an architecture-first plan.
The student requested mechanism-driven optimization rather than seed screening;
the Stage-13 replication proposal is deferred. Diagnostic group labels are used
only for analysis and are never available as model inputs. No new test was run.

For Stage 14, Codex implemented additive prefix-copy and two-component
mixture-of-softmax heads, a wider/shallower control configuration, contract tests,
and a bounded validation-only runner with CPU resource gates. Designs adapt
ideas from Merity et al. (Pointer Sentinel Mixture Models, arXiv:1609.07843) and
Yang et al. (Breaking the Softmax Bottleneck, arXiv:1711.03953); they are not
claimed as original algorithms or reproductions of those papers' benchmark
scores. The unchanged backbone is hash-pinned. Training and evaluation outcomes
must be recorded separately from implementation and test success.

Codex verified all three Stage-14 CPU preflights from their raw repetitions and
implemented a post-run auditor for checkpoint identities, fixed evaluation
coverage, training budgets, averaging ancestry and resource evidence. The new
auditor was checked with synthetic negative cases and real preflight JSONs; a
complete trained-screen audit remains pending until training finishes. It does
not change any source hash pinned by the running job or invoke test evaluation.

After Stage 14 completed, Codex audited all three runs on the Windows machine,
including final weights, exact training budgets, fixed last-five ancestry and
raw CPU resource repetitions. It identified prefix copy as the single-seed
validation winner (1.499334244 BPB), retained negative A/C results, and prepared
a transfer archive with checksums, endpoint/average weights, scores and logs.
It explicitly recorded that FP32 output-head training also differs from the
historical control, so a precision-matched ablation is needed before isolating
the causal contribution of copy. No further training or test run was launched
while collecting these results.

## Stage 15 optimization assistance

AI assisted a validation/train-only B diagnostic, structured candidate selection,
the FP32 matched no-copy control, a causal dual-copy implementation, its unit tests,
and a resource-gated Windows experiment runner. The new route maps h_j to x_(j+1)
with j<t, so accessible values are already observed inputs. The design draws on
continuous neural cache and pointer-mixture ideas, with sources and adaptation
boundaries in `STAGE15_PLAN_20260921.md`. No paper performance claims are transferred
to this task. Proposed gains are unverified until actual CPU FP32 scores arrive.
The long schedule is separately accounted as extra training, not an equal-budget
architecture comparison. Substantive AI assistance must remain disclosed in the
final README/report, and the student must be able to explain the implementation.

## Stage169 strong-student transfer assistance

Codex formulated the bounded Stage169 hypothesis from earlier Stage156–158
validation evidence, implemented the differentiable Stage103 student and
train-only Stage105/Stage155 teacher preflight, prespecified matched
teacher-versus-hard-label arms, deployed and monitored the Windows run, and
audited the complete-validation negative result. The adapted concept is
knowledge distillation from a fixed teacher probability mixture, not a new
algorithm claim. The student must review the loss, source-only training rule,
the two-arm comparison and its failed continuation gate. No new test score
was generated and Stage143 remains the qualified candidate.

## Stages170–173 cross-host resource and backend assistance

Codex set up an independent Linux same-host baseline/Stage143 resource audit,
profiled the inference bottleneck, predeclared and ran a paired OpenVINO versus
ONNX Runtime FP32 feature comparison, and diagnosed why the hosted two-logical-
CPU runner cannot reproduce the course example's four-thread procedure.
The one-thread Linux time ratio failed at 5.50255x; ONNX Runtime's 5.31%
feature-time gain failed the predeclared 12% advancement gate; the attempted
four-thread Linux run produced no candidate timing. These checks changed no
model weights, frozen Stage143 inference files, validation selection or test
score. The student must understand the host-specific limits before reporting
resource compliance.

## Stage174 training-only byte supervision assistance

Codex selected the first/last ByteLevel auxiliary-prediction pilot from a
failure-analysis shortlist, fixed a same-budget quality gate, and implemented
the additional training-only heads, exact-inference export helpers, structural
tests, training runner and Windows task scripts. The concept is generic
auxiliary multitask supervision applied to the existing Stage54 hybrid model;
it is not claimed as an original language-model architecture. Only the
supplied tokenizer and training targets define auxiliary labels. The student
must understand how the hook captures each causal hidden state and how the
heads are removed for inference. No validation gain, resource qualification
or test score is implied by implementation and unit tests alone. The later
complete-validation pilot slightly regressed against its matched control.

## Stage175 current-expert oracle assistance

Codex distinguished the older Stage22 oracle from the current Stage143
experts, implemented and tested a target-only diagnostic for the frozen
Stage143 neural/MKN pair, ran it on complete Windows CPU FP32 validation,
and archived its reconstruction checks and aggregate result. The 1.300186476
BPB oracle uses validation answers to select an expert at each target and is
impossible to deploy; no validation-fitted gate, new checkpoint, or test score
was created. The student must be able to explain both why this oracle is a
feasibility bound and why its gap cannot be reported as a real model gain.

## Stage176 causal gate feasibility assistance

Codex fixed a two-direction, out-of-half diagnostic before observing its
result, implemented the 294-feature zero-start residual gate and tests,
ran it using the unchanged Stage143 predictor, and archived both negative
held-out directions. Although all inputs to the gate were causal, its
coefficients were fitted to validation answers solely for feasibility
analysis; they are not a legal inference asset or submission result. The
combined 1.400044877-BPB diagnostic failed its 1.35 advancement gate.
The student must understand the distinction between a causal input feature
and a coefficient illegally learned from validation targets.

## Stage177 parallel-mixer preflight assistance

Codex designed the parallel local-convolution/global-attention candidate,
implemented zero-start training and inference modules plus structural tests,
and ran an input-only Windows OpenVINO parity, asset and timing preflight.
The candidate was rejected before gradient training because its feature graph
exceeded the predeclared timing ratio. This is an adaptation of standard
causal attention and gated convolution within the existing Transformer,
not an original architecture claim. The student must understand the
additional branch, its zero-start initialization and the distinction between
feature-only timing and complete submission resource qualification.

## Stages178–179 output-head triage and dropout control assistance

Codex synthesized the existing output-head comparisons and recorded why they
did not justify another mixture-of-softmax or head-only training run. It then
selected, preregistered, implemented, unit-tested, remotely ran and audited one
Stage54-matched dropout-0.20 control, including fixed seed, train-target count,
checkpoint averaging, independent CPU FP32 validation and source hashes. The
candidate regressed and was rejected; no test score or new submission model
was produced. The student must understand the one-variable comparison,
training-only versus inference behavior of dropout, and the failed fixed
advancement gate. This is substantive AI assistance, not evidence that every
regularization method fails.

## Stages180–184 assistance

Codex planned, implemented, ran and audited the rejected Stage180 optimizer
pilot and Stage182 head-count pilot, and diagnosed the Stage181 intermediate
readout and Stage183 normalized expert-fusion alternatives on validation only.
For Stage184, Codex proposed the training-only per-example block drop path,
predeclared its matched control and fixed advancement gate, implemented and
tested the hooks, ran the Windows pilot, and audited the complete-validation
result. The Stage184 intervention made the fixed endpoint worse and was
stopped. These are substantive AI-assisted experiment-design, coding,
execution and interpretation contributions; none establishes a superior
submitted model or uses the held-out test set for development selection.

## Stage185 design-space triage and release preparation

Codex rechecked the assignment boundary and existing local experiments,
organized untested and rejected directions using a structured ideation
framework, updated the archived training-cost audit and Stage143 report
draft, and recorded a deadline-aware decision not to launch an unsupported
GPU sweep. This planning and documentation are substantive AI assistance.
The student must review both the technical ranking and final submission
choice; the triage is not a proof that better methods are impossible.

## Stage186 narrow parallel-attention assistance

Codex designed and preregistered a quarter-width attention path in all four
local blocks, implemented and tested the causal zero-start training/inference
views, ran and audited its Windows input-only OpenVINO resource screen, and
ran the one fixed matched 2,400-step GPU validation pilot. The small
0.005196-BPB early gain missed the predeclared 0.020 advancement gate; Codex
stopped without full training, deployment qualification or test scoring.
The student must understand the added attention branch, zero-start equality,
the limited early gain and the distinction between feature-level feasibility
and complete submission compliance.

## Stage143 Chinese implementation explainer (26 September 2026)

Codex drafted a source-linked explanation and self-check covering the
architecture, causal data flow, training lineage, score/resource evidence and
non-deployable oracle distinction. This is substantive AI-assisted writing,
not proof of student understanding. Before submission the student must check
the claims against code and answer the self-check in their own words.

## Stage187 Linux node-level runtime diagnostic (26 September 2026)

Codex preregistered, implemented and ran a validation-input-only OpenVINO
operation profile on the unchanged Stage143 graph, checked output equality,
archived the raw node timings and rejected a speculative local rewrite when
no single operation had a plausible 12% feature-time saving. This is
substantive AI-assisted profiling and interpretation, not a new model,
complete resource qualification, validation BPB or test score. The student
must understand that the Windows resource pass and Linux one-thread failure
remain separate observations.

## Stage143 final-report rendering gate (26 September 2026)

Codex prepared a fail-closed report renderer that requires a committed
freeze, matching full-test CPU FP32 metadata and complete window-loss
sidecar before replacing the historical Stage10 `REPORT.pdf`. It also
created synthetic metadata-only tests for draft-field substitution and
rejection of template drift. No real Stage143 full-test result or final PDF
was generated by this work. This is substantive AI-assisted submission
tooling and report writing; the student must inspect the final rendered
pages, verify all claims and understand the implementation before posting.

## Stage188 train-only Witten–Bell count comparison (26 September 2026)

Codex proposed and preregistered a same-support count-estimator swap,
implemented the train-only Witten–Bell builder and synthetic unit tests,
verified sparse-table identity and normalization, transferred its candidate
to Windows, ran the unchanged complete CPU FP32 validation scorer and
audited the loss/hash records. The result was 1.414884002 BPB, worse than
the retained Stage143 candidate, so the predeclared quality gate stopped
the route without deployment or test scoring. This was substantive
AI-assisted design, coding, execution and interpretation. The student must
understand that the failure is for this exact frozen-gate swap, not proof
that every count estimator is inferior.

## Stage206 causal linear-memory pilot (28 September 2026)

Codex selected the distinct within-window linear-memory core hypothesis,
predeclared the comparison and stop gates, implemented its model, tests,
export/resource preflight, matched Windows training launcher, and audited
the complete validation endpoint and source/checkpoint hashes. The fixed
candidate regressed against its same-step Transformer control and was stopped
before full training, full CPU/RAM qualification or any new test scoring.
This is substantive AI assistance in model design, implementation,
execution and interpretation. The student must review and understand the
causal prefix-sum computation and the negative quality result before
submission.

## Stage207 explicit word-prefix residual (28 September 2026)

Codex selected and preregistered a causal current-word-prefix feature
hypothesis from the measured residual-error group, implemented the feature
extractor, neural residual, synthetic contracts, resource preflight and
Windows train-only validation pilot, then checked hashes and the negative
complete-validation endpoint. The fixed pilot regressed in both validation
halves and was stopped before full CPU/RAM qualification or test scoring.
This is substantive AI-assisted design, coding, experimentation and
interpretation. The student must understand that the retrospective high-loss
group cannot be detected from the true target at inference.

## Post-Stage207 score-gap audit (28 September 2026)

Codex recomputed the validation-only position distribution of the fixed
Stage143/Stage155 target-probability mixture, checked source-array hashes,
and ranked completed architecture, training, count and lexical experiments
against the student's score and course resource limits. It concluded that
none of the measured nearby variants justifies another blind local sweep.
This is substantive AI-assisted analysis and prioritization, not a new
model score or proof that the target is impossible. The student should
review the evidence and make the final choice of any high-risk new route
or below-threshold fallback submission.

## Stage208–210 lexical hierarchy and backend work (28 September 2026)

Codex selected the jointly trained lexical-tree mechanism, wrote and tested
its normalized causal probability implementation, set stopping gates before
results, and ran the Windows input-only CPU/GPU/asset screens. Stage208's
naive head missed the CPU gate. Codex then implemented an exactly equivalent
level-wise tree propagation (Stage209), verified equality and found that it
still missed the preregistered PyTorch CPU gate. An FP32 OpenVINO head-only
screen (Stage210) passed parity, timing and projected assets; that result
does not establish trained-model quality or full runtime qualification.
Codex prepared the fixed same-target quality pilot. Its first launcher
failed on a missing Windows copy of a committed plan file before training;
the separately logged b run completed the unchanged experiment. Its
complete-validation gain was only 0.002391 BPB versus the same-target
control, below the prespecified 0.030 gate, so no full continuation,
deployment qualification or new test followed. These are
substantive AI-assisted architecture, coding, experiment and interpretation
steps. The student must understand and review the tree probability formula,
causal and resource checks, any later quality result, and final code before
submission.

## Stage211 long-horizon train-only auxiliary pilot (28 September 2026)

Codex used a problem-first research-ideation workflow to propose distant
future-token labels as a training-only mechanism, preregistered the matched
pilot and stop threshold, implemented the config, exact-start tests,
synthetic GPU preflight and Windows training launcher, and audited the
complete-validation result and source hashes. The pilot slightly worsened
BPB, so no full run, deployment qualification or new test followed. This is
substantive AI-assisted experimental design, coding, execution and
interpretation. The student must review and understand why future-token
labels are never visible to the causal inference model and why this
negative pilot does not establish a general impossibility result.

## Stage212 sliding-local-attention feasibility (28 September 2026)

Codex used failure-analysis and architecture-efficiency reasoning to select
one fixed local-attention replacement for the convolution blocks. It wrote
the pre-outcome plan, model, causal/export/gradient tests and Windows
OpenVINO/GPU preflight, fixed an inert config-guard error before measurement,
then audited the raw CPU/parity/asset result. The fixed CPU feature-speed
gate failed, so Codex did not train or score this model. This is substantive
AI-assisted architecture design, implementation, testing and interpretation.
The student must review the seven-position mask, the resource stop decision,
and the fact that the failed preflight says nothing about Stage212 BPB.

## Stage213 course-aligned sliding-attention pilot (28 September 2026)

Codex noticed that Stage212's 1.20x internal feature-speed gate was
stricter than the assignment's <=5x complete CPU rule, calculated an
explicit but unverified 4.818x same-host projection, and documented a
separate Stage213 validation-only experiment before any quality data.
It implemented and tested shared-weight initialization, checkpoint identity,
the matched learning-rate prefix, Windows CUDA screening and the fixed
2,400-step pilot. After collecting the full validation metrics and matching
all source hashes, Codex stopped the regressive model without full training,
formal resource qualification or new test scoring. This is substantive
AI-assisted design, coding, execution and interpretation. The student must
understand why a CPU projection is not a course resource pass and why the
negative pilot cannot be reported as an improved score.

## Stage214 lexical error allocation (28 September 2026)

Codex used a structured research-ideation workflow to select a fixed,
validation-only error decomposition before proposing another architecture.
It implemented and ran the SHA-bound analysis of existing Stage143/155 target
probabilities by token spelling, train frequency and causal-prefix presence.
The measured complement was distributed across target types, so Codex
rejected another near-duplicate lexical-head pilot. This is substantive
AI-assisted diagnostic design, coding and interpretation. The student must
review why the true-target groups are retrospective only, why the over-budget
mixture is not a submission model, and why no <1.35 score was achieved.

## Stage215 train-derived semantic input (28 September 2026)

Codex used a problem-first/composition research-ideation workflow to choose
one fixed co-occurrence input mechanism after the Stage214 lexical audit.
It wrote the pre-outcome contract, implemented the train-only basis and
zero-start Transformer input stream, structural tests, Windows CPU preflight
and scheduled same-target GPU pilot. Codex collected and interpreted the
complete-validation trajectory, independently checked the terminal task,
checkpoint and source hashes, and stopped after the endpoint missed its
predeclared gate. This is substantive AI-assisted architecture design,
coding, execution and analysis. The student must understand the PMI/SVD
construction, the input-only resource screen versus full qualification,
the matched control and the absence of any new test result.

## Stage216 causal relative-distance attention bias (28 September 2026)

Codex proposed a per-head learned causal distance bias as a distinct
context-routing mechanism, wrote its fixed pilot/stop plan before training,
implemented the model and causal/normalization/gradient tests, and ran the
Windows OpenVINO/CPU/GPU preflight and matched 2,400-step train/validation
pilot. Codex verified task completion, the checkpoint hash and 26 source
hashes, then stopped because the complete-validation gain was only 0.002174
BPB versus the preregistered 0.030 gate. This is substantive AI-assisted
architecture design, coding, execution and interpretation. The student must
review the 32-bucket causal mask, why zero initialization gives an exact
control, the distinction between synthetic feasibility and final resource
qualification, and the lack of a new test result.

## Stage217 query-conditioned headwise attention gate (28 September 2026)

Codex identified and read the primary gated-attention paper, then proposed
an exact-zero-start headwise post-SDPA gate as a separate small-corpus
hypothesis rather than claiming reproduction of the paper. It wrote the
fixed pilot/stop plan before training, implemented the model, structural
tests and Windows OpenVINO/CPU/GPU preflight, launched a matched 2,400-step
GPU pilot and audited the terminal task, checkpoint and 26 source hashes.
The 0.004968-BPB complete-validation gain missed the predeclared 0.030 gate,
so no full run, deployment qualification or new test followed. This is
substantive AI-assisted literature triage, experimental design, coding,
execution and interpretation. The student must review the distinction from
the cited paper, the causal gate, zero-start control, resource limitations
and negative quality decision before any submission.

## Stage218 frozen-expert fixed-mixture diagnostic (28 September 2026)

Codex planned and implemented a validation-only diagnostic of three
previously trained predictor families, preregistered five fixed mixtures,
cached exact Stage91 target probabilities from SHA-pinned checkpoints and
supplied development data, and independently verified the result using
different probability arithmetic. The best fixed cell scored 1.371031
BPB but was over the resource budget and missed the <1.35 goal, so Codex
did not promote it, tune nearby weights, or access test. This is substantive
AI-assisted analysis, coding and interpretation. The student must review
why true-target arrays are diagnostic only, why fixed cells are not a
mathematical optimum, and why compression alone cannot be called a score
improvement.

## Stage219 training-budget continuation (28 September 2026)

Codex identified that the larger Stage155 backbone had received far fewer
training-target presentations than Stage143's accepted neural lineage. It
wrote and committed a fixed low-learning-rate continuation plan before the
outcome, adapted the existing Stage56 recipe to a SHA-pinned Stage155
average, reused a train/validation-only loader, launched and monitored the
Windows GPU job, and verified the final checkpoint average, source hashes
and full-validation result. The gain was only 0.003748 BPB, below its
predeclared 0.008 gate, so Codex stopped the route without deployment or
test scoring. This is substantive AI-assisted experimental design, coding,
execution and interpretation. The student must understand why the old
comparison was not equal-budget, why the late improvement was insufficient,
and why this negative result does not prove all larger models fail.

## Stage220 frozen positional-complement diagnostic (28 September 2026)

Codex proposed and committed a fixed validation-only analysis of whether
the known two-model complement concentrates near independent window
starts. It implemented the SHA-bound bin and fixed-position calculations,
verified complete target coverage and gain accounting, and stopped the
short-context helper path when both prespecified gates failed. This is
substantive AI-assisted diagnostic design, coding and interpretation. The
student must distinguish the two-model analysis from a deployable
resource-qualified predictor, and understand why target probabilities
cannot become inference features.

## Stages221–223 and literature triage (28 September 2026)

Codex designed and screened a shared-depth variant (Stage221), separately
predeclared and executed its course-aligned equal-target pilot (Stage222),
and stopped after the fixed endpoint was worse than its control. Codex then
proposed, implemented and ran a train-derived top-bigram input pilot
(Stage223). After launching it, Codex discovered and disclosed the prior
Stage204/205 adjacent-token-input experiments, kept the original gate fixed,
verified the terminal checkpoint/source evidence, and stopped the whole
measured pair-input family when the 0.013196-BPB pilot gain missed its 0.030
gate. Codex searched primary research papers, checked candidate methods
against the course constraints and local negative evidence, and recommended
no new experiment without a genuinely distinct, legal hypothesis. This is
substantive AI-assisted design, coding, execution, literature triage, and
interpretation. The student must review the late related-work correction,
the difference between pilot and full-candidate scores, and the course
legality of any future training-tokenization change. The fixed-hypothesis
planning process was informed by Kassis et al. (2026), *Scientific Agent
Skills: A Library of Procedural Knowledge for Research Agents*,
https://doi.org/10.48550/arXiv.2609.00065. No Stage221–223 run opened test
or displaced the frozen Stage143 candidate.

## Submission documentation (29 September 2026)

Codex reorganized the repository README into a short front page while
preserving the full experiment history, and drafted and typeset a three-page
LaTeX report from the frozen Stage143 evidence. It checked the rendered pages
and retained the explicit score, portability limitation, reused-method
citations, and AI-assistance disclosure. This was documentation work: no
model, tokenizer, evaluator, inference asset, or frozen test score changed.
The student must review the final prose and reproducibility instructions
before publishing them.
