# AI assistance disclosure

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
