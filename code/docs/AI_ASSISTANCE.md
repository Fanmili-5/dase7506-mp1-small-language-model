# AI assistance disclosure

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
