# Stage205 shared pair input: failed fixed quality gate

The fixed seed-17 Windows pilot completed all 2,400 updates and 19,660,800
primary train-target presentations in 598.91 training seconds. It replaced
Stage204's 8,192-bucket pair lookup with two shared 64-dimensional token
factor tables and one bias-free projection, keeping the Stage54 backbone,
training sampler and first 2,400 learning rates of the 7,200-step schedule.
The completed checkpoint SHA matched the Windows file, every recorded source
hash matched the tracked local source, and Task Scheduler last result was 0.

The predeclared endpoint scored **1.5095703775471685 BPB** on the entire
376,599-target / 1,148,007-byte validation set. Relative to Stage54's
same-target **1.5199503686120217**, the gain is **0.0103799910648532**
BPB, far below the fixed **0.030** continuation gate. It was also
`0.0002653998762434` BPB worse than Stage204's hashed-pair endpoint
`1.509304977670925`. Early checks cannot replace the selected endpoint.
This single-seed result does not establish that factorized interactions are
universally worse, but it gives no basis to spend a full run on this fixed
route. Stop without a rank, scale, initialization or seed sweep; no final
CPU predictor, formal resource qualification, method freeze or new test
score followed.

The prior synthetic/input-only preflight passed with OpenVINO/eager hidden
error `3.81470e-6`, projected assets `57,181,111` bytes and feature-time
ratio `0.949772`. These were feasibility screens only, not resource claims
for a trained deployable checkpoint. The [pre-outcome plan](STAGE205_FACTORIZED_PAIR_PLAN_20260928.md),
preflight, run plan, complete progress and metrics are retained in
[`../results/stage205-evidence/`](../results/stage205-evidence/).
The runner used the train/validation-only loader, never opened or scored
test, and left the protected Stage143 frozen candidate unchanged.

Stages204--205 jointly narrow the adjacent-token input direction: changing
from bucket-specific to shared low-rank factors reduced the extra parameter
count from roughly 2.36 million to 0.28 million, yet both pilots gained only
about 0.01 BPB at the fixed endpoint versus the same control. They do not
close the approximately 0.05-BPB complete-validation gap from Stage143 to
the student's 1.35 aspiration. Stage143's full-test 1.415657617 BPB remains
above the student's 1.38 minimum; neither pilot is a submission score.
