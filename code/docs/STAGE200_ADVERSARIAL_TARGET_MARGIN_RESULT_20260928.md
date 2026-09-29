# Stage200 adaptive target-margin pilot: stopped

The fixed 2,400-step, seed-17, train/validation-only pilot completed on the
Windows RTX 3070 Ti. Its complete validation score was **1.518770071773339
BPB** over 376,599 targets and 1,148,007 UTF-8 bytes, versus the matched
Stage54 step-2400 control **1.5199503686120217 BPB**. The gain was only
**0.001180296838683 BPB**, far below the pre-outcome admission requirement of
0.0300 BPB (`<=1.4899503686120217`). `admission_passed=false`.

The paper-inspired train-only target margin passed three analytic tests and
one finite CUDA update. The full pilot presented 19,660,800 primary targets,
took 636.13 seconds including validation, and peaked at 5.648 GB allocated /
5.985 GB reserved GPU memory. Its final training-batch mean margin was 0.1051
logit. These are development diagnostics, not inference resource measurements.

The measured variant is rejected. Do not extend it to 7,200 steps, tune its
alpha or seed, export an inference graph, freeze it, or score test. The
protected Stage143 checkpoint and its test result are unchanged. The result
does not establish that all train-only adversarial methods fail; it shows the
fixed adaptation in the [pre-outcome plan](STAGE200_ADVERSARIAL_TARGET_MARGIN_PLAN_20260928.md)
has no convincing gain in this matched setting.

Raw evidence: `../results/stage200-evidence/run.json`, `preflight.json`, and
`metrics.json`. The endpoint checkpoint remains on the Windows training host
under `code/job-logs/stage200-pilot-20260928/endpoint.pt` solely for audit;
its SHA-256 is recorded in `metrics.json`. It is not a release artifact.
