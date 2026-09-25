# Stage173 independent Linux four-thread resource reproduction

The course code README demonstrates three alternating fresh-process CPU
comparisons with **four threads**. Stage143 passed this check on Windows, but
an extra one-thread Linux stress check failed at 5.50255 times its same-host
baseline. Neither observation predicts the four-thread Linux ratio reliably.

Before modifying the model or declaring it portable, repeat the supplied
`benchmark_cpu.py` procedure on GitHub's independent Ubuntu x86-64 CPU using
the unchanged baseline and Stage143 checkpoints, three alternating runs, FP32,
the complete validation split and four threads. The existing hash/coverage
auditor checks the median time ratio against five, maximum process RSS against
4 GiB and conservative inference assets against 64 MiB. Archive both the raw
resource JSON and audit, even if the gate fails. Do not score the test split.

This independent host is not guaranteed to be the instructor's CPU, so a pass
would add evidence rather than prove universal compliance. A fail requires
speed work or a different candidate. Do not change Stage143's frozen files.

## Observed result: host cannot run this protocol

The first [attempt](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36149243520)
stopped when the unchanged Stage143 implementation rejected OpenVINO's
reported thread count. The second
[diagnostic run](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36149506005)
measured the cause before repeating the same failure: this hosted runner
exposed **two logical CPUs**, and OpenVINO reported **one inference thread**
for requested counts 1, 2, 3 and 4. The baseline's first four-thread score
completed, but candidate initialization stopped before any candidate timing.
No four-thread ratio, RAM or score comparison exists from this host. This is
neither a four-thread time failure nor a four-thread qualification.

The one-thread Linux 5.50255x measurement remains valid for its own protocol;
the independent Windows four-thread 3.61770x measurement remains valid for its
own host. The course's four-thread reproduction should be repeated on a CPU
that actually provides at least four OpenVINO inference threads. Do not relax
the strict thread guard and silently compare a one-thread candidate with a
four-thread baseline. No Stage143 predictor files changed and no test score
was produced.
