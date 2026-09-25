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
