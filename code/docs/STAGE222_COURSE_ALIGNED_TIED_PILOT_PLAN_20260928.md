# Stage222: course-aligned re-admission of unchanged shared-depth candidate

Stage221 failed its self-imposed feature-time gate of 1.25x and remains
closed. The observed 1.277846x feature ratio nonetheless projects to
4.29310433x total baseline CPU, under the assignment's 5x limit. This
**new** experiment does not revise Stage221's outcome. Its sole purpose is
to determine whether the exact same, already fixed architecture has enough
quality gain to warrant the expense of an actual full resource check.

Use the Stage221 preflight record unchanged, with configuration SHA-256
`6838cee0b9148ddd090e24988abd514ff66c889178fd6359504924f0b47cd785`
and graph SHA-256
`2116a21aff879a8bbd172b1ff5fd43a258a210e32d7eb779d160319ef0774ea2`.
Require all structural, graph-parity, projected-asset, and synthetic-GPU
checks already recorded there, plus a course-aligned projected **total**
CPU ratio <=4.5, leaving 0.5x baseline for projection error. Compute the
projection solely from that preflight's two fixed medians and Stage143's
frozen total scorer timings; do not inspect validation or test to select the
architecture or revise this threshold. This gate admits only a pilot; an
exact course resource run would still be mandatory before release.

The matched pilot is exactly Stage221's fixed seed-17, 2,400-step,
batch-32 training stream with the first 2,400 learning rates of the
7,200-step Stage54 schedule. It makes no seed, data, tokenizer, model,
sharing, or objective change. Score the complete validation split at
every 300 steps for diagnostics, but decide only on the **2,400-step**
endpoint. The Stage54 matched endpoint is 1.519950368612022 BPB over
376,599 targets. Require a gain >=0.030 BPB (candidate <=1.489950368612022)
to justify a separately predeclared full 7,200-step run; otherwise stop
the route. This pilot has no access to test. Even a gain >=0.030 would not
establish either the student's <1.35 goal or their <1.38 minimum. The
Stage143 fallback and test artifacts remain untouched.
