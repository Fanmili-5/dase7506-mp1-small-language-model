# Stage109: teacher-only second-stage distillation

The resource-qualified Stage85 fixed-weight inference path is already close to
the 5× CPU limit. The Stage92 student has the same inference graph and improved
the fixed calibrated hybrid monotonically from 1.403024 to 1.401736 through
1,800 train-prefix distillation steps, then nearly plateaued. Its 0.75 teacher
cross-entropy / 0.25 next-token NLL objective may leave some of the stronger
heterogeneous teacher's distribution untransferred. Stage109 tests a distinct
mechanism: start from the exact Stage92 four-checkpoint average and continue
with teacher cross-entropy only, using the same frozen 0.55/0.45 Stage71/Stage76
teacher. No validation or test target enters the gradient.

One fixed seed-109017 run uses 2,400 updates, batch 24, AdamW, peak LR 5e-6,
and a predetermined average of updates 1500/1800/2100/2400. The old Stage79
temperature/prior/copy shift and order-5 MKN weight 0.075 are monitoring
constants, not optimization targets. Starting score must reproduce Stage92;
validation at each 300-step boundary diagnoses transfer. No source candidate
is promoted without a fresh calibration, a single serialized checkpoint and
all three independent CPU/RAM/asset gates. Test remains untouched.

## Result

The exact starting point reproduced at 1.4017970557 validation BPB. Every
scheduled check regressed: steps 300/600/900/1200/1500/1800/2100/2400 scored
1.4026051/1.4027019/1.4027480/1.4027137/1.4027698/1.4027547/
1.4027334/1.4027348. The predeclared four-checkpoint average independently
scored **1.4027441799 BPB**, so the original Stage92 average remains better.
The run processed 14,745,600 new training targets in 466.4 seconds of GPU
training, plus validation. No checkpoint was promoted or resource-tested; test
was not scored. Raw `run.json`, `metrics.json`, and average validation records
are in `code/results/stage109-evidence/`. This rejects the teacher-only
continuation, not the original heterogeneous-distillation result.
