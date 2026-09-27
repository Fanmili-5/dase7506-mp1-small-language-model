# Stage193 fixed-endpoint result and stop decision

The [precommitted plan](STAGE193_FRESH_MIXTURE_AWARE_PILOT_20260928.md) and
training source were unchanged while the Windows runs executed. This is a
train/validation-only result, not a new deployable predictor or test score.

The objective-substitution and development-only loader unit tests passed.
The one-update Windows CUDA preflight had finite loss and gradients, with
5,543,439,360 peak allocated bytes, below the 6.5 GiB gate. Both planned
arms then completed exactly 2,400 updates and 19,660,800 primary train-target
presentations. Their 21 source/data hashes, Stage54 config, Stage25 count
checkpoint, seed, schedule, batch, fixed count weight, and complete-validation
target/byte counts matched. Neither script opened or hashed the test file.

| Arm, both scored with fixed 0.0625 count weight | Complete validation BPB |
| --- | ---: |
| Original Stage54 training objective, matched control | 1.5066285785 |
| Fresh mixture-aware primary objective | 1.5232231254 |

The intervention **regressed by 0.0165945469 BPB**, rather than improving by
the preregistered 0.020 BPB. The complete-validation coverage was 376,599
targets and 1,148,007 raw bytes for each arm. The control trained for about
634.8 seconds and the intervention for about 695.5 seconds, excluding their
~2.6-second target-mixture validation pass. These are pilot training times,
not official CPU inference-resource measurements.

**Stop Stage193.** Do not tune the count weight, learning rate, seed, step
count or auxiliary weights around this negative endpoint. Do not launch the
full 7,200-step continuation, export or qualify the pilot checkpoint, or
score test. This result rejects this fixed fresh mixture-aware objective;
it does not prove every integrated count/neural architecture must fail.
The exact records are `../results/stage193-evidence/preflight.json`,
`../results/stage193-evidence/control.json`, and
`../results/stage193-evidence/mixture.json`.
