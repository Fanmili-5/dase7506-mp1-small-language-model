# Stage99: order-6 count expert with distilled Stage92

Stage98's low-overhead cross-fit diagnostic scores 1.4000162855 BPB, very near
1.4. Stage73 established a small order-6 MKN gain against the older Stage67
neural model. Stage99 measures that train-only count extension against the
Stage92 average with the fixed Stage94 calibration and the same weight grid for
both count experts.

This scan uses exact target-only MKN probabilities for validation efficiency.
It does not fit a dynamic gate or prove the CPU, RAM, or asset gates. A positive
result would motivate a separate order-6 gate diagnostic and train-only gate
fitting. Test remains untouched.

## Result

The order-5 reference reproduced 1.4017076623 BPB at weight 0.0625.
With the same Stage92 neural and Stage94 calibration, order-6 MKN selected the
same weight and scored **1.4012884355 BPB**, a **0.0004192268 BPB** gain.
This is enough to justify testing the low-overhead dynamic gate with order six,
but the fixed mixture still misses 1.4. No checkpoint was exported, no resource
gate was run, and test was not scored.
