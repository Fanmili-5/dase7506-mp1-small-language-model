# Stage103: exact gated checkpoint and resource audit

Stage102 selects a 1.3996861632-BPB validation setting using four feature
coefficients learned only from training text. Stage103 packages the Stage92
neural expert, train-only Stage73 order-6 MKN statistics, fixed Stage94
calibration, training-derived gate statistics, and the validation-selected
feature mask/scale into one checkpoint. The model receives only each causal
input window and returns a normalized 2048-way distribution. No target or
future-token access enters its forward path.

The first export uses a direct complete-distribution implementation to verify
all 376,599 validation targets. It then measures the exact checkpoint against
the fixed baseline with three alternating CPU runs and checks peak RSS and
conservative inference asset bytes. Any failure is recorded; the checkpoint
is not promoted until all gates pass. Test remains untouched.

## Result

The exported Stage103 checkpoint reproduced the diagnostic at
**1.399686163285 validation BPB**. Three alternating Windows CPU runs gave a
143.926086-second candidate median and 23.732398-second baseline median, a
**6.064540×** ratio that fails the fixed 5× limit. Peak RSS was 2,043,850,752
bytes and conservative inference assets 53,258,957 bytes, both within their
limits. The checkpoint SHA-256 is
`a58011ed846f007b053c7a430837201223bec2d5611480e5a6cf3ba8f97ba7c2`.
This is an exact quality result, not a qualified submission candidate. Raw
records are in `code/results/stage103-evidence/`; test was not scored.
