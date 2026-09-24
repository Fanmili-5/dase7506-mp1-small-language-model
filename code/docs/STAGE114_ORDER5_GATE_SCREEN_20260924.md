# Stage114: five-order count expert with train-fitted gate

The six-order count model and dynamic gate improved validation to 1.399686 BPB
but exceeded the 5× CPU limit. Stage114 asks whether the cheaper original
five-order count model can retain enough of that gain. It fixes the Stage92
neural model, Stage94 calibration, Stage100 train-fitted gate coefficients,
and Stage25 full-training-data order-five count model. It screens the same
eight feature masks as Stage110 at slope scales 0.2–0.7. The original fixed
order-five 0.0625 mixture must reproduce 1.401708 BPB. Validation selects
only mask and slope scale; no validation gradient is used. Any score under 1.4
still needs exact export and independent CPU/RAM/assets qualification. Test
remains untouched.

## Result

The fixed 0.0625 order-five reference reproduced at **1.401707662 BPB**.
The best of the predeclared rows retained all four causal gate features at
slope scale 0.4 and scored **1.400224946 BPB**. No row was below 1.4.
Thus a cheaper count order nearly recovers the gate benefit, but is not yet
a qualifying improvement. Stage115 exports the best setting only for an exact
CPU resource preflight; its one-repeat timing is not formal qualification.
All 48 screen rows are in `code/results/stage114-evidence/result.json`.
Test was not scored.
