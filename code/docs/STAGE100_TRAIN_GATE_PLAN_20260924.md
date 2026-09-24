# Stage100: train-only causal gate fit

Stage98's low-overhead gate diagnostic scored 1.4000162855 BPB but fitted its
coefficients to validation labels. Stage100 asks whether a gate learned from
supplied training text transfers to validation. A temporary order-5 modified
Kneser-Ney expert uses the first 90% of training tokens. The 90–95% segment
fits the preselected 12-feature linear gate, and the last 5% checks it. The
frozen Stage92 neural model saw all training tokens during its original fit;
this is recorded because its confidence can be optimistic on the gate slices.

The fitted coefficients, mean, and scale are then evaluated on validation
with the fixed full-data order-5 and order-6 count experts. The order-6 expert
maps sixth-order prefixes into the fifth-order indicator so the same learned
gate can be applied without using validation labels. Stage94 calibration is
fixed. A score below 1.4 still requires a serialized inference candidate and
independent 5× CPU, 4 GiB RAM, and 64 MiB asset checks. Test remains untouched.
