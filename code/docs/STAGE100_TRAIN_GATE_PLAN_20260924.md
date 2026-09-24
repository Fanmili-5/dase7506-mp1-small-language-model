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

## Result

The 90–95% fit contained 180,666 targets and the 95–100% train selection held
180,667. The gate improved selection mean NLL from 2.498390 to 2.474408, but
failed to transfer: on validation it assigned mean MKN weight only 0.00169
instead of the fixed 0.0625. Order-5 BPB worsened from 1.4017076623 to
**1.4051350331**; order-6 worsened from 1.4012884355 to **1.4049710399**.
The full neural model had seen the gate slices whereas the proxy MKN had not.
This training-confidence mismatch is a plausible explanation, not a proven
causal attribution. Stage100 exports no gate or checkpoint. Test was not scored.
