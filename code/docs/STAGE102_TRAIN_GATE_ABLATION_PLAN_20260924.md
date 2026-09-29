# Stage102: bounded gate feature ablation

Stage101 transfers some train-only gate information to validation, but the
order-6 candidate is 0.000399 BPB above 1.4. The Stage100 training fit gave
particularly large coefficients to sparse edge count and several correlated
order indicators. Stage102 keeps the Stage100 training-derived feature means,
scales, and coefficients, then screens nine prespecified ablations or clipping
rules at six fixed slope scales. Only supplied validation selects among these
54 architecture/settings combinations; there is no validation gradient fit.

The full-feature scale-0.1 row must reproduce Stage101 exactly. A positive
candidate still requires checkpoint export and independent CPU, RAM, and asset
qualification. Test remains untouched.

## Result

The full-feature scale-0.1 reference reproduced Stage101 at 1.4003987870
BPB. The best predeclared row retained four training-learned features:
`neural_max_logp`, `neural_margin`, `highest_backoff`, and
`highest_max_mass`. At scale 0.5 it scored **1.3996861632 validation BPB**,
a 0.00071262 gain over Stage101 and 0.00031384 below the development target.
The mean count weight was 0.09349. This is a validation-selected setting, not
a fitted validation gate. Stage103 subsequently exported this setting and
reproduced its score, but failed the 5× CPU gate at 6.064540×; it did not
become a qualified candidate. Test was not scored.
