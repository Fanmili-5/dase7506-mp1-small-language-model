# Stage101: anchored train-learned gate

Stage100 fitted all 12 gate slopes on training text but transferred badly:
the fitted intercept nearly suppressed MKN on validation. Stage101 keeps the
training-learned slopes, replaces the intercept with the fixed Stage94
mixture weight 0.0625, and screens only a ten-value attenuation grid from 0 to
1 on validation. This is bounded setting selection, with no gradient fitting
or continuous coefficient optimization on validation labels. Stage100's train
feature means and scales are fixed throughout.

Both full-data order-5 and order-6 experts are scored. Scale 0 must reproduce
their Stage99 fixed-mixture baselines. A score below 1.4 still needs an exact
inference export and independent CPU, RAM and asset qualification. Test remains
untouched.

## Result

Scale 0 reproduced the fixed-mixture baselines within 0.00000031 BPB. The
best prespecified scale was 0.1 for both experts. Order five scored
**1.4010152457 BPB** and order six scored **1.4003987870 BPB**. The latter is
0.00088965 better than its fixed mixture but still 0.00039879 above 1.4.
Larger slopes over-weighted MKN and degraded rapidly. This is a positive
transfer signal for training-learned relative-confidence features, but no
checkpoint has been exported or resource-qualified. Test was not scored.
