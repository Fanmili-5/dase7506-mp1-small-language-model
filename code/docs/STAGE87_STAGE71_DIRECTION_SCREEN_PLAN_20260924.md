# Stage87: Stage71 weight-direction screen

Stage86 shows that further gradient updates through the calibrated deployment
objective regress immediately.  Before another costly training design, Stage87
tests whether the useful Stage71 specialization direction was under- or
over-traversed.

For each fixed scalar `alpha` in `0.50, 0.625, ..., 1.50`, the neural state is
`Stage67 + alpha * (Stage71 - Stage67)`.  Alpha 0 and 1 correspond to the
pre-continuation and final averaged experts; the bounded grid tests a single
weight-space direction and changes neither architecture nor inference cost.
Each state is scored through the exact frozen Stage79 calibration and MKN
expert on validation.  Screened weights are not exported.  Only a material
gain over Stage85 permits a local refinement and exact fused qualification.
Test remains untouched.

## Result

The nine-point screen was convex around the existing Stage71 endpoint. Alpha
1.0 scored 1.4030241601 BPB; both alpha 0.875 (1.4031649982) and alpha 1.125
(1.4032513724) were worse, and the farther points degraded monotonically. The
2.7e-8 difference from Stage85 is numerical-path noise, not a gain. No screened
weights were exported and test was not scored. This route is closed.
