# Stage72: resource-adjusted balanced hybrid preflight

Stage70's 8x304, three-attention/five-convolution graph missed the CPU gate by
only 0.54% (5.0270x) while passing RAM and asset limits.  Stage72 preserves the
unmeasured global/local allocation hypothesis but removes a small, explicit
amount of dense compute: residual width 300 instead of 304, six attention heads
of dimension 50 instead of eight heads of dimension 38, and SwiGLU675 instead
of SwiGLU684.  Attention remains in layers 1/4/7 and convolution in
2/3/5/6/8.

The complete random-weight graph includes prefix-copy64 and the frozen
collapsed MKN expert at weight .075.  It must pass causal/gradient/export tests
and three fresh-process measurements under CPU <=5x baseline, RSS <=4 GiB and
assets <=64 MiB.  Passing permits one matched 7,200-update R-Drop run; failure
stops the branch without training.  This is targeted resource adjustment, not
a seed search or a quality claim.  Test remains untouched.

## Result

All seven causal/gradient/export tests passed.  The complete random-weight
7,958,701-parameter neural-plus-MKN graph measured **4.712355079x** baseline
CPU time across three fresh-process repetitions, 2,037,936,128-byte peak RSS,
and 48,628,272 conservative inference-asset bytes.  All three course limits
pass, with materially more timing headroom than Stage70.  Stage72 therefore
admits exactly one matched 7,200-update R-Drop quality run in Stage74; this
preflight itself makes no quality claim and did not score test.
