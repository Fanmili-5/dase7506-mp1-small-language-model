# Stage96: frozen-backbone output-LoRA distillation

Stage92 shows useful heterogeneous transfer but full-backbone updates absorb
only a small fraction of the teacher advantage. Stage96 freezes the complete
Stage71 backbone, copy route, tied embedding and output bias, and trains only
the existing rank-16 output residual against the same 0.55/0.45 neural teacher.

The exact-zero residual reproduces Stage85. Five deterministic full passes over
supplied training windows optimize 0.75 teacher cross-entropy plus 0.25 hard
next-token NLL. Only 37,376 low-rank parameters train. The residual can later be
folded into one untied vocabulary projection, adding no inference matmul; the
current Stage85 fused path already uses a materialized output projection within
the resource budget. Validation monitors the calibrated neural plus MKN weight
0.075. Test remains untouched.

## Result

The exact-zero start reproduced 1.4030241601 BPB. All five epochs regressed;
the best trained point was epoch 1 at 1.4031641160 and epoch 5 scored
1.4034162016. An audit initially suspected that passing the student object to
the primary teacher scorer caused teacher drift. Inspection of the forward path
showed the scorer uses `neural.head(hidden)`, not the trainable
`student.output_weight()` LoRA projection; the backbone, tied head and bias were
frozen. Stage97 independently loaded and froze Stage71 and reproduced all six
validation scores within 1e-9 BPB. Thus the Stage96 teacher was fixed after
all, and this rank-16 output-LoRA recipe did not improve validation. A broader
claim about all low-rank corrections is not supported. No adapter was exported.
Test was not scored.
