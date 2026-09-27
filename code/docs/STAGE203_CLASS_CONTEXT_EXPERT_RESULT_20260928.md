# Stage203 class-context correction: rejected

The pre-outcome fixed diagnostic completed on Windows CPU FP32. It reproduced
frozen Stage143 at **1.3996861620421424** BPB on all 376,599 validation
targets / 1,148,007 raw UTF-8 bytes, then scored the 64-class correction at
**1.4266257275527654** BPB. The correction **worsened** the score by
0.0269395655106230 BPB. Its loss increased in both fixed halves of the
1,472 independent windows. The predeclared ≥0.030-BPB improvement gate failed.

The SHA-pinned experiment fit token classes and transition tables using the
supplied training text and the frozen Stage143 output matrix only. Its
synthetic causal and normalization checks passed. The projected additional
FP32 asset bytes were 1,069,056, but this was merely a size screen; no
deployable CPU implementation or formal time/RAM qualification was built.
The result is not a test score. Neither the diagnostic nor its runner
tokenized or scored the test text.

Stop this route. Do not tune the number of classes, smoothing or correction
strength on validation after seeing this result. The frozen Stage143
checkpoint, graph, report, and 1.415657616535609 full-test BPB are unchanged;
the student's <1.38 acceptance criterion remains unmet. The exact
pre-outcome plan and implementation were committed as `9bf2697`; raw output
is [`../results/stage203-evidence/result.json`](../results/stage203-evidence/result.json).
