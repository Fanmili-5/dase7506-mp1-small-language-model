# Stage97: fixed-teacher output-LoRA rerun

Stage96 passed the student object to the primary teacher scorer. We initially
suspected the changing LoRA residual affected that scorer. The scorer actually
uses the tied base head, which was frozen, while only the student loss uses the
LoRA output projection. Stage97 verifies this by repeating the same five-pass,
rank-16, batch-32, learning-rate-0.003 recipe and seed 96017 with a separately
loaded Stage71 model. Both Stage71 and Stage76 teachers are explicitly frozen.

The initial adapter exactly reproduced the Stage85 distribution. Stage97's
epoch 0 through 5 BPB values matched Stage96 to within 1e-9 each. The best
validation point remained epoch 0 at 1.4030241601 BPB; epoch 5 scored
1.4034162017. This confirms Stage96's fixed-teacher interpretation and rejects
this particular rank-16 output-LoRA recipe. The old source and results remain
in commit `74862a2`. No adapter was exported, and test remains untouched.
