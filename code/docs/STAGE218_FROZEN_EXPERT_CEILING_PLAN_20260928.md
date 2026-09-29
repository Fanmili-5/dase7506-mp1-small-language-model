# Stage218: fixed three-family frozen-expert validation ceiling

The student's objective remains complete validation <1.35 and complete test
<1.38 within 5x CPU, 4 GiB RAM and 64 MiB assets. Stage143 is the best
qualified predictor at 1.399686162 validation BPB. The nondeployable fixed
Stage143/155 half-mixture reaches 1.376182664. Before spending time on
multi-expert compression, ask whether **already-trained** heterogeneous
predictors even have enough combined validation quality to justify it.

No model is trained or selected for deployment in this diagnostic. Cache
complete validation true-target probabilities for the frozen Stage91
Stage71/Stage76/MKN mixture using its **already selected**
0.5375/0.4250/0.0375 weights. Hash-check its three checkpoints, supplied
train/validation files, tokenizer and implementation. Reproduce Stage91's
1.3816233893 BPB over all 376,599 targets / 1,148,007 raw bytes. Use the
existing SHA-pinned Stage143 and Stage155 validation-target arrays; reproduce
their individual and 50:50 scores. Never read or score test.

Before calculation, fix these five additional probability combinations:

| Cell | Stage143 | Stage155 | Stage91 |
| --- | ---: | ---: | ---: |
| A | 0.50 | 0.25 | 0.25 |
| B | 1/3 | 1/3 | 1/3 |
| C | 0.25 | 0.25 | 0.50 |
| D | 0.50 | 0 | 0.50 |
| E | 0 | 0.50 | 0.50 |

These are fixed simple mixtures, **not a global optimum or an inference
model**. The score is calculated by summing the fixed weighted true-target
probabilities and dividing total NLL by `ln(2)*validation_bytes`. The
component models alone already exceed the course's CPU/asset budget when
combined. Their cached target probabilities cannot be deployed because they
use validation labels; only a future full normalized predictor could be
considered, and would need independent CPU/RAM/asset qualification.

Decision: if even the best predeclared fixed mixture is >=1.35 BPB, do not
spend the deadline compressing these specific frozen experts as a route to
the desired validation target. If it is <1.35, that only warrants a separate
feasibility design; it does not authorize training, test access or release.
The already-frozen Stage143 fallback remains protected.
