# Stage180: hidden-matrix Muon/AdamW pilot

## Question and fixed decision

Could a different optimizer improve the retained causal Transformer without
changing inference assets? The matched Stage54 architecture, seed 17, R-Drop
loss, train-only sampler, batch 32, context 256 and 7,200-step cosine horizon
were retained. Only hidden block matrices used Muon; all other parameters used
AdamW. A 2,400-step pilot was fixed before execution. Continue only if its
complete-validation BPB at step 2,400 was at least 0.020 lower than Stage54's
1.519950369 (threshold 1.499950369), and the compute cost was acceptable.
No test examples were scored.

## Implementation boundary and reuse

`muon_pilot.py` adapts the five-step Newton--Schulz orthogonalization,
Nesterov momentum and rectangular update scaling from Keller Jordan's
[MIT-licensed Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py).
The original license is in `third_party/Muon-LICENSE.txt`. Token embeddings,
normalization weights, biases and output-related weights remained in AdamW.
The fused QKV hidden matrices were treated as 2D matrices; this is a pilot
implementation choice, not proof that every Muon configuration is ineffective.
No inference code or existing checkpoint was changed.

## Complete-validation result

| Step | Stage54 AdamW BPB | Stage180 Muon/AdamW BPB |
| ---: | ---: | ---: |
| 300 | 1.856828 | 1.855560 |
| 600 | 1.700689 | 1.706610 |
| 900 | 1.638410 | 1.649222 |
| 1200 | 1.594749 | 1.610176 |
| 1500 | 1.570368 | 1.584393 |
| 1800 | 1.547857 | 1.563062 |
| 2100 | 1.534521 | 1.546802 |
| 2400 | **1.519950** | **1.534019** |

Stage180 missed the continuation threshold by 0.034068464 BPB and trailed
the control by 0.014068464 BPB. Training time through step 2,400 was 972.36
versus 639.37 seconds on the same Windows RTX 3070 Ti. The auditor checked
all eight full-validation points, 376,599 targets and 1,148,007 UTF-8 bytes
per point, fixed source hashes, parameter partition and the prespecified
decision. The remote checkpoint SHA-256 was independently matched to the
metrics. The result rejects this specific optimizer configuration, not Muon
in general; no full training, export, CPU resource qualification or test run
followed. Stage143 remains the protected candidate at 1.399686162 validation
BPB.

Evidence: `results/stage180-evidence/run.json`, `metrics.json`, `audit.json`;
script: `scripts/audit_stage180_muon_result.py`.
