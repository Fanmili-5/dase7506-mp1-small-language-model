# Stage197: core-architecture triage under the remaining deadline

The user wants a true score improvement, not another passing scaffold. The
course permits arbitrary scratch-trained architectures but fixes the tokenizer,
data, evaluator and 5x CPU / 4 GiB RAM / 64 MiB asset limits. Stage143's
complete validation is 1.399686162. The Stage193/195/196 changes to training
objective or frozen residuals each missed their predeclared quality gates.

We considered the following distinct mechanisms before selecting another
run. This table is not a claim that unrun ideas work.

| Candidate mechanism | Relevant local evidence | Decision for this deadline |
| --- | --- | --- |
| More width | Stage147 matched 2,400-step gain 0.01135 | Low priority |
| More depth | Stage150 matched gain 0.01151; Stage155 full worse than Stage143 | Low priority |
| Width and depth, no counts | Stage155 full 1.40988 | Rejected |
| Parallel attention and convolution per block | Stage177 untrained; projected feature CPU near limit | Second option |
| Shared trunk, two heterogeneous upper paths | Stage153 source/preflight exists; Stage154 one-branch pilot gain 0.006 | First option |
| Sparse token-level experts within a shared trunk | Untested; training/routing and export risk | Third option |
| Independent two-model ensemble | Best diagnostic 1.37618 but excessive assets/CPU | Not deployable |
| Teacher distillation into one strong predictor | Stage169/195/196 gains <=0.0012 | Rejected |
| Train-only frequency weighting | Stage161 regressed | Rejected |
| Contextual token-spelling output | Stage160 regressed | Rejected |
| More exact train-text suffix retrieval | Stage146/152 gains <=0.0031 | Rejected |
| Learned causal gate between fixed experts | Stage176 out-of-half 1.40004 | Rejected |
| Stronger dropout or input masking | Stage165/179 regressed | Rejected |
| Different tokenization/external pretraining | Course rules prohibit the needed changes | Not allowed |

The selection uses the problem-first, performance/resource-tension and
composition lenses: the two independent experts are diverse but over budget;
sharing six layers while retaining heterogeneous upper paths may preserve
some diversity without a second full backbone. A high-capacity MoE is less
grounded in this small supplied training corpus and has no completed export
path. The small-scale benefit is uncertain; large-model literature does not
establish BPB gains here. The strongest objection is that sharing the lower
layers and vocabulary head can erase diversity, as Stage154's weak pilot
suggests. A complete-validation full run, compared at matched target count,
will test that objection directly.

## Predeclared Stage197 full-run rule

Use the existing fixed Stage153 width-288, eight-block shared-trunk model:
first six blocks shared, attention→convolution path A and
convolution→attention path B, normalized 50:50 probability mixture, tied
vocabulary head and prefix copy. First verify causal/normalization unit
tests and a synthetic batch-32 CUDA backward under 8 GiB. The existing
Stage153 OpenVINO feature preflight already recorded graph parity, assets and
a 1.250910 candidate/base synthetic feature-time ratio. That missed its
extra 1.25 pilot gate by 0.000910, but the course's actual rule is <=5x
baseline for the complete predictor; this run will not treat synthetic
timing as resource qualification.

If the GPU memory preflight passes, train once from scratch with seed 17,
the unchanged Stage54 R-Drop recipe, batch 32, 7,200 updates, 58,982,400
primary targets, and the exact 7,200-step learning-rate schedule. Use only
supplied train text for gradients; the development loader must read only
train and validation, not test. Score full validation every 300 steps.
Prespecify the last-five average of checkpoints at 6000/6300/6600/6900/7200
and compare it with the endpoint; select the lower complete-validation BPB.
Against matched Stage54 7,200-step evidence, this is a core-architecture
comparison, not a seed search.

If selected complete-validation BPB does not beat Stage143 by at least 0.015,
stop before count rebuilding or CPU export. If it does, rebuild any count
expert from train only, and run exact compact-CPU FP32 validation plus
three-repeat course time/RAM/assets qualification. A score <=1.35 on full
validation remains the goal; an improvement smaller than that is not mission
completion. No test access follows without a new method freeze and explicit
submission decision. Preserve the protected Stage143 checkpoint throughout.
