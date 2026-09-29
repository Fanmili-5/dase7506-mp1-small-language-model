# Training procedure

The downloadable checkpoint can be evaluated without retraining. The commands below repeat the training procedure; they may produce different weights and BPB.

## Run

Use the NVIDIA installation in [code/README.md](code/README.md). Recorded runs used a Windows RTX 3070 Ti with 8 GiB VRAM, CUDA BF16 training and FP32 validation. From `code/`:

```bash
python scripts/reproduce_training.py --run-dir runs/retrained --plan
python scripts/reproduce_training.py --run-dir runs/retrained
```

The first command prints the sequence without training. The second trains both neural branches, averages prescribed checkpoints, builds training-only counts, fits the gate and exports `runs/retrained/predictor.pt`. It never scores test. Subprocesses record their configurations, training targets and input hashes.

The runner stops on a failed subprocess and refuses an existing run directory. An interrupted run can be continued manually from its printed plan and surviving outputs; there is no automatic cross-stage resume. The combined entry point has passed plan and unit checks, but I have not rerun the full training sequence through it.

## Training schedule

Each continuation starts a fresh AdamW optimizer with weight decay 0.1 and gradient clipping at 1.0. The scripts and configs specify the learning-rate schedules, auxiliary-loss coefficients and sampling streams.

| Step | Script in `code/scripts/` | Updates × batch | Checkpoint average |
| --- | --- | --- | --- |
| Initial hybrid | `train_stage54_hybrid_conv_rdrop.py` | 7,200 × 32 | 6,000–7,200, every 300 |
| Continuation 1 | `train_stage56_hybrid_conv_continuation.py` | 4,800 × 32 | 3,600–4,800, every 300 |
| Continuation 2 | `train_stage61_hybrid_conv_continuation.py` | 4,800 × 32 | 3,600–4,800, every 300 |
| Lower auxiliary loss | `train_stage63_primary_emphasis.py` | 3,600 × 32 | 2,400–3,600, every 300 |
| Byte-composed embeddings | `train_stage65_hybrid_conv_byte_rdrop.py` | 3,600 × 32 | 2,400–3,600, every 300 |
| Output-bias fitting | `fit_stage67_output_bias.py` | 5 training epochs | epochs 3, 4, 5 |
| Count-aware continuation | `train_stage71_mixture_aware.py` | 3,600 × 32 | 2,400–3,600, every 300 |
| Alternate teacher | `train_stage74_balanced_hybrid_rdrop.py` | 7,200 × 32 | 6,000–7,200, every 300 |
| Teacher continuation | `train_stage76_balanced_hybrid_continuation.py` | 4,800 × 32 | 3,600–4,800, every 300 |
| Distillation | `train_stage92_heterogeneous_distillation.py` | 1,800 × 24 | 900, 1,200, 1,500, 1,800 |

Each window presents 256 primary targets. Deep-supervision, future-token and R-Drop losses add work beyond those primary counts. Byte features are folded into tied embeddings before output-bias fitting. The teacher mixes primary/alternate branches at 0.55/0.45; distillation uses 0.75 soft-target and 0.25 hard-target loss. Only the resulting student is deployed.

The runner passes `--fresh-run` to continuation, count and gate scripts so they accept newly trained inputs instead of requiring the original checkpoint hashes and BPB. They still check the data, configuration, state shapes and target coverage.

## Counts, gate and export

`build_kneser_ney.py` builds pruned order-five MKN from training text; `build_stage73_order6.py` adds order-six counts (minimum count 2). Gate fitting uses a proxy count model from the first 90% of training tokens, the next 5% for fitting and the last 5% for fitting-hyperparameter selection. The neural model has seen these segments; they are not an independent neural holdout. No validation/test labels fit gate coefficients.

Settings selected on validation:

- vocabulary temperature 1.125, training unigram prior weight 0.0625, copy-logit shift 0.25;
- features: neural maximum log-probability, neural top-two margin, highest-order MKN backoff and maximum continuation mass;
- count-weight anchor 0.0625, gate slope scale 0.5, bounds 0.0001–0.9999.

`export_retrained.py` exports a single-pass PyTorch predictor and checks normalization and equivalence with its unfused reference on synthetic inputs. This output differs from the downloadable OpenVINO checkpoint. Measure its validation BPB, CPU time and memory use separately; the reported measurements apply to the downloadable model.

## Controlled comparisons

From `code/`:

```bash
python scripts/train_stage26_multi_token.py --run-dir runs/control
python scripts/train_stage47_rdrop.py --run-dir runs/rdrop
```

Both use seed 17, 7,200 updates, batch 32 and the same sampled-window stream: 58,982,400 primary-target presentations. Use `average_checkpoints.py` on steps 6,000, 6,300, 6,600, 6,900 and 7,200, followed by `export_multi_token.py` or `export_stage47_rdrop.py` and CPU validation. R-Drop has two stochastic forward passes: equal targets do not mean equal FLOPs. [RESULTS.md](RESULTS.md) links the measurements and describes the comparison limits.
