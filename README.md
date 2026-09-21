# MP1 — Small Language Model Challenge

This repository is a reproducible working scaffold for DASE7506 Project 1. The
course baseline and fixed scoring pipeline are preserved under `code/`; all new
training utilities, model variants, logs, and documentation are additive.

## Current status

**Current development/submission candidate: Stage-14 B (causal prefix copy),
1.499334244 CPU FP32 validation BPB.** This candidate replaces v1 as the active
optimization candidate, but has not received a final test evaluation. The final
course submission will identify one immutable code version and its matching
predictor checkpoint bundle. Historical freezes remain reproducibility records,
not a restriction on replacing the submission candidate. Do not use historical
test results to tune or select a replacement.

### Earlier development milestones

- Framework and configurable student model: implemented.
- Course contract tests and fixed-file verification: implemented.
- Mac CPU and Windows CUDA smoke tests: passed.
- Windows/CUDA setup and independent one-off experiment jobs: verified.
- First equal-target GPU screen: completed on 2026-09-19 (seed 17).
- Three-seed replication and seed-17 component ladder: completed on 2026-09-19.
- Three-seed simplified-model replication and equal-budget 4,800-step comparison:
  completed on 2026-09-19.
- 3.05M-parameter quality/resource gate and 4,800-step capacity run: completed
  on 2026-09-19.
- 3.9M/4.1M/5.25M capacity screen, 5.25M resource gate, and 4,800-step long run:
  completed on 2026-09-19.
- 5.25M matched 3,600-step dropout 0/0.05/0.10 screen: completed on 2026-09-19.
- Dropout 0.15/0.20 boundary screen: completed on 2026-09-19; 0.10 retained.
- Fresh 4,800-step dropout-0.10 duration screen: completed at 1.575606 CPU FP32
  validation BPB; the curve selected its 4,800-step endpoint.
- Frozen 4,800-step recipe replication at seeds 23 and 42: completed; three-seed
  CPU FP32 validation mean 1.579745 and sample standard deviation 0.003591.
- Historical v1 method and checkpoint: seed-17 4,800-step checkpoint selected by the
  preregistered validation rule and frozen by hashes before test evaluation.
- Formal resource gate: passed at 3.485x baseline CPU time, 1.972 GB peak RSS,
  and 21.14 MB of core inference assets.
- Historical v1 full-test result: 1.605136467 BPB versus 2.102014912 for the initial
  baseline. This v1 predictor remains immutable; subsequent development uses
  validation only and requires a separate freeze before any new test call.

Post-v1 development continues on validation only. A two-model probability
mixture reaches 1.559980227 validation BPB on Windows (1.559980123 on Mac),
compared with 1.575605774 for v1. Three fresh Windows CPU runs measure a 4.644x
median-time ratio, 1.985 GB maximum peak RSS, and a 25.21 MB checkpoint. This
passes the measured Windows limits with limited time headroom. The one-repeat
Mac probe failed at 7.046x and remains recorded; passing on Windows does not
establish portability to every CPU. The ensemble is a development candidate.
Stage 12 has now completed one fresh 7,200-step run and all three predeclared
checkpoint averages. Its best candidate averages the last five checkpoints and
reaches **1.547832503 CPU FP32 validation BPB**, compared with 1.551211927 for
the new single endpoint and 1.575605888 for the Windows v1 control. This keeps
single-model inference and improves validation by 0.027773385 BPB. All candidate
checkpoint hashes have been checked against their evaluation JSONs. This is
single-seed development evidence. Its exact last-5 checkpoint passes a fresh
three-repeat Windows resource gate: 3.643x CPU time, 1.979 GB maximum RSS, and
23.24 MB total inference assets. Fixed-recipe seed-23/42 replication was
preregistered but is now deferred in favor of mechanism-led architecture
screening; see `code/docs/ARCHITECTURE_REDESIGN_20260921.md`.
No new test evaluation has been performed.

Stage 14 implements three mechanism-led candidates: 4-layer/320-wide backbone,
causal within-window prefix copy, and a two-component mixture-of-softmax output.
The heads are in `code/student_structured.py`, with a hash-pinned dependency on
the unchanged `student.py`; both source files must accompany their checkpoints.
Resource-gated training is orchestrated by `scripts/run_architecture_screen.py`.
The single-seed screen is now complete. With the same 7,200 updates and fixed
last-five averaging rule, CPU FP32 validation BPB is **1.557700785 (A)**,
**1.499334244 (B)**, and **1.596398956 (C)**, versus the Stage-12 control
**1.547832503**. B is the new validation leader, improving by 0.048498259 BPB;
A and C are not advanced under this recipe. B's exact averaged checkpoint passes
three final CPU repetitions at 3.831x baseline time, 1.976 GB peak RSS and
23.38 MB inference assets. This is single-seed validation evidence, not a new
test score or a frozen submission. Replication and a precision-matched copy
ablation remain pending. The structured heads train in FP32 inside a BF16
backbone, while the historical control also autocasts its output projection;
therefore the gain is for the full variant, not yet an isolated copy-only effect.
See the architecture plan and `code/results/stage14-audit.json` for evidence.

All three random-initialization resource preflights have passed on Windows:
3.255x CPU time for A (wide/shallow), 3.904x for B (prefix copy), and 4.098x for
C (two-component output); each uses about 1.98 GB peak process RSS. Raw evidence
is in `code/results/stage14-preflight/`. These measurements do not establish
trained-model quality or replace the final exact-checkpoint resource gate.
The bounded job completed on 21 September. A completed screen can be audited with
`python scripts/audit_architecture_screen.py --run-dir runs/stage14-architecture-s17
--output results/stage14-audit.json` (one command, run inside `code/`). The audit
requires the original checkpoints and averaging snapshots and rejects partial
results, mismatched scores, changed sources, or incompatible training budgets.

The first CPU FP32 validation results are baseline 2.072081282, RoPE-only
1.922025745, SwiGLU-only 2.011474415, and modern bundle 1.835656528 BPB. Every run
processed 9,830,400 training targets. These are single-seed validation screening
results, not test scores or leaderboard claims. Replication with seeds 17, 23,
and 42 gives mean validation BPB 2.074152 (baseline), 1.921055 (RoPE-only), and
1.840009 (modern). Both candidates beat their paired baseline in all three seeds.
The seed-17 component ladder finds RoPE+SwiGLU at 1.838138, close to the full
modern bundle; not every added component helps. The simplified candidate's
three-seed mean is 1.837202 BPB (sample standard deviation 0.004797). In the
equal-budget 4,800-step seed-17 comparison, baseline reaches 1.754263, modern
1.698374, and RoPE+SwiGLU 1.700166 CPU FP32 validation BPB. Modern is therefore
the primary capacity-search candidate, but its 0.001791 advantage over the
simplified model is too small to claim robust superiority. No final method is
selected. Run `scripts/summarize_stage3.py` inside `code/` to audit the complete
predeclared Stage-3 evidence.

The 3,049,920-parameter scaled modern candidate reaches 1.733712 BPB in the
same 1,200-step screen and 1.660364 at the equal-budget 4,800-step endpoint. Its
validation-selected 4,200-step checkpoint scores 1.658828 on CPU FP32. A repeated
resource check measures 2.477x baseline CPU time, 1.956 GB peak RSS, and roughly
12.35 MB of core inference assets, all inside the course limits. These remain
validation-only development results. Run `scripts/summarize_stage4.py` to audit
the capacity result, selected checkpoint, hashes, curves, and resource gate.

The 5.25M width-256/depth-6 candidate scores 1.690024 at 1,200 steps. In a fresh
4,800-step run, its endpoint is 1.673348 while its validation-selected 3,000-step
checkpoint scores 1.656612 on CPU FP32, 0.002216 below the 3.05M selected control.
The later curve deteriorates, so this remains a small single-seed candidate gain,
not a final selection. Its repeated CPU ratio is 3.645x baseline, peak RSS is
1.972 GB, and core inference assets are 21.14 MB. Run
`scripts/summarize_stage5.py` to audit all hashes, curves, scores, and limits.

At a matched 3,600-step budget, dropout 0, 0.05, and 0.10 score 1.647222,
1.602406, and 1.597758 validation BPB respectively after CPU FP32 verification.
The 0.10 result is the current validation leader, but it is still a single-seed
hyperparameter-selection result rather than a frozen final model. Run
`scripts/summarize_stage6.py` to audit the complete curves and hashes.

Extending that frozen dropout-0.10 recipe to a fresh 4,800-step cosine run
improves CPU FP32 validation BPB to 1.575606, with the best checkpoint at the
endpoint. This is a 0.022153 BPB improvement over the 3,600-step control. The
recipe is now frozen for seed-23/42 replication; no further seed-17 tuning is
permitted. Run `scripts/summarize_stage8.py` to audit the curve, hashes, cost,
and preregistered decision.

Exact-recipe replication gives CPU FP32 validation BPB 1.575606, 1.582028, and
1.581602 at seeds 17, 23, and 42 respectively (mean 1.579745; sample standard
deviation 0.003591). The preregistered lowest-validation rule selects seed 17.
After the pre-test freeze, the official CPU FP32 full-test score is 1.605136467
BPB; the initial baseline scores 2.102014912. This test result was not used to
change the predictor. See `scripts/summarize_stage9.py` and the frozen manifest
in `outputs/final-candidate-20260919/` in the project workspace.

## AI assistance

OpenAI Codex substantially assisted with assignment analysis, model and training
code, tests, experiment planning, Windows deployment, and execution/analysis of
the initial validation experiments. Human review and
understanding of the implementation are required before submission; they are
not implied by passing automated tests. See `code/docs/AI_ASSISTANCE.md` for the
disclosure and pending review responsibilities.

## Repository map

| Path | Purpose |
|---|---|
| `GUIDE.md` | Supplied assignment specification. |
| `code/model.py` | Untouched official GPT baseline. |
| `code/common.py`, `code/evaluate.py` | Untouched fixed data/evaluation contract. |
| `code/student.py` | Configurable student architecture. |
| `code/configs/student_*.json` | Single-change ablations and exploratory variants. |
| `code/train.py` | Untouched supplied trainer. |
| `code/train_experiment.py` | Reproducible trainer with resume and richer logs. |
| `code/scripts/` | Mac checks, Windows CUDA setup, staged runs, verification. |
| `code/docs/EXPERIMENT_PLAN.md` | Development protocol and freeze policy. |
| `code/docs/WINDOWS_HANDOFF.md` | Exact Mac-to-Windows transfer and CUDA checklist. |
| `code/docs/AI_ASSISTANCE.md` | Required substantive AI-assistance disclosure. |
| `REPORT.pdf` | Eight-page final report with methods, ablations, resources, limitations, and reproduction details. |

## Windows RTX 3070 Ti quick start

From PowerShell in `code/`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows_cuda.ps1
.\scripts\smoke_windows_cuda.ps1
```

After the smoke test passes, inspect the generated validation metrics. The first
equal-target experiment batch can then be launched explicitly:

```powershell
.\scripts\run_stage1_windows.ps1
```

Do not run test evaluation during architecture or hyperparameter development.

## Manual commands

Run all tests:

```bash
python -m unittest discover -s tests -v
python scripts/verify_fixed_files.py
```

Train one validation experiment:

```bash
python train_experiment.py \
  --implementation student \
  --config configs/student_rope.json \
  --device cuda \
  --precision auto \
  --steps 1200 \
  --micro-batch-size 32 \
  --run-dir runs/rope-s17
```

Resume the same interrupted run with exactly the same arguments plus `--resume`.
Resume checks model/trainer hashes, precision, PyTorch version and the training
plan. Legacy smoke states from the initial scaffold are intentionally rejected.
`checkpoint.pt` is always the final equal-target endpoint; `checkpoint-best.pt`
is the lowest observed validation-BPB checkpoint, with its true target count.
Do not substitute the best checkpoint into an equal-target endpoint comparison.
Recorded resumed costs accumulate through saved state; work lost after the last
checkpoint in an unexpected crash is not included and must be disclosed separately.

Collect completed run metadata:

```bash
python scripts/collect_results.py
```

## Scientific and submission safeguards

- Training uses only the supplied training text.
- Model and hyperparameter selection use validation only.
- Every formal comparison records seed, processed targets, parameters, timings,
  hashes, optimizer settings, and checkpoint size.
- The baseline comparison and main ablation use the same number of processed
  targets.
- The final CPU FP32 model must remain within 5× baseline scoring time, 4 GiB
  evaluation RAM, and 64 MiB uncompressed inference assets.
- The final method is frozen by source/config/checkpoint hashes before any test
  result is inspected.

See `code/README.md` for the supplied course instructions and exact evaluator
contract.
