# Stage17: Transformer capacity at training time, compact inference

## Completed outcome

The job completed at 2026-09-21 14:24 UTC with `completed_teacher_rejected`.
CPU FP32 validation of the fixed last-five teacher average: **1.665982500**,
above the locked1.482094298 gate. Neither student trained. The best periodic
GPU FP32 validation was1.521367639 at update4200; the endpoint deteriorated to
1.679397928 while training loss declined. Do not switch averaging/selection
post hoc or use this teacher for distillation. New gradient cost117,964,800;
teacher forward-only distillation cost0. Raw scoring, training curves and
diagnostic evidence: `results/stage17-completed-evidence/`.

The launch record and original plan below are retained as historical evidence.
Stage18 addresses generalization on the smaller qualified F model instead.

## Original launch and preregistration

Launch status (historical): **teacher training launched on Windows at 2026-09-21 13:16:29 UTC**.
Stage15 completed and passed full artifact audit before launch. The runner
refuses to start before Stage15 finishes successfully.
No LSTM/CNN training, seed sweep, external weights/text, or new test evaluation.

Local verification on 2026-09-21: 71 tests, 67 passed and four CUDA-only tests
skipped; all ten official fixed-file hashes unchanged. The seven distillation
tests also passed on Windows, including CUDA BF16 backward. The teacher's real
two-update smoke and exact-plan resume passed. No final Stage17 score yet.
Launch receipt: `results/stage17-launch.json`; immutable startup evidence is in
`results/stage17-launch-evidence/`. This one-off Windows job survives SSH
disconnects, but not Windows sleep/shutdown/logoff; it has a six-hour runtime cap.

## Why this experiment

Stage14 B (6x256 prefix-copy) achieved 1.499334244 CPU FP32 validation BPB.
Stage15 F (8x256, equal 58,982,400 targets) achieved 1.485094298 and passed
the final same-checkpoint resource gate at 4.877905x baseline, 1,988,993,024-byte
peak RSS and 29,681,660-byte conservative inference assets. Its CPU margin is
small. D, the precision-matched no-copy ablation, averaged 1.547137283; this
supports the copy component in this single seeded comparison. G, a 3x-duration
6x256 run, finished at 1.496658491 after averaging, worse than F. Its .002676
gain over B did not reach the fixed .003 resource-retest trigger. The locked
reference for Stage17 is F, and the teacher pass threshold is 1.482094298.
No unseen-test or leaderboard improvement is claimed.

Hypothesis: a larger from-scratch Transformer may learn useful distributions
which can improve a smaller, already inference-feasible Transformer through
distillation. Neither greater teacher capacity nor distillation guarantees gains.
The official guide limits the **submitted predictor's inference** assets/time;
training architecture/cost may differ. All teacher cost must be disclosed.

## Fixed plan and gates

1. Finish Stage15 first. Lock the best resource-qualified Stage15/B validation
   BPB and source/checkpoint hashes before looking at any Stage17 result.
2. Teacher: 18,538,753 parameters; 10 layers, width384, 8 heads, RoPE/RMSNorm/SwiGLU, dropout.1, tied
   embeddings and causal prefix-copy64. Random initialization, train split only.
   The teacher is training-only and is explicitly **not submission-eligible**.
3. Train teacher 14,400 updates, microbatch16, accumulation2, seed17, BF16,
   AdamW(.9,.999), weight decay.1, clip1, peak LR.001, 100 warmup updates,
   existing baseline cosine schedule with minimum ratio.1. Two-update smoke
   then exact-resume continuation; an OOM/error stops rather than auto-expanding
   or silently changing the plan. Full FP32 validation every300 updates.
4. Teacher selection is the fixed uniform average of snapshots at updates
   13200/13500/13800/14100/14400. Evaluate with the unchanged CPU FP32 evaluator.
   If it does not beat the locked reference by at least .003 BPB, stop: no student
   training. This is an efficiency gate, not a theorem that weaker teachers can
   never help. The teacher has no inference-resource gate because it is excluded
   from every deployed predictor; its resource/cost records remain in training logs.
5. On teacher pass, train two fresh 6x256 B students with the same seed,
   microbatch, schedule, 14,400 updates, sampled windows and initial weights:
   (a) CE-only alpha0; (b) KD alpha.5, temperature2. No parameter grid.
   Teacher initialization preserves student CPU/CUDA RNG; teacher is frozen,
   in eval mode, and only observes causal **training** input windows.
6. Loss: `(1-alpha)*CE + alpha*T^2*mean_positions KL(p_teacher_T || p_student_T)`.
   Both models supply complete 2048-token distributions. Average over positions,
   not batch items alone; no target-only teacher, no hard pseudo-text, no validation
   gradient updates. No teacher weights in student inference checkpoints.
7. Average the same last-five student checkpoints. Preserve teacher hash,
   teacher training targets, recipe, and averaging ancestry in the student bundle.
   Score both students on full CPU FP32 validation. Measure exact-checkpoint
   three-repeat CPU/RAM/assets for candidates improving the reference by >=.003.
   Retain the previous qualified candidate if no new predictor passes.
8. Report KD-vs-CE difference separately from improvement over historical B/F/G.
   All outcomes remain single-seed; no claim of statistical robustness.

Each run processes 117,964,800 gradient targets. Maximum new gradient budget is
353,894,400 (teacher + both students), plus 117,964,800 teacher-forward target
positions during KD. Teacher rejection uses only the first third. Count all
pretraining once in total search cost and separately disclose each student's
dependency on it. FP32 periodic validation cost is additional. Saved cumulative
process cost excludes crash-lost unsaved work; document such failures separately.

## Implementation and verification

- `train_distillation.py`: independent additive trainer; active Stage15 source
  files are unchanged. Alpha0 numerically matches the original trainer in a
  synthetic deterministic test. CUDA equivalence is not claimed from a CPU test.
- Tests cover per-position KL/T-scaling, detached teacher, frozen eval mode,
  teacher RNG preservation, hash/provenance guards, exact resume, CE-control
  weight equality, and ancestry-preserving averaging. CUDA BF16 is tested on the
  Windows host only after Stage15 releases its GPU.
- `scripts/run_stage17_distillation.py`: dry-run by default, explicit `--execute`,
  completed-Stage15 guard, fixed budget, teacher rejection, source pinning,
  validation-only scoring, and same-checkpoint resource gate.
- Recovery is explicit; an existing output directory is never automatically
  rerun. Individual trainers support resume with an identical saved plan.
- Source method: Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural
  Network*, https://arxiv.org/abs/1503.02531. Custom implementation; no external
  code, data, pretrained weights or paper result used as this project's evidence.
- Substantive AI assistance: constraint analysis, design, code, tests, experiment
  orchestration and documentation. The student must understand the loss, causal
  contract, teacher exclusion, evidence limits and acknowledge this assistance.

Run inside the existing Windows `code/` workspace after Stage15 completes:

```powershell
.venv\Scripts\python.exe scripts/run_stage17_distillation.py --run-dir runs/stage17-distillation-s17 --baseline runs/stage3-long-baseline-s17/checkpoint-best.pt --stage15-screen runs/stage15-mechanisms-s17/screen.json
# Inspect the bounded plan before appending --execute to start training.
```

The baseline path matches the existing Stage15 launch script; its hash is checked
against Stage15's receipt. The directory contains completed artifacts: do not
reuse it for a new run. The one-off task was
launched through `start_windows_job.ps1 -Job stage17 -RunId stage17-20260921-a`.
