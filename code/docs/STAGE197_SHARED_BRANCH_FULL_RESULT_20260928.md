# Stage197 full-run result: shared-trunk two-path model rejected

The predeclared seed-17, 7,200-step Windows CUDA run completed, presenting
58,982,400 primary next-token training targets. The exact Stage54 R-Drop
optimizer and learning-rate schedule were used, with only the model changed
to the existing Stage153 six-block shared trunk and two heterogeneous upper
paths. The development loader read and hash-verified supplied train and
validation text only. No test text was opened by this run or selection.

The complete 376,599-target, 1,148,007-byte validation endpoint scored
**1.429042653 BPB**. The prespecified uniform same-trajectory average of
steps 6000/6300/6600/6900/7200 independently scored **1.426863400 BPB**
in FP32 and was selected. All five source checkpoint step/target counts and
the complete-run status were verified before averaging. The selected model
improves the matched Stage54 last-five average (1.428594045 BPB) by only
**0.001730645 BPB**, and is **0.027177238 BPB worse** than the protected
Stage143 complete-validation candidate (1.399686162 BPB). It remains
**0.076863400 BPB above** the aspirational 1.35 validation target.

The predeclared Stage143-advancement gate required at least 0.015 BPB gain;
it fails decisively. Stage197 is therefore stopped before train-only count
rebuilding, compact OpenVINO export, complete CPU/RAM/asset qualification,
method freeze, or test scoring. The earlier synthetic GPU memory and input-
only CPU feature checks do not establish course resource compliance. The
Stage143 checkpoint and graph remain unchanged.

Raw evidence: `code/results/stage197-shared-branch-gpu-preflight-v2.json`,
`code/results/stage197-run.json`, `code/results/stage197-progress.json`,
`code/results/stage197-metrics.json`, and `code/results/stage197-selection.json`.
The five trajectory checkpoints and averaged checkpoint remain on the
Windows experiment host, not in the inference bundle. This is a single-seed
negative result for this architecture and recipe, not a claim that all
shared-trunk models fail.
