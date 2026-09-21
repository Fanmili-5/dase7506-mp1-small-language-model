# Windows training handoff

This handoff is intentionally separated from model selection. The Mac repository
is the source of truth until the Windows CUDA smoke test passes.

## 1. Prepare the transfer on Mac

From `code/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_fixed_files.py
.venv/bin/python scripts/make_transfer_bundle.py
```

The last command creates:

```text
../dist/mp1-windows-transfer.zip
../dist/mp1-windows-transfer.zip.sha256
```

The ZIP excludes virtual environments, checkpoints, run directories, bytecode,
and Git metadata. It includes the supplied data, so no dataset download is needed.
The repository also marks `code/data/*` as non-text in `.gitattributes`, preventing
Git for Windows from changing benchmark bytes through CRLF conversion.

## 2. Move and extract

Transfer the ZIP by local network, USB drive, cloud storage, VS Code Remote
Tunnel, or another private channel. Extract it to a short path such as:

```text
C:\mp1
```

Avoid OneDrive-synchronized directories for active training runs because sync
clients can lock or partially upload checkpoint files.

## 3. Install the pinned CUDA environment

Install a current NVIDIA driver and 64-bit Python 3.12. Open PowerShell in
`C:\mp1\code` and run:

```powershell
nvidia-smi
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows_cuda.ps1
```

The setup script creates `.venv`, installs PyTorch 2.7.1 with CUDA 12.6 plus the
pinned NumPy/tokenizers versions, runs all unit tests, validates data hashes, and
requires CUDA to be visible.

## 4. Run the CUDA smoke test

```powershell
.\scripts\smoke_windows_cuda.ps1
```

The smoke test performs ten GPU updates and reloads the resulting checkpoint in
the supplied CPU FP32 evaluator on validation. Do not proceed if any contract,
hash, CUDA, checkpoint-load, or normalization check fails.

## 5. Preserve evidence

Keep the whole generated smoke run until its metadata has been reviewed:

```text
code\runs\smoke-<timestamp>\
```

Useful files are `run.json`, `progress.json`, `metrics.json`, `checkpoint.pt`,
and `validation_cpu_fp32.json`. `resume.pt` is larger and is needed only to resume
an interrupted formal run.

## 6. Launch Stage 1 only after review

```powershell
.\scripts\run_stage1_windows.ps1
```

This runs four predeclared equal-target validation experiments. It does not call
the test split. Copy `results\summary.csv` and the run metadata back to the Mac
for analysis before starting Stage 2.

## Operational safeguards

For an SSH-independent, one-off job while the Windows user remains logged in:

```powershell
.\scripts\start_windows_job.ps1 -Job smoke -RunId smoke-unique-id
# Only after the smoke job and CPU validation pass:
.\scripts\start_windows_job.ps1 -Job stage1 -RunId stage1-unique-id
```

These create on-demand Windows Scheduled Tasks named `MP1-<RunId>`, at limited
privilege, with no stored password and a six-hour execution limit. They are not
recurring automations. Check `job-logs/<RunId>/status.json`, `console.log`, and
the run artifacts. `completed` means the wrapper finished successfully; inspect
the actual metrics before interpreting model quality. If status remains
`running` after a crash, also check Windows Task Scheduler's last result.
Closing SSH does not own the task lifetime, but Windows logout, reboot, sleep,
loss of AC power, or execution limits can still interrupt it. This is not
automatic resume: inspect checkpoints and recorded training arguments first.

- Keep the laptop connected to power and prevent sleep/hibernate on AC power.
- Prefer the lid open with the display off for cooling.
- Monitor temperature and power with `nvidia-smi -l 5` during the first run.
- Do not delete or overwrite a run directory; each experiment needs a unique path.
- Do not sync `.venv`; recreate it with the setup script on each platform.
- Do not move a checkpoint without its exact source/config hashes and metrics.
