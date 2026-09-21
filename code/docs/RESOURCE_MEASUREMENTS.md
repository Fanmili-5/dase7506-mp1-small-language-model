# CPU resource checks

The course scorer is unchanged. Development measurements use the complete
validation split, CPU FP32, four threads and the same machine for both models.

```powershell
.venv\Scripts\python.exe scripts\benchmark_cpu.py `
  --baseline runs\stage1-baseline-s17\checkpoint.pt `
  --candidate runs\stage1-modern-s17\checkpoint.pt `
  --repeats 3 --threads 4 --output results\cpu-stage1.json
```

Each repetition launches a fresh scoring process. Model order alternates to
reduce ordering effects. The reported time is the unmodified scorer's scoring
time, not Python startup or data loading; the median is used for the 5x ratio.
All individual measurements and source/checkpoint hashes are retained. Check
that no other substantial CPU workload is running during the comparison.

Peak memory is the OS-reported lifetime peak resident memory of each child
process, including imports, model construction, loading all supplied splits and
scoring. It is not PyTorch's CUDA allocation counter and is not a sample of only
the current memory usage. Windows also records peak process commit separately.
Implementation references:

- Windows [GetProcessMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo)
  and [PROCESS_MEMORY_COUNTERS](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters).
- On macOS/Linux, `resource.getrusage(RUSAGE_SELF).ru_maxrss` is converted to bytes
  using the platform's units.

The development threshold is 4 GiB peak RSS. Checkpoint byte size alone does not
establish the 64 MiB inference-asset limit: the final bundle must also count any
additional inference assets. Validation resource checks are screening evidence,
not a substitute for the final frozen full-test measurement.

`results/mac-resource-selfcheck-20260919.json` compares the same two-update smoke
checkpoint against itself to test the measurement pipeline. Its `baseline` and
`candidate` labels do NOT identify two different model experiments. It is not
leaderboard evidence, and it must not be included as an improvement result.
