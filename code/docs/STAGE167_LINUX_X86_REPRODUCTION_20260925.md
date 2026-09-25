# Stage167: independent Linux x86-64 validation reproduction

The exact Stage143 source, checkpoint, feature graph, tokenizer and unchanged
scorer were checked on a fresh Ubuntu 24.04 GitHub Actions runner on
25 September 2026. The pinned CPU packages installed with Python 3.12.14;
fixed-file verification and all **188 contract/unit tests** passed. The
successful [run](https://github.com/Fanmili-5/dase7506-mp1-small-language-model/actions/runs/36136115074)
used OpenVINO 2026.4.0 FP32, one CPU inference thread, one stream and no
pinning. The runner exposed only two logical CPUs. An initial four-thread
request was internally reported as one by OpenVINO and deliberately failed
the model's strict thread-setting check; the successful run explicitly
requested and received one. Neither initial failure produced a score.

Full CPU FP32 validation with `--threads 1` scored **1.3996861790624688
BPB**, 376,599 targets and 1,148,007 raw UTF-8 bytes. This differs from the
Windows four-thread result **1.399686162042141** by only
**0.0000000170203278 BPB**. Checkpoint, implementation, scorer and tokenizer
SHA-256 fields match the Windows record. The Linux score JSON is at
`../results/stage166-linux-evidence/stage143-linux-x86-validation.json`;
the original per-window array is preserved as an artifact of the linked CI
run. The measured score-loop time was 50.793923178 seconds.

This is an independent score reproduction, **not** a Linux resource
qualification: no alternating baseline/candidate timing comparison,
whole-process peak-RSS measurement or release-bundle audit was run on this
host. Its one-thread time cannot be compared to the Windows four-thread
resource ratios. It neither reaches the aspirational 1.35 BPB nor accesses
the test split. The separate Mac ARM64 OpenVINO-device failure is documented
in the Stage166 runtime check.
