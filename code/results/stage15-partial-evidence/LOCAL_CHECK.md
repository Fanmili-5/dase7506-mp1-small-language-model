# Partial Stage15 evidence, not a completed screen audit

Source screen snapshot: 2026-09-21T12:38:35.749633+00:00. D and F completed;
E resource-rejected; G still training. No final winner has been selected.

Collected F's real `average-last5.pt` into the local project outputs directory.
Verified SHA256 `5783e16de954168793148d3a95c7559ddd6ec2916987247ea060cd459801aa16`.
Checked all screen source hashes against the local checkout. Called the existing
`check_score` and `check_resource` audit functions on the raw evidence: protocol,
coverage, CPU FP32, source/checkpoint identities, NLL-to-BPB arithmetic, all six
fresh-process resource rows, medians, memory maxima, and pass flags agree.

- F validation BPB: 1.4850942978138386.
- Three-repeat candidate/baseline median time ratio: 4.877905205660784.
- Peak candidate process RSS: 1,988,993,024 bytes.
- Reported conservative total assets: 29,681,660 bytes (includes successor source
  even though F does not require it); 64 MiB limit passes.
- The average checkpoint metadata lists the fixed last-five target counts.
  Source snapshot files have not yet been collected/re-averaged locally, so this
  is **not** a claim of completed trajectory ancestry audit.

No new test evaluation, final freeze, clean-package QA or course submission.
