# MP1 submission portal readiness (checked 26 September 2026 UTC+8)

This is a preparation record, **not** a submission receipt or permission to
score test before a method freeze.

## Source discrepancy and conservative deadline

The supplied [assignment guide](../../guide.md) says to submit student ID
and complete-test BPB **before 29 September 2026** and provide immutable code
and matching checkpoint links by the end of 30 September. The current
[course website](https://xudongwu-0.github.io/courses/dase7506/#submit) says
score submissions close at the end of 30 September; its published
[project configuration](https://xudongwu-0.github.io/courses/dase7506/assets/projects.127dafd8d78d.mjs)
currently has `score_deadline: 2026-10-01T00:00:00+08:00`.
Because those sources disagree, use the **earlier guide deadline** for the
initial score issue unless the instructor explicitly clarifies otherwise.
Do not treat the live site's later date as an extension of the written guide.

## Portal and GitHub sequence

1. Freeze the exact method, source commit, checkpoint and inference assets
   using validation-only evidence. Stop development selection before reading
   the Stage143 test result. The [pre-test bundle audit](../results/stage143-pretest-bundle-evidence/audit.json)
   is a packaging rehearsal, not the freeze. The read-only
   `python scripts/freeze_stage143_method.py` preflight checks the exact 18
   committed inference files, hashes/sizes, validation/resource evidence
   and clean Git state. `eligible_for_freeze_only` does **not** freeze or
   test. Only after the student chooses to freeze should its explicit
   `--confirm --output <new freeze-record path>` mode write the source-commit
   and asset manifest, before any test command is run.
2. Commit the freeze JSON before testing, then use
   `scripts/run_stage143_frozen_test.py --freeze <committed-freeze.json> --output <new-test-result.json>`
   to invoke the unchanged complete-test CPU FP32 scorer on the frozen
   predictor. The wrapper requires a clean repository, the committed freeze
   record and all 18 qualified inference-file hashes before it can invoke
   the scorer. It attaches the freeze-record SHA-256 to the result; the final
   packager requires that binding. This enforces the intended sequence for
   this workflow but does not independently prove a human's chronology.
   Submit **its** BPB, not the 1.399686162 validation BPB or historical Stage10
   test score. Record test target count, byte count, full score JSON and hashes.
3. On the portal, enter the student's actual ID and full-test BPB, accept the
   publication/links consent, choose **Prepare score submission**, then confirm
   creation of the generated issue on GitHub. A prepared URL alone is not a
   submitted issue. Use the same GitHub account for later updates.
4. Prepare a publicly accessible immutable code link (commit or tag) with the
   matching <=10-page report and exact reproduction instructions. The current
   `Fanmili-5/dase7506-mp1-small-language-model` repository is **private** as
   of this check; a private URL does not yet satisfy public review access.
5. Provide a matching checkpoint-bundle link that lets peers evaluate without
   retraining. The local pre-test ZIP is explicitly non-final and is not yet a
   publicly downloadable checkpoint link. Audit its replacement against the
   frozen commit, checkpoint and ONNX graph before posting.
   `scripts/package_stage143_final_release.py` is prepared for **after** the
   freeze, CPU FP32 full-test JSON, and replacement `REPORT.pdf` have all been
   committed. It requires `--freeze`, `--test-result`, `--report` and a new
   `--output` ZIP path; it checks full-test coverage/BPB arithmetic, exact
   evaluator/checkpoint/tokenizer hashes, ancestry of the frozen commit,
   the clean final code commit and all 18 inference-file hashes. It rejects
   the tracked historical Stage10 report. Synthetic metadata unit tests
   cover rejection paths but are not a substitute for a real final bundle
   audit. Independently verify the final PDF page count (<=10) and visual
   layout; the packaging script does not certify those properties.
6. When the portal enters its links-required phase, it asks for the **original
   score issue number** and both links; confirm the generated link issue on
   GitHub. The guide's earlier final-link deadline remains the conservative
   planning target. Preserve the issue URLs/receipts.

## Current blockers and caveats

- Student ID and the user's freeze-versus-one-more-day choice are pending.
- Stage143 has not been frozen or test-scored. The tracked `REPORT.pdf` is an
  earlier Stage10 report; `REPORT_STAGE143_DRAFT.md`/its preview are not final.
- Windows four-thread resource qualification passes, but a separate Linux
  one-thread host measured 5.502555x baseline CPU time. This portability risk
  remains disclosed; a Windows pass is not a universal CPU guarantee.
- The public leaderboard shows self-reported entries; it is not evidence that
  other models' checkpoints or CPU resource limits have been verified.
