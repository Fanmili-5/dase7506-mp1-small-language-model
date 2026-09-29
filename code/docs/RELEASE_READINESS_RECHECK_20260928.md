# MP1 release-readiness recheck, 28 September 2026

This is a read-only consistency audit plus local staging package, **not** a
course submission or an accepted score. Stage143 remains the sole frozen,
Windows-resource-qualified candidate. Its complete validation score is
1.399686162042141 BPB, and its already frozen CPU FP32 full-test score is
1.415657616535609 BPB; both miss the student's 1.35 aim and the test score
misses the student's 1.38 minimum. Stage219 and the other later experiments
did not replace it.

After committing the Stage219 negative result and AI disclosure at code
commit `2828b42d44156ab54450f9a13e7e83160e9bf284`, the existing
Stage143 readiness audit, method-freeze binding, complete-test arithmetic,
1,674-window sidecar, repository file hashes, and inference-asset manifest
all passed read-only revalidation. The report is still the committed
three-page A4 Stage143 PDF with SHA-256
`c0fa2e88e3c7114eea34d15dc6b5d1665b2685f21595036a3e22f038871c7d2d`;
its extracted text reports the matching validation/test scores and Linux
one-thread timing caveat. This check did not rescore test.

A new **local-only** fallback ZIP was built, not uploaded:
`output/final/stage143-fallback-20260928-2828b42.zip` with SHA-256
`e6729661de223480217ab1b5f55a7e3c683e5d0702c014089a239f74e64063aa`.
Its manifest names the exact current code commit, existing pre-test frozen
source commit, matching report/checkpoint hashes and 55,810,412 counted
uncompressed inference-asset bytes. The ZIP is ignored from Git; this note
does not make it an accessible checkpoint link.

As of this recheck, the GitHub repository remains **private**, the student ID
has not been provided in this task, and no course-site score issue or public
immutable code/checkpoint links are verified. The supplied guide asks for an
initial student-ID/full-test-BPB issue **before 29 September (UTC+8)** and
both immutable links by the end of 30 September. The earlier site snapshot
displayed a different score deadline; until clarified, the guide's earlier
date is the safer operational boundary. Publishing the repository or
submitting the below-threshold score requires the student's explicit
decision; neither was done by this audit.
