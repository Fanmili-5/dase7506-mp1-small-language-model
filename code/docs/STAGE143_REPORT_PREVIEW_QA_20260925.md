# Stage143 report preview QA (not a final report)

The tracked `REPORT.pdf` still describes Stage10 and remains unchanged.
Stage143 is not frozen, has no full-test score, and has not been submitted.
The report source is `REPORT_STAGE143_DRAFT.md`. Its development-only PDF can
be regenerated locally with:

```text
<bundled-or-system-python-with-reportlab> code/scripts/render_stage143_report_preview.py
```

The builder writes `output/pdf/STAGE143_REPORT_PREVIEW_NOT_FOR_SUBMISSION.pdf`.
This binary is intentionally ignored by Git so it cannot be mistaken for the
required final repository report. The source Markdown and builder are tracked.

On 25 September 2026, the preview rendered as **three A4 pages / 76,256
bytes**, below the assignment's ten-page report limit. All three pages were
visually inspected after rendering at 1,200-pixel scale: title, section
hierarchy, ablation table, hashes, evaluation commands, references and footer
were legible without clipping or overlap. Text extraction verified the
Stage143 validation BPB, CPU/RAM/asset numbers, explicit pending-test status,
Stage169 negative result, deadline and "NOT FOR SUBMISSION" watermark.
The renderer uses fonts bundled with ReportLab rather than a macOS-only font.
The historical `REPORT.pdf` SHA-256 was
`44607086c854236627d2a375f84a1e92fa8dc2f16b65122c9bb17308c0413c9b`.

This checks layout and selected text, **not** the scientific correctness of
every claim or final-submission readiness. After the method and exact bundle
are frozen, insert the one matching test score and final links, regenerate
the report as the actual `REPORT.pdf`, inspect every final page, check the
page count again, and ensure the old Stage10 PDF is replaced only then.

After adding the Stage170–174 results and refreshing the archived search-cost
total, the development preview was regenerated from the revised Markdown on
25 September 2026. The latest ignored PDF is **three A4 pages / 77,087
bytes**, again below the ten-page cap. All three latest page images were
visually inspected: the code example is kept together, the ablation table,
section breaks, references, hashes, page numbers and development watermark
are legible, with no clipping or nearly empty fourth page. Extracted text
contains the Stage174 failed gate, Linux one-thread ratio, updated 60-record
cost audit and explicit pending-test status. The temporary preview is still
**not** the final `REPORT.pdf`; later candidate changes or the frozen test
result require another render and full visual QA.

On 26 September (UTC+8), the source was updated with the Stage175 answer-
aware oracle, Stage176 failed out-of-half causal gate and Stage177 failed
input-only resource screen. A first render spilled only two checklist bullets
onto a nearly empty fourth page; the negative-results prose was shortened
without changing its evidence claims. The regenerated preview is **three A4
pages / 77,080 bytes**, SHA-256
`1b31a8e045a1dcf97eae8a208c26279137715130e0533460ade954cf06481d27`.
All three latest pages were rendered at 1,200-pixel scale and visually
inspected: no clipping, orphaned checklist page, broken ablation table or
split command block remains. Text extraction confirms the three new Stage
numbers, the `PENDING FREEZE` label and all three `NOT FOR SUBMISSION`
footers. The tracked historical `REPORT.pdf` hash remains unchanged at
`44607086c854236627d2a375f84a1e92fa8dc2f16b65122c9bb17308c0413c9b`.
This is still only a development preview; it has no Stage143 test score,
release manifest or final link audit.

On 26 September (UTC+8), the report draft was refreshed with the Stage178
output-head triage, the completed Stage179 stronger-dropout regression and
the recalculated archived search-cost audit (**61** distinct metrics files,
**59** comparable training-time fields, **50,596.95 seconds**). The latest
ignored PDF is **three A4 pages / 77,118 bytes**, SHA-256
`3d60e89c24b29b3c89a5200f3352bfa40186848b8453a93363c2930b071e3713`.
All three pages were rendered to 1,200-pixel PNGs and visually inspected:
the table, section transitions, hashes, command block, references, page
numbers and development watermark are legible, with no clipping or orphaned
fourth page. Text extraction confirms Stage178/179, the updated search-cost
figures, `PENDING FREEZE` and three `NOT FOR SUBMISSION` footers. This does
not turn the preview into the required matching final report; Stage143 remains
untested and unfrozen.

On 26 September, the report source added the Stage180 optimizer result and
Stage181's rejected intermediate-layer readout, corrected the training-cost
wording, and moved the draft-only finalization checklist out of the report
body (the operational checklist remains in
`SUBMISSION_PORTAL_READINESS_20260926.md`). The first render put only its
last checklist item on a nearly empty fourth page; after removing that
non-report section, the preview returned to **three A4 pages / 76,869 bytes**,
SHA-256 `c8ff8bf5e2766434e67b8a806775c61a0f38c1406b8c8b79ce64cefb9a4adeb7`.
All three final page images were inspected at 1,600-pixel scale. Text
extraction confirms Stage181, `PENDING FREEZE` and three development
watermarks. No clipping, orphan page, broken table or split command block
was observed. The tracked historical `REPORT.pdf` remains unchanged at
SHA-256 `44607086c854236627d2a375f84a1e92fa8dc2f16b65122c9bb17308c0413c9b`.
This is still a development preview, not a Stage143 test result or submission.

After the completed Stage182 head-count pilot, the draft and archived
search-cost figures were refreshed to **63** metrics files, **61** comparable
training-time fields and **52,121.37 seconds / 14.48 hours** of reported
training. The regenerated ignored PDF is **three A4 pages / 76,932 bytes**,
SHA-256 `717a85eec7d16bd62b1abd121281426922ef86483a2a71c8f00b7126ff2c5517`.
All three pages were rendered to 1,200-pixel PNGs and visually inspected;
the table, hashes, command block, references and footers remain legible,
with no clipping or orphan page. Extracted text confirms Stage182, the new
cost total, `PENDING FREEZE` and three `NOT FOR SUBMISSION` watermarks.
The tracked historical `REPORT.pdf` was not replaced.

On 26 September, the development draft added Stage183's rejected geometric
fusion screen and the freeze-gated test-entry wording. The Linux portability
paragraph was shortened without changing its measured one-thread failure or
the unavailable four-thread conclusion, so section 6 starts cleanly on page
three. The regenerated ignored PDF is **three A4 pages / 76,621 bytes**,
SHA-256 `04300a1725fb7d27f946ce7db8b0473f7bd2112429302afbcca5398a6213e7b5`.
All three pages were rendered to 1,200-pixel PNGs and visually inspected:
no clipped text, orphaned paragraph, split command block or broken ablation
table. Text extraction confirms Stage183, the frozen-test-entry reference,
`PENDING FREEZE`, and all three `NOT FOR SUBMISSION` footers. The tracked
historical `REPORT.pdf` remains unchanged; this preview is not a final report.
