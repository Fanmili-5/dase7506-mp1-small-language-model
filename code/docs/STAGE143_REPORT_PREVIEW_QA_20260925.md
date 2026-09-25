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
