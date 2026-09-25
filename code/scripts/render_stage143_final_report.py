"""Render a Stage143 final PDF only from a committed freeze and full-test record.

This script never opens benchmark test text or runs the scorer. It refuses to
replace the historical Stage10 REPORT.pdf without an explicit flag, and the
resulting PDF still requires visual review before submission.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from scripts.package_stage143_final_release import (  # noqa: E402
    FINAL, HISTORICAL_STAGE10_REPORT_SHA, expected_files, git, read_json, sha,
    validate_freeze, validate_test, validate_window_nll,
)
SOURCE = ROOT / "REPORT_STAGE143_DRAFT.md"
FINAL_REPORT = ROOT / "REPORT.pdf"
BASELINE_TEST_BPB = 2.102014912


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Final report template drift: expected once: {old[:64]!r}")
    return source.replace(old, new, 1)


def compose(source: str, test: dict, frozen_commit: str) -> str:
    """Fill only verified final fields; fail if the draft wording drifts."""
    prefix = "# DASE7506 Project 1 — Stage143 report draft\n\n"
    if not source.startswith(prefix) or source.count("## 1. Abstract") != 1:
        raise ValueError("Unexpected Stage143 draft title or abstract")
    preamble, body = source.split("## 1. Abstract", 1)
    if "**Draft only; do not submit.**" not in preamble:
        raise ValueError("Missing draft-only preamble")
    text = "# DASE7506 Project 1 — Stage143 final report\n\n## 1. Abstract" + body
    bpb = float(test["bpb"])
    if (not math.isfinite(bpb) or bpb <= 0 or not re.fullmatch(r"[0-9a-f]{40}", frozen_commit)):
        raise ValueError("Invalid verified test score or frozen commit")
    text = replace_once(
        text, "Full-test BPB: **PENDING FREEZE**.",
        f"Frozen Stage143 complete-test CPU FP32 BPB: **{bpb:.9f}** over "
        f"**{test['targets']:,}** targets and **{test['utf8_bytes']:,}** raw UTF-8 bytes.")
    text = replace_once(text, "The current development model combines",
                        "The frozen model combines")
    text = replace_once(
        text,
        "not be presented as Stage143's test score. The current candidate's complete\n"
        "validation BPB is reported above;\n"
        "test comparison and relative improvement remain pending.",
        "not be presented as Stage143's test score. The frozen Stage143 result above\n"
        f"improves on the original baseline by **{(1 - bpb / BASELINE_TEST_BPB) * 100:.2f}%** "
        "in BPB on the same complete test split; the development validation score\n"
        "is reported separately.")
    text = replace_once(
        text,
        "The prospective checkpoint and feature graph are pinned separately (these\n"
        "are candidate hashes, not a final release manifest):",
        "The frozen checkpoint and feature graph are pinned separately by the\n"
        "committed pre-test freeze record:")
    text = replace_once(
        text,
        "After committing a method freeze, use the SHA-bound\n"
        "[`frozen-test entry point`](code/scripts/run_stage143_frozen_test.py) once\n"
        "for the ranked CPU FP32 score. **Do not run it yet.**\n"
        "The final code link must be immutable and must match the complete checkpoint\n"
        "bundle, ONNX graph, dependencies and report. **PENDING FINAL HASH/LINK AUDIT.**",
        "The method was frozen at committed source revision\n"
        f"`{frozen_commit}` before the SHA-bound\n"
        "[`frozen-test entry point`](code/scripts/run_stage143_frozen_test.py)\n"
        "produced the complete CPU FP32 score reported above. The freeze record,\n"
        "test JSON and window-loss sidecar bind this result to the exact\n"
        "checkpoint and source hashes. Public code/checkpoint links and peer\n"
        "reproduction require a separate final release audit.")
    text = replace_once(
        text,
        "The current candidate has\n"
        "not yet been tested or submitted; exact final accuracy and ranking are\n"
        "unknown.",
        "A post-freeze complete-test result is recorded above, but ranking and\n"
        "eligibility on the course CPU remain subject to independent verification.")
    text = replace_once(text, "and this report draft. Earlier conceptual work",
                        "and this report. Earlier conceptual work")
    if re.search(r"(?i)\bdraft\b|PENDING FREEZE|PENDING FINAL|DO NOT RUN IT YET|NOT FOR SUBMISSION", text):
        raise ValueError("Final report still contains draft or pending language")
    return text


def committed_file(path: Path) -> None:
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Freeze/test evidence must be inside the repository") from error
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"],
                                            cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Evidence is not committed: {relative}") from error
    if committed != path.read_bytes():
        raise ValueError(f"Evidence changed after commit: {relative}")


def checked_inputs(freeze_path: Path, test_path: Path) -> tuple[dict, dict]:
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Commit all existing work before rendering the final report")
    committed_file(freeze_path)
    committed_file(test_path)
    final = read_json(FINAL)
    freeze = read_json(freeze_path)
    test = read_json(test_path)
    files = validate_freeze(freeze, final, sha(FINAL))
    validate_test(test, final, sha(freeze_path), freeze["source_commit"])
    validate_window_nll(test, test_path.with_suffix(".window-nll.npy"))
    subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"],
                    git("rev-parse", "HEAD")], cwd=ROOT, check=True)
    if files != expected_files(final):
        raise ValueError("Frozen file set differs from qualification")
    for relative, row in files.items():
        path = CODE / relative
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen inference file changed: {relative}")
    return freeze, test


def footer(canvas, doc) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4

    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd8e1"))
    canvas.line(45, 37, A4[0] - 45, 37)
    canvas.setFont("Vera", 7)
    canvas.setFillColor(colors.HexColor("#657888"))
    canvas.drawString(45, 25, "DASE7506 MP1 - FROZEN STAGE143 REPORT")
    canvas.drawRightString(A4[0] - 45, 25, str(doc.page))
    canvas.restoreState()


def render(source: str, output: Path) -> int:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate
    from scripts.render_stage143_report_preview import make_story, register_fonts, styles

    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4,
                            leftMargin=45, rightMargin=45,
                            topMargin=46, bottomMargin=50,
                            title="DASE7506 MP1 Stage143 final report",
                            author="DASE7506 MP1 student")
    doc.build(make_story(source, styles()), onFirstPage=footer,
              onLaterPages=footer)
    info = subprocess.check_output(["pdfinfo", str(output)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if not match or not 1 <= int(match.group(1)) <= 10:
        raise ValueError("Final PDF page count is missing or exceeds ten")
    extracted = subprocess.check_output(["pdftotext", str(output), "-"], text=True)
    if ("FROZEN STAGE143 REPORT" not in extracted
            or "NOT FOR SUBMISSION" in extracted
            or "PENDING" in extracted):
        raise ValueError("Final PDF text contains a draft marker or lacks final footer")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--test-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-historical-report", action="store_true")
    args = parser.parse_args()
    target = args.output.resolve()
    if target == FINAL_REPORT.resolve():
        if not args.replace_historical_report or sha(FINAL_REPORT) != HISTORICAL_STAGE10_REPORT_SHA:
            raise ValueError("Replacing historical REPORT.pdf requires explicit flag and exact old hash")
    elif target.exists() or args.replace_historical_report:
        raise ValueError("Use a new preview path; replacement flag is only for REPORT.pdf")
    freeze, test = checked_inputs(args.freeze, args.test_result)
    text = compose(SOURCE.read_text(encoding="utf-8"), test, freeze["source_commit"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="stage143-final-", suffix=".pdf",
                                     dir=target.parent, delete=False) as handle:
        scratch = Path(handle.name)
    try:
        pages = render(text, scratch)
        os.replace(scratch, target)
    finally:
        scratch.unlink(missing_ok=True)
    print(json.dumps({"status": "rendered_after_frozen_test_manual_visual_qa_required",
                      "report": str(target), "pages": pages,
                      "sha256": sha(target), "test_bpb": test["bpb"]}, indent=2))


if __name__ == "__main__":
    main()
