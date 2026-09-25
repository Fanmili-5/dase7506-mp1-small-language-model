"""Package Stage143 only after a recorded freeze, CPU test score and final report.

This script does not evaluate or read the test text. It binds already-produced
artifacts and refuses to package the historical Stage10 report. The PDF page
count and visual layout must be checked separately before publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
sys.path.insert(0, str(CODE))

FINAL = CODE / "results/stage143-evidence/final.json"
CHECKPOINT = "checkpoints/stage143-openvino-order6.pt"
HISTORICAL_STAGE10_REPORT_SHA = "44607086c854236627d2a375f84a1e92fa8dc2f16b65122c9bb17308c0413c9b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def expected_files(final: dict) -> dict:
    files = {CHECKPOINT: {
        "sha256": final["checkpoint_sha256"], "bytes": final["checkpoint_bytes"]}}
    for relative in final["inference_files"]:
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or relative == CHECKPOINT):
            raise ValueError(f"Unsafe or duplicate inference path: {relative}")
        files[relative] = {
            "sha256": final["source_hashes"][relative],
            "bytes": final["asset_sizes"][relative],
        }
    if sum(row["bytes"] for row in files.values()) != final["conservative_asset_bytes"]:
        raise ValueError("Qualified inference asset total changed")
    if final["conservative_asset_bytes"] > 64 * 1024**2:
        raise ValueError("Inference assets exceed course limit")
    return files


def validate_freeze(freeze: dict, final: dict, qualification_sha: str) -> dict:
    if (freeze.get("status") != "method_frozen_before_test"
            or freeze.get("method_frozen") is not True
            or freeze.get("protocol") != final["protocol"]
            or freeze.get("candidate") != "stage143-openvino-order6"
            or freeze.get("qualification_sha256") != qualification_sha
            or freeze.get("validation_bpb") != final["validation_bpb"]
            or freeze.get("conservative_inference_asset_bytes")
            != final["conservative_asset_bytes"]
            or freeze.get("dirty_entries") != []
            or not freeze.get("frozen_at_utc")):
        raise ValueError("Missing or inconsistent pre-test method freeze")
    commit = freeze.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("Missing frozen source commit")
    expected = expected_files(final)
    if freeze.get("inference_files") != expected:
        raise ValueError("Frozen inference set differs from qualified files")
    return expected


def validate_test(test: dict, final: dict, freeze_sha256: str,
                  frozen_source_commit: str) -> None:
    if (test.get("protocol") != final["protocol"]
            or test.get("split") != "test"
            or test.get("device") != "cpu"
            or test.get("precision") != "fp32"
            or test.get("targets") != 428405
            or test.get("utf8_bytes") != 1292013
            or test.get("checkpoint_sha256") != final["checkpoint_sha256"]
            or test.get("implementation_sha256")
            != final["source_hashes"]["student_stage143_openvino_singlepass.py"]
            or test.get("evaluator_sha256") != final["source_hashes"]["evaluate.py"]
            or test.get("tokenizer_sha256")
            != final["source_hashes"]["data/tokenizer.json"]
            or test.get("freeze_record_sha256") != freeze_sha256
            or test.get("frozen_source_commit") != frozen_source_commit):
        raise ValueError("Test score does not identify the frozen CPU FP32 predictor")
    bpb, nll = test.get("bpb"), test.get("nll_nats")
    if (not isinstance(bpb, (int, float)) or not math.isfinite(bpb) or bpb <= 0
            or not isinstance(nll, (int, float)) or not math.isfinite(nll)
            or abs(bpb - nll / math.log(2) / test["utf8_bytes"]) > 1e-10):
        raise ValueError("Test BPB arithmetic is inconsistent")


def validate_repository(freeze: dict, files: dict, report: Path,
                        freeze_path: Path, test_path: Path) -> str:
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Final release requires a clean committed worktree")
    commit = git("rev-parse", "HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"],
                    commit], cwd=REPO, check=True)
    if report.resolve() != (REPO / "REPORT.pdf").resolve():
        raise ValueError("Final report must be the tracked repository REPORT.pdf")
    if not report.read_bytes().startswith(b"%PDF-"):
        raise ValueError("Final report is not a PDF")
    if sha(report) == HISTORICAL_STAGE10_REPORT_SHA:
        raise ValueError("Historical Stage10 report cannot describe Stage143")
    if int(git("cat-file", "-s", "HEAD:REPORT.pdf")) != report.stat().st_size:
        raise ValueError("Final report is not committed at HEAD")
    for evidence in (freeze_path, test_path):
        try:
            relative = evidence.resolve().relative_to(REPO.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("Freeze and test evidence must be in the final code commit") from error
        if int(git("cat-file", "-s", f"HEAD:{relative}")) != evidence.stat().st_size:
            raise ValueError(f"Final evidence is not committed at HEAD: {relative}")
    for relative, row in files.items():
        path = CODE / relative
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Frozen inference file changed: {relative}")
        if int(git("cat-file", "-s", f"HEAD:code/{relative}")) != row["bytes"]:
            raise ValueError(f"Inference file missing from final commit: {relative}")
    return commit


def build(freeze_path: Path, test_path: Path, report: Path, output: Path) -> dict:
    from scripts.package_stage143_pretest_candidate import archived_entry

    if output.exists():
        raise FileExistsError("Refusing to overwrite final bundle")
    final, freeze, test = read_json(FINAL), read_json(freeze_path), read_json(test_path)
    files = validate_freeze(freeze, final, sha(FINAL))
    validate_test(test, final, sha(freeze_path), freeze["source_commit"])
    commit = validate_repository(freeze, files, report, freeze_path, test_path)
    manifest = {
        "status": "final_stage143_bundle_after_method_freeze",
        "protocol": final["protocol"],
        "frozen_source_commit": freeze["source_commit"],
        "release_code_commit": commit,
        "freeze_record_sha256": sha(freeze_path),
        "full_test_result_sha256": sha(test_path),
        "full_test_cpu_fp32_bpb": test["bpb"],
        "full_test_targets": test["targets"],
        "full_test_utf8_bytes": test["utf8_bytes"],
        "report_sha256": sha(report),
        "conservative_inference_asset_bytes": final["conservative_asset_bytes"],
        "files": files,
        "manual_pdf_page_and_layout_check_required": True,
    }
    readme = (
        "Stage143 frozen checkpoint and inference assets.\n"
        "Use release_code_commit in BUNDLE_MANIFEST.json for the exact code/report.\n"
        "Install per code/README.md; extract this ZIP over its code/ directory.\n"
        "From code/: python evaluate.py --checkpoint "
        "checkpoints/stage143-openvino-order6.pt --device cpu "
        "--precision fp32 --threads 4 --split test --output reproduced-test.json\n"
        "The manifest binds the freeze record, already-scored full-test JSON "
        "and matching report by SHA-256.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", allowZip64=True) as archive:
        for relative, row in sorted(files.items()):
            data = (CODE / relative).read_bytes()
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"File changed during packaging: {relative}")
            archived_entry(archive, "code/" + relative, data)
        archived_entry(archive, "BUNDLE_MANIFEST.json",
                       (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        archived_entry(archive, "README_FINAL.txt", readme.encode())
    verify(output, manifest)
    return manifest


def verify(output: Path, manifest: dict) -> None:
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        expected = {"code/" + name for name in manifest["files"]}
        if len(names) != len(set(names)) or set(names) != expected | {
                "BUNDLE_MANIFEST.json", "README_FINAL.txt"}:
            raise ValueError("Final archive members differ from manifest")
        if json.loads(archive.read("BUNDLE_MANIFEST.json")) != manifest:
            raise ValueError("Final archive manifest changed")
        for relative, row in manifest["files"].items():
            data = archive.read("code/" + relative)
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Final archive entry changed: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--test-result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.freeze, args.test_result, args.report, args.output)
    print(json.dumps({
        "status": manifest["status"],
        "release_code_commit": manifest["release_code_commit"],
        "full_test_cpu_fp32_bpb": manifest["full_test_cpu_fp32_bpb"],
        "conservative_inference_asset_bytes": manifest["conservative_inference_asset_bytes"],
        "zip_sha256": sha(args.output),
        "zip_bytes": args.output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
