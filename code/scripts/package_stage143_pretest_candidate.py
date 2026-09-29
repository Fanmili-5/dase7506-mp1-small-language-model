"""Build a hash-checked Stage143 candidate bundle without freezing or testing.

The archive is a pre-test staging artifact, never a final-submission claim.
It contains the exact inference files counted in Stage143's resource audit
plus its checkpoint. The code repository supplies the fixed data/scorer split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
EVIDENCE = CODE / "results/stage143-evidence/final.json"
CHECKPOINT = CODE / "checkpoints/stage143-openvino-order6.pt"
FIXED_ZIP_TIME = (2026, 9, 26, 0, 0, 0)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archived_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate bundle: {output}")
    final = json.loads(EVIDENCE.read_text(encoding="utf-8-sig"))
    if (final["protocol"] != "7506-mp1-wt2-v2" or not final["qualified"]
            or not final["no_test_scoring"] or final["split"] != "validation"):
        raise ValueError("Stage143 validation-only evidence is not qualified")
    files = {"checkpoints/stage143-openvino-order6.pt": CHECKPOINT}
    for relative in final["inference_files"]:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe inference path: {relative}")
        files[relative] = CODE / relative
    entries = {}
    contents = {}
    for relative, path in sorted(files.items()):
        data = path.read_bytes()
        digest = sha256(data)
        expected = (final["checkpoint_sha256"] if relative.startswith("checkpoints/")
                    else final["source_hashes"][relative])
        if digest != expected:
            raise ValueError(f"Changed candidate file: {relative}")
        expected_bytes = (final["checkpoint_bytes"] if relative.startswith("checkpoints/")
                          else final["asset_sizes"][relative])
        if len(data) != expected_bytes:
            raise ValueError(f"Changed candidate file size: {relative}")
        entries[relative] = {"sha256": digest, "bytes": len(data)}
        contents[relative] = data
    counted = sum(row["bytes"] for row in entries.values())
    if counted != final["conservative_asset_bytes"] or counted > 64 * 1024**2:
        raise ValueError("Inference asset count differs from Stage143 audit")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {
        "status": "pretest_candidate_not_frozen_or_submitted",
        "protocol": final["protocol"],
        "code_commit": commit,
        "source_evidence": "code/results/stage143-evidence/final.json",
        "validation_bpb": final["validation_bpb"],
        "cpu_ratio_windows": final["cpu_ratio"],
        "peak_rss_bytes_windows": final["peak_rss_bytes"],
        "conservative_inference_asset_bytes": counted,
        "test_scored": False,
        "files": entries,
    }
    readme = (
        "Stage143 PRE-TEST candidate bundle; NOT frozen or submitted.\n"
        "Use with the matching code commit in BUNDLE_MANIFEST.json.\n"
        "Extract the code/ directory into that repository's code/ directory.\n"
        "The code repository supplies fixed data, tokenizer checks and scorer.\n"
        "From code/: python evaluate.py --checkpoint "
        "checkpoints/stage143-openvino-order6.pt --device cpu "
        "--precision fp32 --threads 4 --split validation\n"
        "Do not score the test split until the method is frozen.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", allowZip64=True) as archive:
        for relative, data in contents.items():
            archived_entry(archive, "code/" + relative, data)
        archived_entry(archive, "BUNDLE_MANIFEST.json",
                       (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        archived_entry(archive, "README_PRETEST.txt", readme.encode())
    return manifest


def verify(archive_path: Path) -> dict:
    final = json.loads(EVIDENCE.read_text(encoding="utf-8-sig"))
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entry")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if (manifest["status"] != "pretest_candidate_not_frozen_or_submitted"
                or manifest["protocol"] != final["protocol"]
                or manifest["test_scored"] is not False
                or manifest["validation_bpb"] != final["validation_bpb"]):
            raise ValueError("Candidate manifest does not match validation evidence")
        if not isinstance(manifest["code_commit"], str) or len(manifest["code_commit"]) != 40:
            raise ValueError("Missing immutable code commit")
        expected = {"code/" + name for name in manifest["files"]}
        if set(names) != expected | {"BUNDLE_MANIFEST.json", "README_PRETEST.txt"}:
            raise ValueError("Archive members differ from manifest")
        if set(manifest["files"]) != set(final["inference_files"]) | {
                "checkpoints/stage143-openvino-order6.pt"}:
            raise ValueError("Bundle inference file set does not match Stage143 audit")
        for name, row in manifest["files"].items():
            data = archive.read("code/" + name)
            if len(data) != row["bytes"] or sha256(data) != row["sha256"]:
                raise ValueError(f"Archive member does not match manifest: {name}")
            expected_digest = (final["checkpoint_sha256"] if name.startswith("checkpoints/")
                               else final["source_hashes"][name])
            expected_bytes = (final["checkpoint_bytes"] if name.startswith("checkpoints/")
                              else final["asset_sizes"][name])
            if row != {"sha256": expected_digest, "bytes": expected_bytes}:
                raise ValueError(f"Archive member differs from qualified Stage143: {name}")
        if (sum(row["bytes"] for row in manifest["files"].values()) !=
                final["conservative_asset_bytes"]):
            raise ValueError("Unexpected inference asset sum")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
    manifest = verify(args.output)
    print(json.dumps({
        "status": manifest["status"], "code_commit": manifest["code_commit"],
        "files": len(manifest["files"]),
        "conservative_inference_asset_bytes": manifest["conservative_inference_asset_bytes"],
        "zip_sha256": sha256(args.output.read_bytes()), "zip_bytes": args.output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
