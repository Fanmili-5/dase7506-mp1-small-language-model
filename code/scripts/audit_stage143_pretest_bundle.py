"""Audit the pre-test bundle and an independent clean-extract validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from package_stage143_pretest_candidate import CODE, EVIDENCE, verify


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--remote-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify(args.bundle)
    final = json.loads(EVIDENCE.read_text(encoding="utf-8-sig"))
    validation = json.loads(args.validation.read_text(encoding="utf-8-sig"))
    bundle_sha = sha(args.bundle)
    if bundle_sha != args.remote_bundle_sha256.lower():
        raise ValueError("Windows-transferred bundle hash differs")
    if (validation["split"] != "validation" or validation["device"] != "cpu"
            or validation["precision"] != "fp32"
            or validation["protocol"] != final["protocol"]
            or validation["targets"] != 376599
            or validation["utf8_bytes"] != 1148007
            or abs(validation["bpb"] - final["validation_bpb"]) > 1e-10
            or validation["checkpoint_sha256"] != final["checkpoint_sha256"]
            or validation["implementation_sha256"] != final["implementation_sha256"]
            or validation["evaluator_sha256"] != final["source_hashes"]["evaluate.py"]
            or validation["tokenizer_sha256"] != final["source_hashes"]["data/tokenizer.json"]):
        raise ValueError("Clean-extract validation is not the qualified Stage143 model")
    audit = dict(
        status="pretest_bundle_clean_extract_validation_audited_not_frozen_or_submitted",
        bundle_sha256=bundle_sha, bundle_bytes=args.bundle.stat().st_size,
        bundle_code_commit=manifest["code_commit"],
        inference_files=len(manifest["files"]),
        conservative_inference_asset_bytes=final["conservative_asset_bytes"],
        checkpoint_sha256=final["checkpoint_sha256"],
        graph_sha256=final["graph_sha256"],
        clean_extract_validation_bpb=validation["bpb"],
        clean_extract_validation_targets=validation["targets"],
        clean_extract_validation_utf8_bytes=validation["utf8_bytes"],
        clean_extract_validation_seconds=validation["seconds"],
        no_test_scoring=True,
        source_evidence="results/stage143-evidence/final.json",
        validation_evidence=args.validation.resolve().relative_to(CODE).as_posix(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
