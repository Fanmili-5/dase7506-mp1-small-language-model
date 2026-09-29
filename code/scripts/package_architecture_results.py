"""Archive audited Stage-14 scores, final weights and logs, not a submission ZIP.

Periodic snapshots and resume states remain on the training machine. Their
hashes are checked by the auditor before this smaller transfer archive is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    run = args.run_dir
    screen = json.loads((run / "screen.json").read_text(encoding="utf-8-sig"))
    audit = json.loads((run / "audit-completed.json").read_text(encoding="utf-8-sig"))
    status = json.loads((args.job_dir / "status.json").read_text(encoding="utf-8-sig"))
    if screen["status"] != "completed" or status["status"] != "completed":
        raise ValueError("Only completed jobs may be archived")
    if audit["screen_sha256"] != digest(run / "screen.json"):
        raise ValueError("Audit is for a different screen")
    files = {"screen.json": run / "screen.json", "audit-completed.json": run / "audit-completed.json"}
    for path in (run / "preflight").glob("*-resource.json"):
        files[f"preflight/{path.name}"] = path
    for row in screen["candidates"]:
        if row["status"] == "resource_rejected":
            continue
        directory = run / row["name"]
        names = ["run.json", "metrics.json", "progress.json", "checkpoint.pt", "average-last5.pt",
                 "endpoint-validation-cpu-fp32.json", "average-validation-cpu-fp32.json",
                 "endpoint-validation-cpu-fp32.window-nll.npy", "average-validation-cpu-fp32.window-nll.npy"]
        if row.get("final_resource_pass") is not None:
            names += ["final-resource.json", "final-resource.log"]
        for name in names:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files[f"{row['name']}/{name}"] = path
    for name in ("status.json", "console.log"):
        files[f"job-logs/{name}"] = args.job_dir / name
    code = Path(__file__).resolve().parents[1]
    for relative, expected in screen["source_hashes"].items():
        source = code / relative
        if digest(source) != expected:
            raise ValueError(f"Changed frozen source: {relative}")
        files[f"source/{relative}"] = source
    auditor = code / "scripts/audit_architecture_screen.py"
    if digest(auditor) != audit["audit_script_sha256"]:
        raise ValueError("Auditor changed since verification")
    files["source/scripts/audit_architecture_screen.py"] = auditor
    manifest = {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in files.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("TRANSFER_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": digest(args.output), "files": len(files),
                      "note": "Evidence archive, not a course submission bundle."}))


if __name__ == "__main__":
    main()
