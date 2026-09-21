"""Create a source/data ZIP for moving the project to the Windows trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "runs", "dist", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pt", ".npy"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db"}


def include(path: Path) -> bool:
    relative = path.relative_to(REPO)
    return (
        path.name not in EXCLUDED_NAMES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "dist/mp1-windows-transfer.zip")
    args = parser.parse_args()
    files = [path for path in sorted(REPO.rglob("*")) if path.is_file() and include(path)]
    manifest = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(REPO).as_posix()
            data = path.read_bytes()
            archive.writestr(relative, data)
            manifest[relative] = {"sha256": digest(data), "bytes": len(data)}
        archive.writestr(
            "TRANSFER_MANIFEST.json",
            json.dumps({"files": manifest}, indent=2).encode("utf-8") + b"\n",
        )
    archive_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(args.output), "sha256": archive_hash, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
