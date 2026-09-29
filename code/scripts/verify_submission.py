"""Read-only checks of frozen assets, recorded score and release identity; no scoring."""
from __future__ import annotations

import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts.final_submission import verify


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
