"""Asset-only staging checks; no benchmark split is opened."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_stage143_portable_assets as portable


class Stage143PortableAssetTests(unittest.TestCase):
    def test_current_qualified_assets_match_without_freeze_or_test(self) -> None:
        result = portable.audit()
        self.assertEqual(result["status"], "portable_assets_match_qualification_only")
        self.assertEqual(result["inference_files"], 18)
        self.assertEqual(result["conservative_inference_asset_bytes"], 55810412)
        self.assertFalse(result["method_frozen"])
        self.assertFalse(result["test_scored_by_this_script"])
        self.assertFalse(result["test_text_opened_by_this_script"])

    def test_missing_or_changed_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            with patch.object(portable, "CODE", Path(scratch)):
                with self.assertRaisesRegex(FileNotFoundError, "Missing qualified file"):
                    portable.audit()
            final = json.loads(portable.FINAL.read_text(encoding="utf-8"))
            final["checkpoint_sha256"] = "0" * 64
            fake_final = Path(scratch) / "final.json"
            fake_final.write_text(json.dumps(final), encoding="utf-8")
            with patch.object(portable, "FINAL", fake_final):
                with self.assertRaisesRegex(ValueError, "Qualified file changed"):
                    portable.audit()


if __name__ == "__main__":
    unittest.main()
