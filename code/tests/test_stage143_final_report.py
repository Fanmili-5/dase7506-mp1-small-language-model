"""Report-template tests use synthetic metadata; they never score test text."""
from __future__ import annotations

import unittest

from scripts.render_stage143_final_report import SOURCE, compose


class Stage143FinalReportTests(unittest.TestCase):
    def test_fills_exactly_one_frozen_score_without_draft_markers(self) -> None:
        # 2.0 is fictional gate-test metadata, not the student's MP1 score.
        result = compose(SOURCE.read_text(encoding="utf-8"),
                         {"bpb": 2.0, "targets": 428405, "utf8_bytes": 1292013},
                         "0" * 40)
        self.assertIn("**2.000000000**", result)
        self.assertIn("**428,405** targets", result)
        self.assertIn("**1,292,013** raw UTF-8 bytes", result)
        self.assertIn("Stage143 final report", result)
        self.assertNotIn("PENDING", result)
        self.assertNotIn("Draft only", result)
        self.assertNotIn("Do not run it yet", result)
        self.assertNotIn("not yet been tested", result)

    def test_template_drift_and_invalid_score_are_rejected(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        metadata = {"bpb": 2.0, "targets": 428405, "utf8_bytes": 1292013}
        with self.assertRaisesRegex(ValueError, "template drift"):
            compose(source.replace("PENDING FREEZE", "changed"), metadata,
                    "0" * 40)
        with self.assertRaisesRegex(ValueError, "Invalid verified test score"):
            compose(source, {**metadata, "bpb": float("nan")}, "0" * 40)
        with self.assertRaisesRegex(ValueError, "Invalid verified test score"):
            compose(source, metadata, "not-a-commit")


if __name__ == "__main__":
    unittest.main()
