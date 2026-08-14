import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import cnn
from cnn_benchmark import completed_report, dataset_summary, validate_dataset


class CnnBenchmarkTests(unittest.TestCase):
    def valid_data(self):
        n = 250
        y = np.zeros((n, len(cnn.V3_TARGETS)), dtype=np.float32)
        # Put enough positives in the chronological holdout (last 20%).
        y[-15:, :] = 1.0
        return {
            "X": np.zeros((n, cnn.IN_CH, cnn.SIZE, cnn.SIZE), dtype=np.float32),
            "y": y,
            "B": np.zeros(n, dtype=np.float32),
            "times": np.array([f"2026-01-01T{i:04d}" for i in range(n)]),
        }

    def test_compatible_dataset_passes_contract(self):
        data = self.valid_data()
        self.assertEqual(validate_dataset(data), [])
        summary = dataset_summary(data)
        self.assertEqual(summary["samples"], 250)
        self.assertTrue(summary["has_advection_baseline"])
        self.assertEqual(summary["input_shape"], [cnn.IN_CH, cnn.SIZE, cnn.SIZE])

    def test_legacy_dataset_is_blocked_without_baseline_or_channels(self):
        data = self.valid_data()
        data.pop("B")
        data["X"] = np.zeros((250, 3, cnn.SIZE, cnn.SIZE), dtype=np.float32)
        reasons = validate_dataset(data)
        self.assertTrue(any("missing required arrays: B" in reason for reason in reasons))
        self.assertTrue(any("expected X shape" in reason for reason in reasons))

    def test_completed_report_requires_baseline_comparison_for_conclusion(self):
        summary = dataset_summary(self.valid_data())
        metrics = {
            target: {"pr_auc": 0.1, "brier": 0.2, "beats_baseline": target == "rain120_15mm"}
            for target in cnn.V3_TARGETS
        }
        report = completed_report(summary, metrics)
        self.assertEqual(report["status"], "completed")
        self.assertIn("added skill", report["targets"]["rain120_15mm"]["conclusion"])
        self.assertIn("CNN shows added skill", report["overall_conclusion"])


if __name__ == "__main__":
    unittest.main()
