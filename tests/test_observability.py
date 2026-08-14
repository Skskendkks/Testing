import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from fetch import prediction_details
from health import failure_status, success_status


class ObservabilityTests(unittest.TestCase):
    def test_prediction_details_identifies_rules_and_blended_modes(self):
        rules = {"rain_1h": 0.1, "amber_3h": 0.2, "red_3h": 0.02, "tc3_6h": 0.1}
        ai = {"amber_3h": 0.8}
        weights = {"amber_3h": 0.5}
        final = {"rain_1h": 0.1, "amber_3h": 0.5, "red_3h": 0.02, "tc3_6h": 0.1}
        details = prediction_details(rules, ai, weights, final)
        self.assertEqual(details["rain_1h"]["mode"], "rules")
        self.assertEqual(details["amber_3h"]["mode"], "blended")
        self.assertEqual(details["amber_3h"]["final"], 0.5)
        self.assertEqual(details["amber_3h"]["ai"], 0.8)

    def test_success_status_marks_optional_source_loss_as_degraded(self):
        row = {"ts": "2026-08-14T10:00:00+00:00", "rain_1h": ""}
        status = success_status(row, f3_available=False, model_modes={"amber_3h": "rules"})
        self.assertEqual(status["status"], "degraded")
        self.assertEqual(status["components"]["rainfall"], "unavailable")
        self.assertEqual(status["components"]["f3_nowcast"], "unavailable")

    @patch("health.load_health", return_value={"last_success_at": "2026-08-14T09:00:00+00:00"})
    def test_failure_status_preserves_last_success(self, _load_health):
        status = failure_status("test failure")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["last_success_at"], "2026-08-14T09:00:00+00:00")
        self.assertEqual(status["summary"], "test failure")


if __name__ == "__main__":
    unittest.main()
