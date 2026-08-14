import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import features
from data_quality import DATA_SCHEMA_VERSION


class PortableModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.row = {name: 0.0 for name in features.FEATURE_COLS}
        self.row["temp_mean"] = 28.0
        self.row["hum_mean"] = 80.0

    def lr_entry(self, *, std=None, coef=None):
        n = len(features.FEATURE_COLS)
        return {
            "intercept": 0.0,
            "mean": [0.0] * n,
            "std": [1.0] * n if std is None else std,
            "coef": [0.0] * n if coef is None else coef,
        }

    def test_legacy_zero_std_with_zero_coef_is_safe(self):
        entry = self.lr_entry()
        entry["std"][2] = 0.0
        entry["coef"][2] = 0.0
        with patch.object(features, "load_trees", return_value={}), patch.object(
            features, "load_weights", return_value={"meta": {"data_schema": DATA_SCHEMA_VERSION}, "amber_3h": entry}
        ):
            out = features.predict_ai(self.row)
        self.assertIn("amber_3h", out)
        self.assertTrue(math.isfinite(out["amber_3h"]))
        self.assertGreaterEqual(out["amber_3h"], 0.0)
        self.assertLessEqual(out["amber_3h"], 1.0)

    def test_invalid_zero_std_with_nonzero_coef_is_skipped(self):
        entry = self.lr_entry()
        entry["std"][2] = 0.0
        entry["coef"][2] = 0.2
        with patch.object(features, "load_trees", return_value={}), patch.object(
            features, "load_weights", return_value={"meta": {"data_schema": DATA_SCHEMA_VERSION}, "amber_3h": entry}
        ):
            out = features.predict_ai(self.row)
        self.assertNotIn("amber_3h", out)

    def test_artifact_with_old_data_schema_is_ignored(self):
        entry = self.lr_entry()
        with patch.object(features, "load_trees", return_value={}), patch.object(
            features, "load_weights", return_value={"meta": {"data_schema": "1"}, "amber_3h": entry}
        ):
            out = features.predict_ai(self.row)
        self.assertNotIn("amber_3h", out)

    def test_non_finite_input_is_replaced_by_feature_default(self):
        row = dict(self.row, temp_mean="NaN")
        self.assertEqual(features.feature_vector(row)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
