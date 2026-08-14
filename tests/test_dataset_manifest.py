import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from dataset_manifest import DATASET_SCHEMA, build_manifest, time_gap_minutes


class DatasetManifestTests(unittest.TestCase):
    def arrays(self):
        return {
            "X": np.zeros((3, 5, 32, 32), dtype=np.float32),
            "y": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            "B": np.array([0.0, 10.0, 20.0], dtype=np.float32),
            "times": np.array([
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:12:00+00:00",
                "2026-01-01T01:24:00+00:00",
            ]),
            "label_times": np.array([
                "2026-01-01T02:00:00+00:00",
                "2026-01-01T02:12:00+00:00",
                "2026-01-01T03:24:00+00:00",
            ]),
            "lead_minutes": np.array([120.0, 120.0, 120.0], dtype=np.float32),
        }

    def test_time_gap_minutes_uses_timestamp_order(self):
        gaps = time_gap_minutes(self.arrays()["times"])
        self.assertEqual(gaps.tolist(), [12.0, 72.0])

    def test_manifest_records_hash_gaps_and_positive_counts(self):
        arrays = self.arrays()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid_dataset.npz"
            np.savez_compressed(path, **arrays)
            manifest = build_manifest(
                path,
                arrays,
                thresholds_mm=[15.0, 25.0, 35.0],
                provenance={"source": "test", "requested_days": ["2026-01-01"]},
            )
        self.assertEqual(manifest["dataset_schema"], DATASET_SCHEMA)
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        self.assertEqual(manifest["time_range"]["gaps_over_60_minutes"], 1)
        self.assertEqual(manifest["targets"]["rain120_15mm"]["positive_count"], 2)
        self.assertEqual(manifest["provenance"]["source"], "test")


if __name__ == "__main__":
    unittest.main()
