import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from backfill import build_samples, dedupe_snapshots


class BackfillDatasetTests(unittest.TestCase):
    def leads(self, value):
        return [np.full((32, 32), value + index, dtype=np.float32) for index in range(4)]

    def test_dedupe_snapshots_preserves_first_snapshot_per_time(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        unique, duplicates = dedupe_snapshots([(ts, self.leads(1)), (ts, self.leads(9))])
        self.assertEqual(duplicates, 1)
        self.assertEqual(len(unique), 1)
        self.assertEqual(float(unique[0][1][0][0, 0]), 1.0)

    def test_build_samples_tracks_label_time_and_lead_minutes(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        snapshots = [(start, self.leads(0)), (start + timedelta(minutes=120), self.leads(20))]
        x, y, baseline, input_times, label_times, lead_minutes = build_samples(snapshots)
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), 1)
        self.assertEqual(input_times, [start.isoformat()])
        self.assertEqual(label_times, [(start + timedelta(minutes=120)).isoformat()])
        self.assertEqual(lead_minutes, [120.0])
        self.assertEqual(baseline, [3.0])
        self.assertEqual(y[0], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
