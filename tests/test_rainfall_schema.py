import gzip
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from backfill_tabular import build_rows, parse_isd_lite
from data_quality import (
    DATA_SCHEMA_VERSION,
    RAIN_SOURCE_HKO_DISTRICT_MAXIMA,
    RAIN_SOURCE_NOAA_STATION,
    validate_row,
)
from fetch import CSV_COLUMNS, build_row


class RainfallSchemaTests(unittest.TestCase):
    def test_hko_row_uses_peak_as_one_hour_feature_and_leaves_three_hour_blank(self):
        weather = {
            "temperature": {"data": [{"value": 28.0}]},
            "humidity": {"data": [{"value": 80.0}]},
            "rainfall": {"data": [{"max": 4.0}, {"max": 9.0}, {"max": 2.0}]},
        }
        row, *_ = build_row(weather, {}, [], {}, [], {"tc_dist_km": 2000}, None)
        self.assertEqual(row["data_schema"], DATA_SCHEMA_VERSION)
        self.assertEqual(row["rain_source"], RAIN_SOURCE_HKO_DISTRICT_MAXIMA)
        self.assertEqual(row["rain_total"], 15.0)
        self.assertEqual(row["rain_main"], 9.0)
        self.assertEqual(row["rain_1h"], 9.0)
        self.assertEqual(row["rain_3h"], "")
        self.assertTrue(validate_row(row))

    def test_hko_validation_rejects_negative_or_fabricated_three_hour_rain(self):
        row = {
            "data_schema": DATA_SCHEMA_VERSION,
            "rain_source": RAIN_SOURCE_HKO_DISTRICT_MAXIMA,
            "rain_total": 2.0,
            "rain_main": 2.0,
            "rain_1h": 2.0,
            "rain_3h": -1.0,
        }
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_isd_parser_preserves_missing_hourly_precipitation(self):
        text = "2025 01 01 00 182 110 10192 90 40 7 -9999 120\n"
        obs = parse_isd_lite(gzip.compress(text.encode("ascii")))
        self.assertEqual(len(obs), 1)
        self.assertIsNone(obs[0]["precip1h"])
        self.assertEqual(obs[0]["precip6h"], 12.0)

    def test_backfilled_station_row_stays_unscaled_and_source_marked(self):
        obs = [{
            "ts": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "temp": 20.0,
            "rh": 70.0,
            "precip1h": 1.2,
            "precip6h": None,
        }]
        row = build_rows(obs, [], CSV_COLUMNS)[0]
        self.assertEqual(row["data_schema"], DATA_SCHEMA_VERSION)
        self.assertEqual(row["rain_source"], RAIN_SOURCE_NOAA_STATION)
        self.assertEqual(row["rain_total"], 1.2)
        self.assertEqual(row["rain_main"], 1.2)
        self.assertEqual(row["rain_1h"], 1.2)
        self.assertEqual(row["rain_3h"], "")
        self.assertTrue(validate_row(row))


if __name__ == "__main__":
    unittest.main()
