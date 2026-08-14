"""Lightweight validation for snapshot rows and the versioned rainfall schema."""

import math

DATA_SCHEMA_VERSION = "2"
RAIN_SOURCE_HKO_DISTRICT_MAXIMA = "hko_district_maxima_1h"
RAIN_SOURCE_NOAA_STATION = "noaa_isd_lite_station_1h"


def _number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def validate_row(row):
    """Validate a version-2 snapshot without inventing unavailable rainfall data."""
    if str(row.get("data_schema", "")) != DATA_SCHEMA_VERSION:
        raise ValueError(f"expected data_schema={DATA_SCHEMA_VERSION}")
    if row.get("rain_source") not in {
        RAIN_SOURCE_HKO_DISTRICT_MAXIMA,
        RAIN_SOURCE_NOAA_STATION,
    }:
        raise ValueError("unknown rain_source")

    total = _number(row, "rain_total")
    main = _number(row, "rain_main")
    one_hour = _number(row, "rain_1h")
    three_hour = _number(row, "rain_3h")
    for name, value in (("rain_total", total), ("rain_main", main), ("rain_1h", one_hour), ("rain_3h", three_hour)):
        if value is not None and value < 0:
            raise ValueError(f"{name} must not be negative")

    source = row["rain_source"]
    if source == RAIN_SOURCE_HKO_DISTRICT_MAXIMA:
        # A temporary API omission must not stop the official-data poll. Such a
        # row is retained with blank rain metrics but excluded from model training.
        if all(value is None for value in (total, main, one_hour)):
            return True
        if None in (total, main, one_hour):
            raise ValueError("HKO rainfall metrics are incomplete")
        if total < main:
            raise ValueError("district-maxima sum must be at least the peak district value")
        if abs(one_hour - main) > 1e-9:
            raise ValueError("rain_1h must equal the peak district one-hour value")
        if three_hour is not None:
            raise ValueError("three-hour district rainfall is unavailable and must be blank")

    return True


def training_row_is_compatible(row):
    """Return true only for valid rows with live HKO district-rainfall semantics."""
    try:
        validate_row(row)
    except ValueError:
        return False
    return (
        row.get("rain_source") == RAIN_SOURCE_HKO_DISTRICT_MAXIMA
        and row.get("rain_1h") not in (None, "")
    )
