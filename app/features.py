import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
WEIGHTS_JSON = MODEL_DIR / "weights.json"

FEATURE_COLS = [
    "temp_mean",
    "hum_mean",
    "rain_total",
    "rain_1h",
    "rain_3h",
    "hum_1h_delta",
    "temp_1h_delta",
    "hour",
    "season",
    "w_WTS",
    "tc_dist_km",
    "tc_wind_kts",
    "tc_24h_dist_km",
]

TARGETS = ["rain_1h", "amber_3h", "red_3h", "tc3_6h"]

TARGET_LABELS = {
    "rain_1h": "Rain next 1h",
    "amber_3h": "Amber Rainstorm ≤3h",
    "red_3h": "Red Rainstorm ≤3h",
    "tc3_6h": "TC Signal 3+ ≤6h",
}


def sigmoid(x):
    if x <= -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def load_weights():
    if not WEIGHTS_JSON.exists():
        return None
    with open(WEIGHTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def feature_vector(row):
    return [float(row.get(col, 0.0) or 0.0) for col in FEATURE_COLS]


def predict_from_row(row):
    weights = load_weights()
    if not weights:
        return {}
    out = {}
    for target in TARGETS:
        entry = weights.get(target)
        if not entry or "coef" not in entry:
            continue
        mean = entry["mean"]
        std = entry["std"]
        coef = entry["coef"]
        intercept = entry["intercept"]
        x = feature_vector(row)
        z = intercept
        for xi, m, s, c in zip(x, mean, std, coef):
            z += ((xi - m) / s) * c
        out[target] = sigmoid(z)
    return out


def blend_weight(weights):
    if not weights or "meta" not in weights:
        return 0.0
    trained = [t for t in TARGETS if weights.get(t, {}).get("coef")]
    n = weights["meta"].get("n_total", 0)
    if not trained:
        return 0.0
    return max(0.2, min(0.5, 0.2 + (n - 48) / 952 * 0.3))
