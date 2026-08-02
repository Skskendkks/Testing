import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
WEIGHTS_JSON = MODEL_DIR / "weights.json"
TREES_JSON = MODEL_DIR / "trees.json"
BLEND_JSON = MODEL_DIR / "blend.json"

# v4: added f3_* (gridded-nowcast scalars), tc_trend_toward, tc_dist_rate.
# Old model files whose coef length mismatches are ignored at predict time.
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
    "tc_trend_toward",
    "tc_dist_rate",
    "f3_max",
    "f3_mean",
    "f3_trend",
]

FEATURE_DEFAULTS = {
    "tc_dist_km": 2000.0,
    "tc_24h_dist_km": 2000.0,
}

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


def _load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_weights():
    return _load_json(WEIGHTS_JSON)


def load_trees():
    return _load_json(TREES_JSON)


def load_blend():
    return _load_json(BLEND_JSON)


def _feat(row, col):
    v = row.get(col, None)
    if v in (None, ""):
        return FEATURE_DEFAULTS.get(col, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return FEATURE_DEFAULTS.get(col, 0.0)


def feature_vector(row):
    return [_feat(row, col) for col in FEATURE_COLS]


def apply_cal(p, cal):
    """Platt calibration: sigmoid(a*p + b), fitted on the validation fold."""
    if not cal:
        return p
    return sigmoid(cal["a"] * p + cal["b"])


def _lr_prob(entry, x):
    z = entry["intercept"]
    for xi, m, s, c in zip(x, entry["mean"], entry["std"], entry["coef"]):
        z += ((xi - m) / s) * c
    return sigmoid(z)


def _tree_raw(tree, x):
    """Pure-Python traversal of one exported HistGradientBoosting tree."""
    feat, thr, left, right, leaf, val = (
        tree["feat"], tree["thr"], tree["left"], tree["right"], tree["leaf"], tree["val"],
    )
    i = 0
    while not leaf[i]:
        i = left[i] if x[feat[i]] <= thr[i] else right[i]
    return val[i]


def _trees_prob(entry, x):
    raw = entry["baseline"]
    for tree in entry["trees"]:
        raw += _tree_raw(tree, x)
    return sigmoid(raw)


def predict_ai(row):
    """Calibrated probability per target from the shipped model (trees preferred, LR fallback).

    A target only appears in trees.json / weights.json if it beat the persistence
    baseline on time-ordered validation (train.py enforces this), so presence == shipped.
    """
    trees = load_trees() or {}
    weights = load_weights() or {}
    x = feature_vector(row)
    out = {}
    for target in TARGETS:
        entry = trees.get(target)
        if isinstance(entry, dict) and entry.get("trees"):
            if entry.get("n_features", len(FEATURE_COLS)) != len(FEATURE_COLS):
                continue
            out[target] = apply_cal(_trees_prob(entry, x), entry.get("cal"))
            continue
        entry = weights.get(target)
        if not isinstance(entry, dict) or "coef" not in entry:
            continue
        if len(entry["coef"]) != len(FEATURE_COLS):
            continue  # stale model trained on an older feature set
        out[target] = apply_cal(_lr_prob(entry, x), entry.get("cal"))
    return out


# Backwards-compatible alias
predict_from_row = predict_ai


def blend_weight(weights):
    """Legacy global ramp — fallback when model/blend.json is absent."""
    if not weights or "meta" not in weights:
        return 0.0
    trained = [t for t in TARGETS if weights.get(t, {}).get("coef")]
    n = weights["meta"].get("n_total", 0)
    if not trained:
        return 0.0
    return max(0.2, min(0.5, 0.2 + (n - 48) / 952 * 0.3))


def blend_weights():
    """Per-target AI blend weight learned on the validation fold (P5).

    Returns {target: w} with w in [0, 1]. Falls back to the legacy global ramp.
    """
    blend = load_blend()
    if blend and isinstance(blend.get("targets"), dict):
        return {t: float(blend["targets"].get(t, 0.0)) for t in TARGETS}
    legacy = blend_weight(load_weights())
    return {t: legacy for t in TARGETS}
