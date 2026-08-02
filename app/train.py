import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FEATURE_COLS, TARGETS, TARGET_LABELS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
WEIGHTS_JSON = MODEL_DIR / "weights.json"
METRICS_JSON = MODEL_DIR / "metrics.json"

ROWS_PER_HOUR = 3

LOOKAHEAD = {
    "rain_1h": 4,
    "amber_3h": 12,
    "red_3h": 12,
    "tc3_6h": 24,
}

MIN_POSITIVES = 5
MIN_ROWS = 200


def load_rows():
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("tc_dist_km") in (None, ""):
            r["tc_dist_km"] = 2000
        if r.get("tc_wind_kts") in (None, ""):
            r["tc_wind_kts"] = 0
        if r.get("tc_24h_dist_km") in (None, ""):
            r["tc_24h_dist_km"] = 2000
    return rows


def label(row, target):
    if target == "rain_1h":
        return 1 if _f(row, "rain_1h") > 1.0 else 0
    look = LOOKAHEAD[target]
    flag_map = {"amber_3h": "w_RAIN_AMBER", "red_3h": "w_RAIN_RED", "tc3_6h": "w_TC3"}
    return 1 if _f(row, flag_map[target]) > 0 else 0


def _f(row, key):
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main():
    rows = load_rows()
    n = len(rows)
    out = {"meta": {"n_total": n, "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds")}}
    metrics = {"n_total": n}
    if n < MIN_ROWS:
        out["meta"]["note"] = f"only {n} rows; need {MIN_ROWS} to train — rules-only mode"
        MODEL_DIR.mkdir(exist_ok=True)
        WEIGHTS_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
        METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[train] skipped: {n} rows < {MIN_ROWS}")
        return
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score
    except ImportError:
        print("[train] scikit-learn not installed — run `pip install scikit-learn` (retrain workflow does this)")
        raise

    X_all = [[_f(r, c) for c in FEATURE_COLS] for r in rows]
    model_out = dict(out)
    for target in TARGETS:
        horizon = LOOKAHEAD[target]
        ys = []
        xs = []
        for i in range(n - horizon):
            if target == "rain_1h":
                y = 1 if (_f(rows[i + horizon], "rain_total") - _f(rows[i], "rain_total")) > 1.0 else 0
            else:
                flag = {"amber_3h": "w_RAIN_AMBER", "red_3h": "w_RAIN_RED", "tc3_6h": "w_TC3"}[target]
                y = 1 if any(_f(r, flag) > 0 for r in rows[i + 1: i + horizon + 1]) else 0
            ys.append(y)
            xs.append(X_all[i])
        n_pos = sum(ys)
        if n_pos < MIN_POSITIVES:
            print(f"[train] {target}: skipped ({n_pos} positives < {MIN_POSITIVES})")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(xs, ys)
        preds = clf.predict(xs)
        m = {
            "intercept": float(clf.intercept_[0]),
            "coef": [float(c) for c in clf.coef_[0]],
            "n_pos": n_pos,
            "n_neg": len(ys) - n_pos,
            "acc": round(accuracy_score(ys, preds), 3),
            "precision": round(precision_score(ys, preds, zero_division=0), 3),
            "recall": round(recall_score(ys, preds, zero_division=0), 3),
        }
        model_out[target] = m
        metrics[target] = {k: v for k, v in m.items() if k != "coef"}
        print(f"[train] {target}: pos={n_pos} acc={m['acc']} prec={m['precision']} rec={m['recall']}")

    means = []
    stds = []
    for c in FEATURE_COLS:
        values = [_f(r, c) for r in rows]
        mean = sum(values) / n
        var = sum((x - mean) ** 2 for x in values) / n
        means.append(round(mean, 4))
        stds.append(round(var ** 0.5, 4) if var ** 0.5 > 1e-6 else 1e-6)
    trained = [t for t in TARGETS if t in model_out]
    for target in trained:
        model_out[target]["mean"] = means
        model_out[target]["std"] = stds

    MODEL_DIR.mkdir(exist_ok=True)
    WEIGHTS_JSON.write_text(json.dumps(model_out, indent=2), encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train] done: {sum(1 for t in TARGETS if t in model_out)}/{len(TARGETS)} targets trained")


if __name__ == "__main__":
    main()
