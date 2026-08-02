"""v4 trainer: LR vs gradient-boosted trees, time-ordered eval, calibration, blend weights.

Outputs (all pure-Python-consumable at inference; see features.py):
  model/weights.json  — LR coefficients (target present only if LR chosen AND ships)
  model/trees.json    — exported HGB trees (target present only if trees chosen AND ships)
  model/blend.json    — per-target rules-vs-AI blend weight (P5)
  model/metrics.json  — per-target PR-AUC / Brier vs persistence & climatology (P4)

A target "ships" only if the calibrated model beats the persistence baseline
(current warning state continues) on BOTH PR-AUC and Brier on a time-ordered
validation split. Otherwise that target stays rules-only (blend weight 0).
"""

import csv
import json
import sys
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FEATURE_COLS, TARGETS, apply_cal, feature_vector, sigmoid, _trees_prob
from rules import rule_probs

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
WEIGHTS_JSON = MODEL_DIR / "weights.json"
TREES_JSON = MODEL_DIR / "trees.json"
BLEND_JSON = MODEL_DIR / "blend.json"
METRICS_JSON = MODEL_DIR / "metrics.json"

# True horizons (hours), matched to target names. Labels are computed by
# timestamp (bisect on ts), so polling cadence never changes the window (P4).
HORIZON_HOURS = {
    "rain_1h": 1.0,
    "amber_3h": 3.0,
    "red_3h": 3.0,
    "tc3_6h": 6.0,
}

FLAG = {"amber_3h": "w_RAIN_AMBER", "red_3h": "w_RAIN_RED", "tc3_6h": "w_TC3"}

MIN_ROWS = 48
MIN_POSITIVES = 4       # need a few in train AND val
MIN_CAL_POSITIVES = 5   # minimum val positives to fit Platt calibration
VAL_FRAC = 0.2
EPS = 1e-6


def _f(row, key, default=0.0):
    v = row.get(key, None)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rows():
    if not SNAPSHOT_CSV.exists():
        return []
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            r["_dt"] = datetime.fromisoformat(r["ts"])
        except (ValueError, KeyError):
            continue
        out.append(r)
    out.sort(key=lambda r: r["_dt"])
    return out


def build_samples(rows, target):
    """Timestamp-based lookahead labels. Returns (X, y, row_indices)."""
    ts = [r["_dt"] for r in rows]
    horizon = timedelta(hours=HORIZON_HOURS[target])
    last = ts[-1]
    X, y, idx = [], [], []
    for i, r in enumerate(rows):
        if last < r["_dt"] + horizon * 0.8:
            break  # tail rows lack lookahead coverage
        j0 = bisect_right(ts, r["_dt"])
        j1 = bisect_right(ts, r["_dt"] + horizon)
        future = rows[j0:j1]
        if not future:
            continue
        if target == "rain_1h":
            label = 1 if max(_f(fr, "rain_total") for fr in future) - _f(r, "rain_total") > 1.0 else 0
        else:
            label = 1 if any(_f(fr, FLAG[target]) > 0 for fr in future) else 0
        X.append(feature_vector(r))
        y.append(label)
        idx.append(i)
    return X, y, idx


def persistence_scores(rows, idx, target):
    """Baseline (a): current state continues."""
    out = []
    for i in idx:
        r = rows[i]
        if target == "rain_1h":
            out.append(1.0 if _f(r, "rain_1h") > 1.0 else 0.0)
        else:
            out.append(1.0 if _f(r, FLAG[target]) > 0 else 0.0)
    return out


def fit_platt(p_val, y_val):
    """1-D logistic map on raw probabilities; None if not enough signal."""
    from sklearn.linear_model import LogisticRegression
    if sum(y_val) < MIN_CAL_POSITIVES or sum(y_val) == len(y_val):
        return None
    lr = LogisticRegression(max_iter=1000)
    lr.fit([[p] for p in p_val], y_val)
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def export_hgb(clf):
    """Export HistGradientBoostingClassifier to plain lists for pure-Python traversal."""
    import numpy as np
    trees = []
    for stage in clf._predictors:
        for pred in stage:
            nodes = pred.nodes
            names = nodes.dtype.names
            thr_field = "num_threshold" if "num_threshold" in names else "threshold"
            trees.append({
                "feat": [int(v) for v in nodes["feature_idx"]],
                "thr": [float(v) for v in nodes[thr_field]],
                "left": [int(v) for v in nodes["left"]],
                "right": [int(v) for v in nodes["right"]],
                "leaf": [bool(v) for v in nodes["is_leaf"]],
                "val": [float(v) for v in nodes["value"]],
            })
    baseline = float(np.asarray(clf._baseline_prediction).ravel()[0])
    return {"baseline": baseline, "trees": trees, "n_features": len(FEATURE_COLS)}


def parity_ok(entry, clf, X_check):
    """Pure-Python traversal must reproduce sklearn's predict_proba."""
    ps = clf.predict_proba(X_check)[:, 1]
    for x, p_ref in zip(X_check, ps):
        if abs(_trees_prob(entry, list(x)) - float(p_ref)) > 1e-4:
            return False
    return True


def blend_search(rules_p, ai_p, y):
    """Grid-search w minimizing Brier of (1-w)*rules + w*ai on the val fold (P5)."""
    best_w, best_b = 0.0, None
    for step in range(11):
        w = step / 10.0
        b = sum(((1 - w) * r + w * a - t) ** 2 for r, a, t in zip(rules_p, ai_p, y)) / len(y)
        if best_b is None or b < best_b - 1e-12:
            best_w, best_b = w, b
    return best_w, best_b


def main():
    rows = load_rows()
    n = len(rows)
    generated = datetime.now().isoformat(timespec="seconds")
    meta = {"n_total": n, "generated": generated}
    weights_out = {"meta": dict(meta)}
    trees_out = {"meta": dict(meta)}
    blend_out = {"meta": dict(meta), "targets": {t: 0.0 for t in TARGETS}}
    metrics = {"n_total": n, "generated": generated, "targets": {}}
    if rows:
        metrics["train_range"] = [rows[0]["ts"], rows[-1]["ts"]]

    MODEL_DIR.mkdir(exist_ok=True)
    if n < MIN_ROWS:
        weights_out["meta"]["note"] = f"only {n} rows; need {MIN_ROWS} to train — rules-only mode"
        _write_all(weights_out, trees_out, blend_out, metrics)
        print(f"[train] skipped: {n} rows < {MIN_ROWS}")
        return

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    def brier(y, p):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        return float(np.mean((p - y) ** 2))

    for target in TARGETS:
        X, y, idx = build_samples(rows, target)
        report = {"n_samples": len(y), "n_pos": int(sum(y))}
        metrics["targets"][target] = report
        if len(y) < MIN_ROWS or sum(y) < MIN_POSITIVES:
            report["status"] = f"skipped ({sum(y)} positives < {MIN_POSITIVES} or too few rows)"
            print(f"[train] {target}: {report['status']}")
            continue

        split = max(1, int(len(y) * (1 - VAL_FRAC)))
        X_tr, X_va = np.array(X[:split]), np.array(X[split:])
        y_tr, y_va = np.array(y[:split]), np.array(y[split:])
        val_idx = idx[split:]
        report["n_val"] = len(y_va)
        report["n_val_pos"] = int(y_va.sum())
        if y_tr.sum() == 0 or y_va.sum() == 0 or y_tr.sum() == len(y_tr):
            report["status"] = "skipped (a fold has a single class — need more positives)"
            print(f"[train] {target}: {report['status']}")
            continue

        # --- candidate 1: standardized Logistic Regression (existing baseline model)
        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        std[std < 1e-6] = 1e-6
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit((X_tr - mean) / std, y_tr)
        p_lr = lr.predict_proba((X_va - mean) / std)[:, 1]

        # --- candidate 2: gradient-boosted trees (P3)
        w_pos = len(y_tr) / (2.0 * max(1, y_tr.sum()))
        w_neg = len(y_tr) / (2.0 * max(1, (len(y_tr) - y_tr.sum())))
        sw = np.where(y_tr == 1, w_pos, w_neg)
        hgb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, min_samples_leaf=10,
            early_stopping=False, random_state=0,
        )
        hgb.fit(X_tr, y_tr, sample_weight=sw)
        p_hgb = hgb.predict_proba(X_va)[:, 1]

        ap_lr = float(average_precision_score(y_va, p_lr))
        ap_hgb = float(average_precision_score(y_va, p_hgb))
        report["lr"] = {"pr_auc": round(ap_lr, 4), "brier": round(brier(y_va, p_lr), 4)}
        report["trees"] = {"pr_auc": round(ap_hgb, 4), "brier": round(brier(y_va, p_hgb), 4)}

        use_trees = ap_hgb > ap_lr or (ap_hgb == ap_lr and brier(y_va, p_hgb) < brier(y_va, p_lr))
        entry = None
        if use_trees:
            try:
                entry = export_hgb(hgb)
                if not parity_ok(entry, hgb, X_va[: min(20, len(X_va))]):
                    print(f"[train] {target}: tree export parity failed — falling back to LR")
                    entry, use_trees = None, False
            except Exception as e:
                print(f"[train] {target}: tree export failed ({e}) — falling back to LR")
                entry, use_trees = None, False
        p_raw = p_hgb if use_trees else p_lr

        # --- calibration on the val fold (P4)
        cal = fit_platt(list(p_raw), list(y_va))
        p_cal = np.array([apply_cal(float(p), cal) for p in p_raw])
        if brier(y_va, p_cal) > brier(y_va, p_raw):
            cal, p_cal = None, p_raw  # calibration hurt (tiny fold) — drop it

        # --- baselines (P4): persistence + climatology
        p_pers = np.array(persistence_scores(rows, val_idx, target))
        p_clim = np.full(len(y_va), float(y_tr.mean()))
        ap_model = float(average_precision_score(y_va, p_cal))
        ap_pers = float(average_precision_score(y_va, p_pers))
        b_model, b_pers, b_clim = brier(y_va, p_cal), brier(y_va, p_pers), brier(y_va, p_clim)
        report["model"] = "trees" if use_trees else "lr"
        report["calibrated"] = cal is not None
        report["pr_auc"] = round(ap_model, 4)
        report["brier"] = round(b_model, 4)
        report["persistence"] = {"pr_auc": round(ap_pers, 4), "brier": round(b_pers, 4)}
        report["climatology"] = {"brier": round(b_clim, 4)}

        ships = (
            ap_model >= ap_pers - 1e-9 and b_model <= b_pers + 1e-9
            and (ap_model > ap_pers + EPS or b_model < b_pers - EPS)
        )
        report["ships"] = bool(ships)
        if not ships:
            report["status"] = "rules-only (did not beat persistence baseline)"
            print(f"[train] {target}: NOT shipped — PR-AUC {ap_model:.3f} vs pers {ap_pers:.3f}, "
                  f"Brier {b_model:.4f} vs pers {b_pers:.4f}")
            continue

        # --- learned blend weight on the val fold (P5)
        rules_val = [rule_probs(rows[i])[target] for i in val_idx]
        w_blend, b_blend = blend_search(rules_val, list(p_cal), list(y_va))
        blend_out["targets"][target] = w_blend
        report["blend_w"] = w_blend
        report["blend_brier"] = round(b_blend, 4)

        if use_trees:
            entry["cal"] = cal
            entry["n_pos"] = int(sum(y))
            trees_out[target] = entry
        else:
            weights_out[target] = {
                "intercept": float(lr.intercept_[0]),
                "coef": [float(c) for c in lr.coef_[0]],
                "mean": [round(float(m), 4) for m in mean],
                "std": [round(float(s), 4) for s in std],
                "cal": cal,
                "n_pos": int(sum(y)),
                "n_neg": int(len(y) - sum(y)),
            }
        report["status"] = "shipped"
        print(f"[train] {target}: shipped {report['model']} — PR-AUC {ap_model:.3f} "
              f"(pers {ap_pers:.3f}), Brier {b_model:.4f} (pers {b_pers:.4f}), blend w={w_blend}")

    _write_all(weights_out, trees_out, blend_out, metrics)
    shipped = [t for t in TARGETS if metrics["targets"].get(t, {}).get("ships")]
    print(f"[train] done: {len(shipped)}/{len(TARGETS)} targets shipped: {shipped or 'none'}")


def _write_all(weights_out, trees_out, blend_out, metrics):
    WEIGHTS_JSON.write_text(json.dumps(weights_out, indent=2), encoding="utf-8")
    TREES_JSON.write_text(json.dumps(trees_out), encoding="utf-8")
    BLEND_JSON.write_text(json.dumps(blend_out, indent=2), encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
