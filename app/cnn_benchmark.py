"""Run a reproducible CNN nowcast benchmark instead of serving live forecasts.

The experiment asks one narrow question: do F3-grid CNN inputs add predictive skill
for the selected two-hour rainfall thresholds beyond HKO's own F3 advection nowcast?
It does not issue a public weather forecast or safety alert.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cnn

DATASET_PATH = ROOT / "data" / "grid_dataset.npz"
MODEL_REPORT = ROOT / "model" / "cnn_evaluation.json"
SITE_REPORT = ROOT / "site" / "data" / "cnn_evaluation.json"
MIN_SAMPLES = 200
MIN_HOLDOUT_POSITIVES = 10

TASK = {
    "name": "F3-grid rainfall nowcast skill benchmark",
    "question": "Does the CNN add skill beyond HKO F3's own two-hour advection nowcast?",
    "input": "Four F3 half-hourly forecast grids plus one inter-frame change grid over Hong Kong.",
    "target": "A later F3 lead-0 grid approximately 90–150 minutes after the input snapshot.",
    "evaluation": "Chronological last-20% holdout; PR-AUC and Brier score against the F3 advection baseline.",
    "limitation": "The target is a later F3 product, not independent rain-gauge ground truth. Results measure added skill over this proxy baseline only.",
}


def _iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def publish(report):
    _write(MODEL_REPORT, report)
    _write(SITE_REPORT, report)


def dataset_summary(data):
    x = data["X"] if "X" in data else None
    y = data["y"] if "y" in data else None
    times = data["times"] if "times" in data else None
    summary = {
        "samples": int(x.shape[0]) if x is not None and x.ndim else 0,
        "input_shape": list(x.shape[1:]) if x is not None and x.ndim == 4 else None,
        "targets": list(cnn.V3_TARGETS),
        "positive_counts": y.sum(axis=0).astype(int).tolist() if y is not None and y.ndim == 2 else None,
        "time_start": str(times[0]) if times is not None and len(times) else None,
        "time_end": str(times[-1]) if times is not None and len(times) else None,
        "has_advection_baseline": "B" in data,
    }
    return summary


def validate_dataset(data):
    """Return clear blocking reasons before allocating training compute."""
    reasons = []
    required = {"X", "y", "B", "times"}
    keys = set(data.files if hasattr(data, "files") else data.keys())
    missing = sorted(required - keys)
    if missing:
        reasons.append(f"dataset is missing required arrays: {', '.join(missing)}")
    x = data["X"] if "X" in keys else None
    y = data["y"] if "y" in keys else None
    b = data["B"] if "B" in keys else None
    times = data["times"] if "times" in keys else None
    if x is None or x.ndim != 4 or x.shape[1] != cnn.IN_CH or tuple(x.shape[2:]) != (cnn.SIZE, cnn.SIZE):
        shape = tuple(x.shape) if x is not None else None
        reasons.append(f"expected X shape (N, {cnn.IN_CH}, {cnn.SIZE}, {cnn.SIZE}), got {shape}")
    if y is None or y.ndim != 2 or x is None or y.shape[1] != len(cnn.V3_TARGETS) or y.shape[0] != x.shape[0]:
        reasons.append("y must align with X and contain one column per benchmark threshold")
    if b is not None and (b.ndim != 1 or x is None or b.shape[0] != x.shape[0]):
        reasons.append("B must contain one F3 advection-baseline value per sample")
    if times is not None and (x is None or len(times) != x.shape[0]):
        reasons.append("times must contain one timestamp per sample")
    if x is not None and x.shape[0] < MIN_SAMPLES:
        reasons.append(f"need at least {MIN_SAMPLES} samples for a holdout benchmark; found {x.shape[0]}")
    if not reasons and x is not None and y is not None and times is not None:
        n_val = max(1, int(x.shape[0] * 0.2))
        pos = y[-n_val:].sum(axis=0)
        insufficient = [cnn.V3_TARGETS[i] for i, count in enumerate(pos) if count < MIN_HOLDOUT_POSITIVES]
        if insufficient:
            reasons.append(
                f"chronological holdout needs at least {MIN_HOLDOUT_POSITIVES} positives per target; insufficient: {', '.join(insufficient)}"
            )
        ordered = [str(t) for t in times]
        if ordered != sorted(ordered):
            reasons.append("times are not chronological")
    return reasons


def blocked_report(summary, reasons):
    return {
        "schema_version": 1,
        "generated": _iso_now(),
        "status": "blocked",
        "task": TASK,
        "dataset": summary,
        "blocking_reasons": reasons,
        "next_step": "Rebuild data/grid_dataset.npz with app/backfill.py using the 5-channel schema and B baseline before running this benchmark.",
    }


def completed_report(summary, metrics):
    targets = {}
    for target in cnn.V3_TARGETS:
        metric = metrics[target]
        targets[target] = {
            **metric,
            "conclusion": "added skill versus F3 baseline" if metric.get("beats_baseline") else "no demonstrated added skill versus F3 baseline",
        }
    any_skill = any(item.get("beats_baseline") for item in metrics.values())
    return {
        "schema_version": 1,
        "generated": _iso_now(),
        "status": "completed",
        "task": TASK,
        "dataset": summary,
        "holdout": "last 20% of timestamp-sorted samples; the model never trains on these samples",
        "targets": targets,
        "overall_conclusion": "CNN shows added skill for at least one tested threshold" if any_skill else "CNN does not yet demonstrate added skill beyond the F3 baseline",
    }


def run(epochs=40, seed=0, quiet=False):
    if not DATASET_PATH.exists():
        report = blocked_report({"samples": 0, "input_shape": None, "targets": cnn.V3_TARGETS, "positive_counts": None,
                                 "time_start": None, "time_end": None, "has_advection_baseline": False},
                                ["data/grid_dataset.npz is missing"])
        publish(report)
        return report
    data = np.load(DATASET_PATH, allow_pickle=False)
    summary = dataset_summary(data)
    reasons = validate_dataset(data)
    if reasons:
        report = blocked_report(summary, reasons)
        publish(report)
        return report
    weights, metrics = cnn.train(data["X"], data["y"], B=data["B"], epochs=epochs, seed=seed, quiet=quiet)
    cnn.save_weights(weights, metrics, int(data["X"].shape[0]))
    report = completed_report(summary, metrics)
    publish(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN nowcast skill; never serves a live forecast")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run(epochs=args.epochs, seed=args.seed, quiet=args.quiet)
    print(f"[cnn-benchmark] status={report['status']}")
    if report["status"] == "blocked":
        for reason in report["blocking_reasons"]:
            print(f"[cnn-benchmark] blocked: {reason}")
    else:
        print(f"[cnn-benchmark] {report['overall_conclusion']}")


if __name__ == "__main__":
    main()
