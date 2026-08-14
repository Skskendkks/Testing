"""Validate portable model artifacts before they are committed by a retraining workflow."""

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import TARGETS, predict_ai

SNAPSHOT_CSV = ROOT / "data" / "snapshots.csv"


def latest_row():
    if not SNAPSHOT_CSV.exists():
        raise SystemExit("[artifact-check] snapshots.csv is missing")
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("[artifact-check] snapshots.csv is empty")
    return rows[-1]


def main():
    row = latest_row()
    try:
        probabilities = predict_ai(row)
    except Exception as exc:
        raise SystemExit(f"[artifact-check] inference failed: {type(exc).__name__}: {exc}") from exc

    invalid = {
        target: probability
        for target, probability in probabilities.items()
        if target not in TARGETS or not isinstance(probability, (int, float))
        or not math.isfinite(probability) or not 0.0 <= probability <= 1.0
    }
    if invalid:
        raise SystemExit(f"[artifact-check] invalid probabilities: {invalid}")
    print(
        f"[artifact-check] ok: snapshot={row.get('ts')} "
        f"shipped_targets={','.join(sorted(probabilities)) or 'none'}"
    )


if __name__ == "__main__":
    main()
