"""Manifest and quality summaries for reproducible F3 CNN datasets."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DATASET_SCHEMA = "f3-cnn-v4"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": round(float(np.min(values)), 3),
        "median": round(float(np.median(values)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "max": round(float(np.max(values)), 3),
    }


def time_gap_minutes(times):
    if len(times) < 2:
        return np.array([], dtype=float)
    parsed = np.array([datetime.fromisoformat(str(value)).timestamp() for value in times], dtype=float)
    return np.diff(parsed) / 60.0


def build_manifest(dataset_path, arrays, *, thresholds_mm, provenance):
    """Describe the exact dataset artifact without embedding arrays in source control."""
    x = arrays["X"]
    y = arrays["y"]
    b = arrays["B"]
    times = np.asarray(arrays["times"]).astype(str)
    label_times = np.asarray(arrays["label_times"]).astype(str)
    lead_minutes = np.asarray(arrays["lead_minutes"], dtype=float)
    gaps = time_gap_minutes(times)
    target_names = [f"rain120_{int(threshold)}mm" for threshold in thresholds_mm]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema": DATASET_SCHEMA,
        "generated_at": _utc_now(),
        "dataset_file": Path(dataset_path).name,
        "dataset_sha256": sha256_file(dataset_path),
        "arrays": {
            "X_shape": [int(value) for value in x.shape],
            "y_shape": [int(value) for value in y.shape],
            "B_shape": [int(value) for value in b.shape],
            "times_shape": [int(value) for value in times.shape],
            "label_times_shape": [int(value) for value in label_times.shape],
            "lead_minutes_shape": [int(value) for value in lead_minutes.shape],
        },
        "time_range": {
            "input_start": times[0] if len(times) else None,
            "input_end": times[-1] if len(times) else None,
            "label_start": label_times[0] if len(label_times) else None,
            "label_end": label_times[-1] if len(label_times) else None,
            "input_gap_minutes": _summary(gaps),
            "gaps_over_30_minutes": int((gaps > 30).sum()),
            "gaps_over_60_minutes": int((gaps > 60).sum()),
            "duplicate_input_timestamps": int(len(times) - len(set(times.tolist()))),
            "lead_minutes": _summary(lead_minutes),
        },
        "targets": {
            target: {"threshold_mm": float(threshold), "positive_count": int(y[:, index].sum())}
            for index, (target, threshold) in enumerate(zip(target_names, thresholds_mm))
        },
        "baseline": {
            "name": "F3 two-hour advection lead maximum",
            "min": round(float(np.min(b)), 4) if len(b) else None,
            "max": round(float(np.max(b)), 4) if len(b) else None,
        },
        "provenance": {
            **provenance,
            "git_commit": os.environ.get("GITHUB_SHA") or "local",
        },
    }


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
