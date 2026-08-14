"""Persisted health status for the poller and static dashboard."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SITE_DATA_DIR = ROOT / "site" / "data"
HEALTH_JSON = DATA_DIR / "health.json"
SITE_HEALTH_JSON = SITE_DATA_DIR / "health.json"


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_health():
    return _read_json(HEALTH_JSON, {}) or {}


def latest_snapshot_time():
    latest = _read_json(DATA_DIR / "latest.json", {}) or {}
    return latest.get("ts")


def success_status(row, *, f3_available, model_modes=None, model_status=None):
    rain_available = row.get("rain_1h") not in (None, "")
    degraded = not rain_available or not f3_available
    return {
        "status": "degraded" if degraded else "ok",
        "updated_at": now_iso(),
        "last_success_at": row.get("ts"),
        "last_failure_at": None,
        "summary": "poll completed with optional data unavailable" if degraded else "poll completed",
        "components": {
            "hko_weather": "ok",
            "rainfall": "ok" if rain_available else "unavailable",
            "f3_nowcast": "ok" if f3_available else "unavailable",
            "model": model_status or ("rules-only" if model_modes and all(mode == "rules" for mode in model_modes.values()) else "available"),
        },
    }


def failure_status(reason):
    previous = load_health()
    last_success_at = previous.get("last_success_at") or latest_snapshot_time()
    return {
        "status": "failed",
        "updated_at": now_iso(),
        "last_success_at": last_success_at,
        "last_failure_at": now_iso(),
        "summary": reason or "poll workflow failed",
        "components": {
            "hko_weather": "unknown",
            "rainfall": "unknown",
            "f3_nowcast": "unknown",
            "model": "unknown",
        },
    }


def publish(status):
    write_json(HEALTH_JSON, status)
    write_json(SITE_HEALTH_JSON, status)
