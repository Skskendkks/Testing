import csv
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import TARGETS, TARGET_LABELS, blend_weights, predict_ai
import jtwc
from notify import ai_alert_keys, load_notified, save_notified, send_email
from rules import rule_probs
from data_quality import (
    DATA_SCHEMA_VERSION,
    RAIN_SOURCE_HKO_DISTRICT_MAXIMA,
    validate_row,
)
from health import publish as publish_health, success_status
import grid as gridmod

try:
    import cnn as cnnmod
except ImportError:
    cnnmod = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
SITE_DATA_DIR = ROOT / "site" / "data"
SNAPSHOT_CSV = DATA_DIR / "snapshots.csv"
LATEST_JSON = DATA_DIR / "latest.json"
HISTORY_JSON = DATA_DIR / "history.json"
LAST_WARN = STATE_DIR / "last_warnings.json"

WEATHER_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
WARNSUM_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en"

HK_OFFSET = timedelta(hours=8)

CSV_COLUMNS = [
    "data_schema",
    "rain_source",
    "ts",
    "temp_mean",
    "hum_mean",
    "rain_total",
    "rain_main",
    "rain_1h",
    "rain_3h",
    "hum_1h_delta",
    "temp_1h_delta",
    "hour",
    "season",
    "w_TCSGNL",
    "w_TC1",
    "w_TC3",
    "w_TC8",
    "w_RAIN_AMBER",
    "w_RAIN_RED",
    "w_RAIN_BLACK",
    "w_WTS",
    "w_WMSGNL",
    "w_WL",
    "w_WFNTSA",
    "w_WHOT",
    "w_WCOLD",
    "w_WFROST",
    "w_WFIRE",
    "tc_dist_km",
    "tc_wind_kts",
    "tc_24h_dist_km",
    "tc_trend_toward",
    "tc_dist_rate",
    "f3_max",
    "f3_mean",
    "f3_trend",
]

WARNSUMS_TO_FLAGS = {
    "WTCSGNL": "w_TCSGNL",
    "WRAIN": "w_RAIN",
    "WTS": "w_WTS",
    "WMSGNL": "w_WMSGNL",
    "WL": "w_WL",
    "WFNTSA": "w_WFNTSA",
    "WHOT": "w_WHOT",
    "WCOLD": "w_WCOLD",
    "WFROST": "w_WFROST",
    "WFIRE": "w_WFIRE",
}

LEVEL_NAMES = {
    "w_TC1": "TC Signal No. 1",
    "w_TC3": "TC Signal No. 3",
    "w_TC8": "TC Signal No. 8/9/10",
    "w_RAIN_AMBER": "Amber Rainstorm",
    "w_RAIN_RED": "Red Rainstorm",
    "w_RAIN_BLACK": "Black Rainstorm",
}


def http_get(url, attempts=3):
    """Fetch JSON with bounded retries for transient HKO network failures."""
    error = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            error = exc
            if attempt < attempts:
                wait_seconds = attempt
                print(f"[poll] request failed ({type(exc).__name__}); retrying in {wait_seconds}s")
                time.sleep(wait_seconds)
    raise RuntimeError(f"request failed after {attempts} attempts: {error}") from error


def parse_levels(messages):
    flags = {}
    for text in messages:
        up = text.upper()
        if "RAINSTORM" in up:
            flags["w_RAIN_AMBER"] = "AMBER" in up
            flags["w_RAIN_RED"] = "RED" in up
            flags["w_RAIN_BLACK"] = "BLACK" in up
        if "TROPICAL CYCLONE" in up and "SIGNAL" in up:
            m = re.search(r"(\d{1,2})", up)
            if m:
                level = int(m.group(1))
                flags["w_TC1"] = level == 1
                flags["w_TC3"] = level == 3
                flags["w_TC8"] = level >= 8
    return flags


def load_csv_rows():
    if not SNAPSHOT_CSV.exists():
        return []
    with open(SNAPSHOT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_within(rows, minutes):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        if ts >= cutoff:
            out.append(r)
    return out


def _f(row, key):
    return float(row.get(key, 0.0) or 0.0)


def f3_features(leads):
    """P6: scalar summaries of the F3 gridded nowcast for the tabular models."""
    if not leads:
        return {"f3_max": 0.0, "f3_mean": 0.0, "f3_trend": 0.0}
    maxes = [max(max(r) for r in g) for g in leads]
    means = [sum(sum(r) for r in g) / (len(g) * len(g[0])) for g in leads]
    return {
        "f3_max": round(max(maxes), 2),
        "f3_mean": round(sum(means) / len(means), 3),
        "f3_trend": round(maxes[-1] - maxes[0], 2),
    }


def build_row(weather, warnsum, messages, levels, prior_rows, tc_feats, f3_feats=None):
    now = datetime.now(timezone.utc)
    temps = [d["value"] for d in weather.get("temperature", {}).get("data", []) if isinstance(d.get("value"), (int, float))]
    hums = [d["value"] for d in weather.get("humidity", {}).get("data", []) if isinstance(d.get("value"), (int, float))]
    rains = [d["max"] for d in weather.get("rainfall", {}).get("data", []) if isinstance(d.get("max"), (int, float))]
    temp_mean = round(sum(temps) / len(temps), 1) if temps else None
    hum_mean = round(sum(hums) / len(hums), 1) if hums else None
    # HKO supplies a one-hour window per district. The sum is an aggregate of
    # district maxima, not a geographic accumulation; the peak is the only
    # directly comparable one-hour rainfall feature. Do not subtract two
    # independent one-hour windows to manufacture 1h/3h accumulations.
    rain_total = round(sum(rains), 1) if rains else None
    rain_main = round(max(rains), 1) if rains else None
    rain_1h = rain_main
    rain_3h = ""

    recent_60 = rows_within(prior_rows, 60)
    hum_1h_delta = (hum_mean - _f(recent_60[0], "hum_mean")) if recent_60 and hum_mean is not None else 0.0
    temp_1h_delta = (temp_mean - _f(recent_60[0], "temp_mean")) if recent_60 and temp_mean is not None else 0.0

    hk_now = now + HK_OFFSET
    row = {
        "data_schema": DATA_SCHEMA_VERSION,
        "rain_source": RAIN_SOURCE_HKO_DISTRICT_MAXIMA,
        "ts": now.isoformat(timespec="seconds"),
        "temp_mean": temp_mean if temp_mean is not None else "",
        "hum_mean": hum_mean if hum_mean is not None else "",
        "rain_total": rain_total if rain_total is not None else "",
        "rain_main": rain_main if rain_main is not None else "",
        "rain_1h": round(rain_1h, 1) if rain_1h is not None else "",
        "rain_3h": rain_3h,
        "hum_1h_delta": round(hum_1h_delta, 1),
        "temp_1h_delta": round(temp_1h_delta, 1),
        "hour": hk_now.hour,
        "season": 1 if 5 <= hk_now.month <= 11 else 0,
    }
    for code, flag in WARNSUMS_TO_FLAGS.items():
        row[flag] = 1 if code in warnsum else 0
    for flag in ("w_TC1", "w_TC3", "w_TC8", "w_RAIN_AMBER", "w_RAIN_RED", "w_RAIN_BLACK"):
        row[flag] = 1 if levels.get(flag) else 0
    for key, value in tc_feats.items():
        row[key] = value
    # P6: JTWC distance rate of change (km/h; negative = approaching)
    prev_dist = _f(recent_60[0], "tc_dist_km") if recent_60 and recent_60[0].get("tc_dist_km") not in ("", None) else None
    row["tc_dist_rate"] = round(tc_feats.get("tc_dist_km", 2000) - prev_dist, 1) if prev_dist else 0.0
    for key, value in (f3_feats or f3_features(None)).items():
        row[key] = value
    validate_row(row)
    return row, temps, hums, rains


def blend_probs(rules_p, ai_p, w_map):
    """P5: per-target learned blend weight (falls back to rules where AI is absent)."""
    out = {}
    for t in TARGETS:
        r = rules_p.get(t, 0.0)
        if t in ai_p:
            w = w_map.get(t, 0.0)
            out[t] = round(r * (1 - w) + ai_p[t] * w, 3)
        else:
            out[t] = round(r, 3)
    return out


def prediction_details(rules_p, ai_p, w_map, probabilities):
    """Explain whether each displayed probability is rules-only or blended."""
    details = {}
    for target in TARGETS:
        mode = "rules"
        if target in ai_p and w_map.get(target, 0.0) > 0:
            mode = "blended"
        details[target] = {
            "final": probabilities.get(target, rules_p.get(target, 0.0)),
            "rule": rules_p.get(target, 0.0),
            "ai": ai_p.get(target),
            "blend_weight": round(w_map.get(target, 0.0), 2) if target in ai_p else 0.0,
            "mode": mode,
        }
    return details


def active_warning_names(warnsum):
    return [f"{entry.get('name', code)} ({entry.get('actionCode', '')})" for code, entry in warnsum.items()]


def official_changes(prev_state, warnsum, levels, messages):
    changes = []
    prev_warnsum = prev_state.get("warnsum", {}) if prev_state else {}
    prev_levels = prev_state.get("levels", {}) if prev_state else {}
    if not prev_state:
        return changes, False
    for code, entry in warnsum.items():
        name = entry.get("name", code)
        action = entry.get("actionCode", "")
        if code not in prev_warnsum:
            changes.append(f"NEW: {name} ({action})")
        elif prev_warnsum[code].get("actionCode") != action and action:
            changes.append(f"UPDATE: {name} ({action})")
    for code, entry in prev_warnsum.items():
        if code not in warnsum:
            changes.append(f"CANCELED: {entry.get('name', code)}")
    for flag in ("w_TC1", "w_TC3", "w_TC8"):
        cur = 1 if levels.get(flag) else 0
        prev = 1 if prev_levels.get(flag) else 0
        if cur and not prev:
            changes.append(f"ESCALATED: {LEVEL_NAMES[flag]} now in force")
        elif not cur and prev:
            changes.append(f"DOWNGRADED: {LEVEL_NAMES[flag]} no longer in force")
    for flag in ("w_RAIN_AMBER", "w_RAIN_RED", "w_RAIN_BLACK"):
        cur = 1 if levels.get(flag) else 0
        prev = 1 if prev_levels.get(flag) else 0
        if cur and not prev:
            changes.append(f"ESCALATED: {LEVEL_NAMES[flag]} now in force")
        elif not cur and prev:
            changes.append(f"DOWNGRADED: {LEVEL_NAMES[flag]} no longer in force")
    return changes, True


def write_csv(rows, new_row):
    validate_row(new_row)
    DATA_DIR.mkdir(exist_ok=True)
    all_rows = rows + [new_row]
    with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    return all_rows


def build_history(rows):
    buckets = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    for r in rows:
        if r.get("data_schema") != DATA_SCHEMA_VERSION:
            continue
        ts = datetime.fromisoformat(r["ts"])
        if ts < cutoff:
            continue
        key = (ts + HK_OFFSET).strftime("%Y-%m-%dT%H")
        b = buckets.setdefault(key, {"temps": [], "hums": [], "rain": [], "rain1h": [], "warns": 0})
        if r.get("temp_mean") != "":
            b["temps"].append(_f(r, "temp_mean"))
        if r.get("hum_mean") != "":
            b["hums"].append(_f(r, "hum_mean"))
        # For the chart, use the directly observed one-hour peak rather than
        # the sum of district maxima, which is not a territorial accumulation.
        b["rain"].append(_f(r, "rain_1h"))
        b["rain1h"].append(_f(r, "rain_1h"))
        active = sum(1 for k, v in r.items() if k and k.startswith("w_") and v == "1")
        b["warns"] = max(b["warns"], active)
    out = []
    for key in sorted(buckets):
        b = buckets[key]
        out.append({
            "h": key,
            "temp": round(sum(b["temps"]) / len(b["temps"]), 1) if b["temps"] else None,
            "hum": round(sum(b["hums"]) / len(b["hums"]), 1) if b["hums"] else None,
            "rain": round(max(b["rain"]), 1) if b["rain"] else None,
            "rain1h": round(max(b["rain1h"]), 1) if b["rain1h"] else None,
            "warns": b["warns"],
        })
    return out[-168:]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pages_url():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo:
        owner = repo.split("/")[0]
        return f"https://{owner}.github.io/{repo.split('/')[1]}/"
    return "https://<your-org>.github.io/testing/"


def main():
    weather = http_get(WEATHER_URL)
    warnsum = http_get(WARNSUM_URL)
    messages = weather.get("warningMessage", []) or []
    levels = parse_levels(messages)
    tc_state = jtwc.scan()
    tc_feats = jtwc.snapshot_features(tc_state)

    snap = None
    try:
        snap = gridmod.fetch_snapshot()
    except Exception as e:
        print(f"[poll] F3 grid fetch skipped: {e}")

    rows = load_csv_rows()
    f3_feats = f3_features(snap["leads"]) if snap else None
    row, temps, hums, rains = build_row(weather, warnsum, messages, levels, rows, tc_feats, f3_feats)

    rules_p = rule_probs(row)
    ai_p = predict_ai(row)
    w_map = blend_weights()
    probs = blend_probs(rules_p, ai_p, w_map)

    v3 = None
    v3_grid_ts = None
    if cnnmod is not None and snap:
        try:
            v3_grid_ts = snap["ts"]
            v3 = cnnmod.predict_frames(snap["leads"])
            if v3:
                probs.update(v3)
        except Exception as e:
            print(f"[poll] v3 cnn skipped: {e}")

    details = prediction_details(rules_p, ai_p, w_map, probs)
    if v3:
        for target, probability in v3.items():
            details[target] = {"final": probability, "rule": None, "ai": probability, "blend_weight": 1.0, "mode": "cnn"}

    prev_state = None
    if LAST_WARN.exists():
        with open(LAST_WARN, encoding="utf-8-sig") as f:
            prev_state = json.load(f)
    changes, has_prev = official_changes(prev_state, warnsum, levels, messages)

    notify_lines = []
    for c in changes:
        notify_lines.append(f"* {c}")
    if has_prev and changes:
        notify_lines.append("")
        notify_lines.append("Official HKO warning state changed. Details in email.")

    now = datetime.now(timezone.utc)
    notified = load_notified()
    alert_keys = ai_alert_keys(probs, notified, now)
    for k in alert_keys:
        mode = details.get(k, {}).get("mode", "rules")
        mode_label = "rules-only" if mode == "rules" else ("rules + AI" if mode == "blended" else mode.upper())
        notify_lines.append(f"* {TARGET_LABELS[k]}: probability {probs[k]:.0%} ({mode_label} lead alert)")

    nearest_tc = tc_state.get("nearest")
    tc_line = None
    if nearest_tc and nearest_tc["distance_km"] <= jtwc.REPORT_RADIUS_KM:
        n = nearest_tc
        toward = "closing in" if n.get("moving_toward_hk") else "not closing"
        tc_line = (
            f"Tropical cyclone {n['id']}: {n['distance_km']} km from HK, bearing {n['bearing_deg']}°, "
            f"wind {n['wind_kts']} kts, pressure {n['pressure_mb']} mb, 24h forecast distance "
            f"{n['forecast_24h_km']} km ({toward})"
        )

    if notify_lines and os.environ.get("DISABLE_EMAIL") != "1":
        hk_now = now + HK_OFFSET
        body_lines = [f"Testing alert — {hk_now.strftime('%Y-%m-%d %H:%M')} HKT", ""]
        body_lines.extend(notify_lines)
        body_lines.append("")
        body_lines.append("Experimental nowcast probabilities (next 1-6h):")
        for t in TARGETS:
            mode = details.get(t, {}).get("mode", "rules")
            mode_label = "rules-only" if mode == "rules" else ("rules + AI" if mode == "blended" else mode.upper())
            body_lines.append(f"  {TARGET_LABELS[t]}: {probs[t]:.0%} [{mode_label}]")
        if tc_line:
            body_lines.append("")
            body_lines.append(tc_line)
        body_lines.append("")
        body_lines.append(f"Dashboard: {pages_url()}")
        body_lines.append("")
        body_lines.append("Experimental, unofficial prediction. Always check https://www.hko.gov.hk for official warnings.")
        if send_email("[Testing] Weather alert", "\n".join(body_lines)):
            for k in alert_keys:
                notified[k] = now.isoformat(timespec="seconds")

    rows = write_csv(rows, row)
    write_json(LATEST_JSON, {
        "data_schema": row["data_schema"],
        "rain_source": row["rain_source"],
        "ts": row["ts"],
        "temp_mean": row["temp_mean"],
        "hum_mean": row["hum_mean"],
        "rain_total": row["rain_total"],
        "rain_main": row["rain_main"],
        "rain_1h": row["rain_1h"],
        "rain_3h": row["rain_3h"],
        "active_warnings": active_warning_names(warnsum),
        "special_tips": weather.get("specialWxTips", []),
        "predictions": probs,
        "prediction_details": details,
        "v3_grid_ts": v3_grid_ts,
        "official_levels": {k: v for k, v in LEVEL_NAMES.items() if levels.get(k)},
        "tc": tc_state.get("nearest"),
        "tc_scanned": tc_state.get("scanned", []),
        "blend_ai_weight": {t: round(w_map.get(t, 0.0), 2) for t in TARGETS},
    })
    write_json(HISTORY_JSON, build_history(rows))
    write_json(SITE_DATA_DIR / "latest.json", json.loads(LATEST_JSON.read_text(encoding="utf-8")))
    write_json(SITE_DATA_DIR / "history.json", json.loads(HISTORY_JSON.read_text(encoding="utf-8")))
    model_modes = {target: detail["mode"] for target, detail in details.items()}
    publish_health(success_status(row, f3_available=snap is not None, model_modes=model_modes))

    STATE_DIR.mkdir(exist_ok=True)
    write_json(LAST_WARN, {"warnsum": warnsum, "levels": levels})
    save_notified(notified)

    print(f"[poll] {row['ts']} temp={row['temp_mean']} hum={row['hum_mean']} rain={row['rain_total']}mm")
    print(f"[poll] probs: " + ", ".join(f"{t}={p}" for t, p in probs.items()))
    if tc_line:
        print(f"[poll] {tc_line}")


if __name__ == "__main__":
    main()
