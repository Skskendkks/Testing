import gzip
import io
import json
import math
import re
import urllib.request
from datetime import datetime, timezone

BTK_INDEX_URL = "https://ftp.nhc.noaa.gov/atcf/btk/"
BTK_FILE_URL = "https://ftp.nhc.noaa.gov/atcf/btk/b{basin}{num}{year}.dat"
AID_FILE_URL = "https://ftp.nhc.noaa.gov/atcf/aid_public/a{basin}{num}{year}.dat.gz"

HK_LAT = 22.3027
HK_LON = 114.1742
NEAR_RADIUS_KM = 2500
REPORT_RADIUS_KM = 1500


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Testing/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def parse_latlon(s, is_lat):
    m = re.fullmatch(r"(\d+)([NSEW])", s.strip())
    if not m:
        return None
    value = int(m.group(1)) / 10.0
    if is_lat and m.group(2) == "S":
        value = -value
    if not is_lat and m.group(2) == "W":
        value = -value
    return value


def parse_fix(fields):
    lat = parse_latlon(fields[6], True)
    lon = parse_latlon(fields[7], False)
    if lat is None or lon is None:
        return None
    try:
        wind = int(fields[8])
        pressure = int(fields[9] or 0)
    except ValueError:
        return None
    if wind < 20:
        return None
    return {"lat": lat, "lon": lon, "wind": wind, "pressure": pressure}


def parse_btk(text, storm_id):
    fixes = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 10:
            continue
        try:
            dt = datetime.strptime(fields[2], "%Y%m%d%H")
        except ValueError:
            continue
        fix = parse_fix(fields)
        if fix:
            fix["dt"] = dt
            fixes.append(fix)
    return sorted(fixes, key=lambda f: f["dt"])


def parse_aid_forecasts(text):
    out = {}
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 10 or fields[4] != "OFCL":
            continue
        try:
            fhr = int(fields[5])
        except ValueError:
            continue
        fix = parse_fix(fields)
        if fix:
            out[fhr] = fix
    return out


def find_storms():
    try:
        listing = http_get(BTK_INDEX_URL).decode("utf-8", "replace")
    except Exception:
        return []
    year = str(datetime.now(timezone.utc).year)
    storms = []
    for m in re.finditer(r"b([a-z]{2})(\d{2})(\d{4})\.dat", listing):
        basin, num, y = m.group(1).upper(), m.group(2), m.group(3)
        if y == year and basin == "WP":
            storms.append({"basin": basin, "num": num, "id": f"WP{num}"})
    return storms


def fetch_btk(storm):
    url = BTK_FILE_URL.format(basin=storm["basin"].lower(), num=storm["num"], year=datetime.now(timezone.utc).year)
    return http_get(url).decode("utf-8", "replace")


def fetch_aid(storm):
    url = AID_FILE_URL.format(basin=storm["basin"].lower(), num=storm["num"], year=datetime.now(timezone.utc).year)
    raw = http_get(url)
    return gzip.decompress(raw).decode("utf-8", "replace")


def storm_info(storm, fixes, forecasts):
    if not fixes:
        return None
    latest = fixes[-1]
    dist = haversine_km(HK_LAT, HK_LON, latest["lat"], latest["lon"])
    bearing = bearing_deg(HK_LAT, HK_LON, latest["lat"], latest["lon"])
    target_24 = min(forecasts, key=lambda f: abs(f - 24)) if forecasts else None
    forecast = forecasts.get(target_24) if target_24 is not None else None
    info = {
        "id": storm["id"],
        "ts": latest["dt"].strftime("%Y-%m-%dT%H:%M"),
        "lat": round(latest["lat"], 1),
        "lon": round(latest["lon"], 1),
        "wind_kts": latest["wind"],
        "pressure_mb": latest["pressure"] or None,
        "distance_km": round(dist),
        "bearing_deg": round(bearing),
        "forecast_24h_km": round(haversine_km(HK_LAT, HK_LON, forecast["lat"], forecast["lon"])) if forecast else None,
    }
    info["moving_toward_hk"] = bool(info["forecast_24h_km"] and info["forecast_24h_km"] < dist)
    return info


def scan():
    storms = find_storms()
    results = []
    for storm in storms:
        try:
            fixes = parse_btk(fetch_btk(storm), storm["id"])
            forecasts = parse_aid_forecasts(fetch_aid(storm)) if fixes else {}
            info = storm_info(storm, fixes, forecasts)
            if info:
                results.append(info)
        except Exception:
            continue
    near = [s for s in results if s["distance_km"] <= NEAR_RADIUS_KM]
    near.sort(key=lambda s: s["distance_km"])
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scanned": [s["id"] for s in results],
        "nearest": near[0] if near else None,
    }


def snapshot_features(tc_state):
    n = tc_state.get("nearest") or {}
    dist = n.get("distance_km") or 2000
    return {
        "tc_dist_km": min(dist, 2000),
        "tc_wind_kts": n.get("wind_kts") or 0,
        "tc_24h_dist_km": min(n.get("forecast_24h_km") or 2000, 2000),
        "tc_trend_toward": 1 if n.get("moving_toward_hk") else 0,
    }
