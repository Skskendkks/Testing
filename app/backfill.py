import io
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import N_LEADS, OUT_SIZE, downsample, parse_grid_csv, window_indices

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "grid_dataset.npz"

ARCHIVE_GET = "https://app.data.gov.hk/v1/historical-archive/get-file?url={}&time={}"
TARGET_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"
INPUT_SCALE = 50.0


def http_get_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "SkyWager/1.0 (personal weather nowcast)"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_day(day, verbose=False):
    url = ARCHIVE_GET.format(urllib.parse.quote(TARGET_URL, safe=""), day.strftime("%Y%m%d"))
    data = http_get_bytes(url)
    zf = zipfile.ZipFile(io.BytesIO(data))
    snapshots = []
    for name in zf.namelist():
        if "Gridded_rainfall_nowcast.csv" not in name:
            continue
        text = zf.read(name).decode("utf-8", errors="replace")
        frames, latv, lonv = parse_grid_csv(text)
        if not frames:
            continue
        ilat, ilon = window_indices(latv, lonv)
        if not ilat or not ilon:
            continue
        endings = sorted(frames.keys())
        if len(endings) < N_LEADS:
            continue
        leads = [downsample(frames[e], ilat, ilon) for e in endings[:N_LEADS]]
        ts = datetime.strptime(endings[0], "%Y%m%d%H%M").replace(tzinfo=timezone.utc) - timedelta(hours=8)
        snapshots.append((ts, leads))
    snapshots.sort(key=lambda s: s[0])
    if verbose:
        print(f"[backfill] {day:%Y-%m-%d}: {len(snapshots)} snapshots")
    return snapshots


def build_dataset(days, verbose=True):
    all_snap = []
    for d in days:
        all_snap.extend(fetch_day(d, verbose))
    all_snap.sort(key=lambda s: s[0])
    samples_x = []
    samples_y = []
    times = []
    for ts, leads in all_snap:
        x = np.stack([np.array(g) for g in leads[:3]]) / INPUT_SCALE
        maxmm = float(np.max(np.array(leads[3])))
        y = np.array([1.0 if maxmm >= t else 0.0 for t in [15.0, 25.0, 35.0]], dtype=np.float32)
        samples_x.append(x)
        samples_y.append(y)
        times.append(ts.isoformat())
    X = np.stack(samples_x).astype(np.float32)
    y = np.stack(samples_y).astype(np.float32)
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATASET_PATH, X=X, y=y, times=np.array(times))
    print(f"[backfill] dataset: {X.shape[0]} samples -> {DATASET_PATH} ({DATASET_PATH.stat().st_size / 1e6:.1f} MB)")
    for k, t in enumerate(["rain120_15mm", "rain120_25mm", "rain120_35mm"]):
        print(f"  {t}: {int(y[:, k].sum())} positives")


if __name__ == "__main__":
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    build_dataset(days)
