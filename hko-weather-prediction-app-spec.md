# HKO Weather & Warning Prediction App — Product Spec

## 1. Goal

Build an app that:
1. Pulls live and historical weather data from the Hong Kong Observatory (HKO) Open Data API.
2. Uses a machine learning / AI model to predict, a few hours ahead, the likelihood of:
   - Tropical Cyclone Warning Signal: No. 1, 3, 8, 9, 10
   - Strong Monsoon Signal (No. 11 sometimes referenced informally — clarify with HKO docs; official signals are 1/3/8/9/10)
   - Rainstorm Warning: Amber, Red, Black
   - Other warnings: Thunderstorm (WTS), Landslip (WL), Flooding in northern New Territories (WFNTSA)
3. Shows the user a short-term (1–6 hour) forecast/nowcast plus warning probability, not just raw HKO data.

This is technically feasible as a *nowcasting / warning-probability classifier*, not as a full physics-based weather model. Treat HKO's own forecasts and warnings as ground truth labels to train against.

---

## 2. Data Sources (all free, no API key required)

### 2.1 Real-time current weather — `rhrread`
```
https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en
```
Returns (per ~10 min update):
- `temperature.data[]` — station, value (°C)
- `humidity.data[]` — station, value (%)
- `rainfall.data[]` — district, max/min (mm), main flag
- `uvindex.data[]`
- `warningMessage[]` — active warning texts
- `icon[]` — current icon codes

### 2.2 Weather warning summary — `warnsum`
```
https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en
```
Warning codes:
- `WTCSGNL` — Tropical Cyclone Warning Signal (No. 1/3/8/9/10)
- `WRAIN` — Rainstorm Warning (Amber/Red/Black)
- `WTS` — Thunderstorm Warning
- `WMSGNL` — Strong Monsoon Signal
- `WL` — Landslip Warning
- `WFNTSA` — Flooding in northern New Territories
- `WHOT` / `WCOLD` / `WFROST` / `WFIRE` — Hot/Cold/Frost/Fire Danger

### 2.3 Detailed warning info — `warningInfo`
```
https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warningInfo&lang=en
```
Returns full text + issue/cancel timestamps for active warnings — use this to build historical labeled events.

### 2.4 9-day forecast — `fnd`
```
https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en
```

### 2.5 Local weather forecast (next few hours narrative) — `flw`
```
https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en
```

### 2.6 Historical climate data — `opendata.php`
```
https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP&station=HKO&year=2020&format=json
```
Use `CLMTEMP`, `CLMMAXT`, `CLMMINT`, `CLMWSPD`, `CLMPRES` etc. to build long historical training sets.

### 2.7 Radar / satellite imagery (optional, for CNN-based nowcasting)
HKO publishes rainfall radar images — useful if you want to do image-based rain nowcasting (harder, optional v2 feature).

Full API doc (all fields, all languages): `https://www.hko.gov.hk/en/weatherAPI/doc/files/HKO_Open_Data_API_Documentation.pdf`

---

## 3. What Is Realistically Predictable

| Target | Feasibility | Approach |
|---|---|---|
| Rain / no rain next 1-3 hrs | High | Nowcasting classifier using recent rainfall trend, humidity, pressure drop |
| Amber Rainstorm probability | Medium-High | Binary/multi-class classifier trained on historical warning issue times + weather features |
| Red / Black Rainstorm probability | Medium | Same as above; fewer historical events = less training data, expect lower accuracy |
| Typhoon Signal No.1/3 | Medium | Mostly driven by tropical cyclone track/distance — better modeled from JTWC/HKO cyclone track data than local station data alone |
| Typhoon Signal No.8/9/10 | Lower (rare events) | Very few historical events (10/yr region-wide, No.10 extremely rare) — treat as a rules+ML hybrid, not pure ML |
| Thunderstorm Warning | Medium-High | Correlates well with humidity, pressure, recent lightning/rain trends |

**Be honest with the next AI/dev**: full typhoon-signal prediction from scratch is very hard with limited historical samples (Hong Kong issues Signal 8 only a few times per year). A realistic v1 scope is:
- Rain/no-rain + rainstorm-warning-level nowcasting (0–3 hr) — this is achievable and useful.
- Typhoon signal prediction should mostly **reuse HKO's own current signal + forecast** rather than being predicted from raw weather alone, or should ingest tropical cyclone track data as a feature, not just local station readings.

---

## 4. Suggested Architecture

```
┌─────────────┐    ┌───────────────┐    ┌──────────────┐    ┌───────────┐
│ HKO Open API│───▶│ Data Ingestion │───▶│ Feature Store│───▶│ ML Model  │
│ (rhrread,   │    │ (cron every    │    │ (rolling     │    │ (nowcast  │
│ warnsum,    │    │ 5-10 min)      │    │ windows,     │    │ classifier)│
│ warningInfo,│    │                │    │ lag features)│    │           │
│ fnd, CLM*)  │    └───────────────┘    └──────────────┘    └─────┬─────┘
└─────────────┘                                                    │
                                                                     ▼
                                                          ┌─────────────────┐
                                                          │ Prediction API  │
                                                          │ (probability of │
                                                          │ each warning)   │
                                                          └────────┬────────┘
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │ App UI          │
                                                          │ (web/mobile)    │
                                                          │ push notify     │
                                                          └─────────────────┘
```

### 4.1 Data Ingestion
- Cron job / scheduled worker polls `rhrread`, `warnsum`, `warningInfo` every 5–10 minutes.
- Store raw JSON snapshots + parsed rows in a time-series DB (SQLite for MVP, TimescaleDB/InfluxDB for scale).
- Backfill historical training data using `opendata.php` (CLM* series) for at least 3–5 years, plus scrape/store historical warning issue/cancel times from `warningInfo` going forward (HKO doesn't provide deep warning history via API — you must start logging from day one, or find a third-party historical warning dataset).

### 4.2 Feature Engineering
Per time step, compute:
- Current temp, humidity, pressure, wind, rainfall (multi-station)
- Rolling stats: 30 min / 1 hr / 3 hr / 6 hr mean, delta, trend (pressure drop rate is a strong storm signal)
- Time features: hour of day, month, is-typhoon-season (May–Nov)
- Lag features: value N steps ago
- Current active warning state (one-hot) as an input feature too (warnings often persist/escalate)

### 4.3 Model Choices
- **Baseline**: Logistic Regression / XGBoost per warning type (multi-label binary classifiers) — fast, interpretable, works with limited data.
- **v2**: LSTM / GRU / Temporal Fusion Transformer over the rolling feature window for better temporal pattern capture.
- **Typhoon-specific**: augment with tropical cyclone best-track data (JTWC / HKO tropical cyclone track) — distance & bearing to HK, cyclone intensity, forecast track — as extra features, since local-only station data is a weak predictor for typhoon signals.

### 4.4 Prediction API
- Expose `/predict` endpoint returning probability (0–1) for each warning type over the next 1/3/6 hr horizon.
- Also return the *current official* HKO warning as a baseline/fallback so the UI can compare "AI prediction" vs "official state."

### 4.5 App UI
- Simple dashboard: current conditions, AI probability bars per warning type, push notification when probability crosses a threshold (e.g., >60%).
- Always disclose this is an unofficial, experimental prediction and link to the official HKO warning for safety-critical decisions.

---

## 5. MVP Scope (Recommended First Build)

1. Data collector service that polls `rhrread` + `warnsum` + `warningInfo` every 5 min and stores to local DB.
2. After 2–4 weeks of collected data, train a simple XGBoost classifier for:
   - Rain probability next 1–3 hrs
   - Amber Rainstorm probability next 1–3 hrs
3. Simple web dashboard (can reuse the `website-building` skill) showing current HKO data + your model's short-term probability.
4. Compare against HKO's own `flw` (local forecast text) as a sanity check baseline.
5. Defer Red/Black Rainstorm and Typhoon Signal 8+ prediction to v2, since they need longer historical data collection and cyclone-track features.

---

## 6. Legal / Safety Notes

- HKO Open Data is free for reuse (with attribution), per HKO's Terms and Conditions.
- This app is **not an official forecast**. Any UI must clearly state predictions are experimental/unofficial and that users should always refer to HKO's official warnings for safety decisions, especially for severe typhoon/rainstorm signals.
- Do not use this app as sole basis for evacuation, work, or school suspension decisions.

---

## 7. Reference Links

- Open Data API doc: https://www.hko.gov.hk/en/weatherAPI/doc/files/HKO_Open_Data_API_Documentation.pdf
- Open Data intro: https://www.hko.gov.hk/en/abouthko/opendata_intro.htm
- Weather API base: https://data.weather.gov.hk/weatherAPI/opendata/weather.php
- Climate/Opendata base: https://data.weather.gov.hk/weatherAPI/opendata/opendata.php
- Earth Weather (AI model viewer, for reference/comparison): https://maps.hko.gov.hk/wxviewer

---

## 8. Implementation Update Log (actual build vs. spec)

Summary of how the shipped app (v1 → v3) maps to — and deviates from — this spec.

### 8.1 Data sources — confirmed vs. corrected

| Spec | Reality (verified against live endpoints) |
|---|---|
| §2.1 `rhrread` real-time | Used by poller (`app/fetch.py`). No historical archive — labels must be accumulated from day one. |
| §2.2 `warnsum` | Used by poller for warning-change detection + email alerts. No historical archive. |
| §2.3 `warningInfo` | **Not used** — never required: `rhrread.warningMessage` + `warnsum` give the level/action data the models need. |
| §2.4/§2.5 `fnd`/`flw` | Not used (forecast text adds no signal for the classifier). |
| §2.6 `opendata.php` CLM* series | **Correction**: deep daily history IS available without the legacy `opendata.php` endpoint — data.gov.hk serves per-station CSVs since 1884+ (e.g. `data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_RF_ALL.csv`; also `TEMP`, `RH`, `MSLP`). Daily scale only (not hourly), so not yet ingested; candidate feature enrichment for a future release. |
| §2.7 Radar imagery | **Correction**: not exposed via HKO Open Data and no public archive — a radar-image CNN is **not feasible**. |
| New: gridded rainfall nowcast (F3) | HKO's `hko_data/F3/Gridded_rainfall_nowcast.csv`: 121×121 grid at ~2 km, 4 half-hourly lead frames, updated every 12 min. **Full history available** from the data.gov.hk historical archive (`app.data.gov.hk/v1/historical-archive/get-file?url=…&time=YYYYMMDD`, daily ZIPs, ~15-min snapshots). This is the training data for the v3 CNN. |
| New: JTWC/NOAA ATCF (V2) | Official cyclone best-track/forecast lines give distance/bearing/wind/pressure from HK as model features (`app/jtwc.py`). |

### 8.2 Architecture — as built

- Runs entirely on GitHub Actions (student Pro budget): `poll` (hourly) runs `app/fetch.py`; `retrain` (03:00 UTC + after each poll) runs `app/train.py` and `app/cnn.py`.
- No DB: `data/snapshots.csv` (time-series rows, committed), `state/*.json` (warning delta + alert cooldown), `data/latest.json` + `site/data/*` (static dashboard).
- `rhrread`/`warnsum` snapshots are hourly, not 5–10 min (§4.1) — deliberate budget trade-off; 20-min polling was evaluated but exceeds the 3000-min/month Actions allowance when combined with retrains.

### 8.3 Models — as built

| Spec (§4.3) | Built |
|---|---|
| Baseline LR/XGBoost | Shipped: one balanced **Logistic Regression per target** (4 targets: `rain_1h`, `amber_3h`, `red_3h`, `tc3_6h`), sklearn nightly, coefficients exported to `model/weights.json`; pure-Python sigmoid inference. |
| v2 LSTM/GRU/TFT | Not needed — temporal info carried by engineered deltas (1h/3h rainfall, humidity delta) + lookahead labels; JTWC track features handle the typhoon case (§4.3 typhoon bullet). |
| v3 (new) | **Pure-NumPy ConvNet** (2 conv layers → dense → 3 sigmoid heads) nowcasting heavy rain ~2h ahead from the last 3 F3 lead frames (HK window, 32×32). Trained on archived grids (`app/backfill.py` → `data/grid_dataset.npz` → `model/cnn_weights.json`). Current: 650 samples / 7 days, val accuracy 0.92 for ≥15 mm/30 min. |
| Hybrid | Final probability = `rules × (1-w) + AI × w`; `w` ramps 0.2→0.5 once a model trains (was 0.0 below 200 rows — fixed so the AI actually contributes). |

### 8.4 Fixes in this update

- **AI model never activated**: training required 200 rows (rules-only below that) and blend weight was 0.0 below 200 rows → lowered to ≥48 rows / ≥2 positives and a nonzero blend floor once trained.
- **Label horizons now cadence-robust**: lookahead converted from rows to hours (`rows_per_hour()` in `app/train.py`), so faster/slower polling no longer silently changes label windows.
- **Email alerts silently skipped**: `send_email` swallowed missing SMTP secrets / SMTP errors. Root cause of "gov warning issued but no message" — verify `SMTP_USER`/`SMTP_APP_PASSWORD`/`NOTIFY_TO` repo secrets; alerts fire only on the single poll that detects the change.

### 8.5 Status vs. spec sections

- §3 targets: implemented for rain/Amber/Red/Typhoon-3 probabilities (rules + LR + CNN). No.8/9/10 left as rules-only signals (rare events, per spec's honest note).
- §4.1: hourly polling; history backfill via F3 archive + daily CSVs (see 8.1).
- §4.4/4.5: implemented as static JSON dashboard (no API server) + email notifications (>60% threshold with 6-h cooldown, plus official warning-change emails).
- §5 MVP: superseded by the shipped v1-v3 feature set above.

---

## 9. v4 Accuracy Upgrade Plan (prediction quality)

Goal of v4: raise real prediction skill, not just add features. The single biggest accuracy bottleneck today is **training data volume and positive-sample count** — the LR models have only weeks of hourly rows (a handful of Amber/Red/TC positives), and the CNN has 650 samples / 7 days, so its 0.92 val accuracy is likely inflated by class imbalance (predicting "no heavy rain" is right ~90%+ of the time). Fix data first, models second.

### 9.1 Correction to §8.1: historical labels DO exist

Two findings that supersede "labels must be accumulated from day one":

| Source | What it gives | Coverage |
|---|---|---|
| **HKO Warnings/Signals Database** (`hko.gov.hk/en/wxinfo/climat/warndb/warndba.shtml`) | Official issue/cancel times per warning | Rainstorm since **1998**, TC signals since **1946**, Thunderstorm since 1967, SMS since 1950, Landslip since 1983 |
| **data.gov.hk historical archive of `warnsum` and `rhrread`** (same `app.data.gov.hk/v1/historical-archive` mechanism already used for F3) | Timestamped snapshots of past warnings **and** past station readings (temp/RH/rainfall per district) | Multi-year, monthly/daily ZIP archives |

Consequence: the tabular training set is no longer limited by our own polling start date. Backfilling archived `rhrread` + `warnsum` snapshots turns weeks of data into **years**, and warndb provides clean event labels for validation/cross-checking the snapshot-derived labels.

### 9.2 Upgrade tasks (priority order)

| # | Task | Change | Expected accuracy impact |
|---|---|---|---|
| P1 | **Tabular history backfill** | New `app/backfill_tabular.py`: download archived `rhrread`+`warnsum` ZIPs (start with the last 2–3 wet seasons, Apr–Oct), parse into the same `snapshots.csv` schema, dedupe by timestamp. Use warndb to sanity-check label extraction. | Largest single win. Amber/Red positives go from single digits to hundreds of events; `red_3h` and `tc3_6h` become genuinely trainable. |
| P2 | **Event-oriented F3 backfill** | Use warndb dates (all Amber/Red/Black days in archive range) to pick which daily F3 ZIPs to download, plus an equal number of random quiet days for negatives. Extend `data/grid_dataset.npz` from 7 days to full wet seasons. | Fixes CNN class imbalance; val metrics become trustworthy; heavy-rain recall should rise materially. |
| P3 | **LR → gradient-boosted trees** | Train with sklearn `HistGradientBoostingClassifier` (already in the nightly retrain env), export trees to `model/trees.json`, evaluate pure-Python tree traversal at inference (keeps the no-runtime-deps constraint). Keep LR as fallback if trees don't beat it on validation. | Captures feature interactions LR cannot (e.g. high humidity + fast pressure/rainfall delta + active WTS). Typical clear gain on tabular nowcasting. |
| P4 | **Proper evaluation + calibration** | Time-ordered train/val split (never random — leakage). Report per-target **PR-AUC** and **Brier score**, plus skill vs two baselines: (a) persistence (current warning continues), (b) climatology. Apply isotonic or Platt calibration on the validation fold; store calibration curve in `model/*.json`. | The >60% notify threshold becomes meaningful; prevents shipping a model that looks accurate but is worse than persistence. |
| P5 | **Learned blend weights** | Replace the fixed 0.2→0.5 ramp: per target, grid-search `w ∈ {0, 0.1, …, 1.0}` on the validation fold minimizing Brier score; store per-target `w` in `model/blend.json`. | Strong targets (rain_1h) get more AI weight; weak targets (tc) automatically stay rules-heavy. |
| P6 | **Cross-feed F3 → tabular** | Add F3-derived scalar features to the tabular models: max/mean nowcast rainfall over the HK window per lead frame, and frame-to-frame trend. Also add JTWC distance *rate of change* (approaching vs departing). | Gives the tree/LR models the radar-equivalent signal they currently never see; helps Amber/Red lead time. |
| P7 (optional) | **CNN input upgrade** | Use all 4 lead frames + 1 delta frame (frame diff) as channels; compare against a trivial advection/persistence baseline and only keep the CNN if it beats it. | Honest check that the CNN adds skill beyond "rain keeps moving the same way". |

### 9.3 Constraints honoured

- **GitHub Actions budget**: backfills (P1/P2) are one-off manual `workflow_dispatch` runs, not recurring — they cost minutes once, not monthly. Nightly retrain stays on the same schedule; tree training on a few years of hourly rows is still seconds-to-minutes on sklearn.
- **Repo storage**: archived tabular rows compress well (append to `snapshots.csv` or a parquet/`.csv.gz` sibling if the CSV grows past ~50 MB). F3 grids stay in `.npz`, downsampled to the 32×32 HK window before saving — never commit raw ZIPs.
- **Pure-Python inference**: preserved — tree traversal, sigmoid, and calibration mapping are all dependency-free at predict time.

### 9.4 Acceptance criteria for v4

- Every shipped model must beat the persistence baseline on time-ordered validation (PR-AUC and Brier), per target — otherwise that target stays rules-only.
- Report and commit a `model/metrics.json` (per-target PR-AUC, Brier, positive count, train range) so accuracy is tracked release-to-release instead of anecdotally.

### 9.5 v4 implementation log (shipped)

All of P1–P7 implemented:

| # | As built |
|---|---|
| P1 | `app/backfill_tabular.py`: daily rhrread+warnsum ZIPs from the data.gov.hk historical archive → snapshots.csv schema (default 20-min cadence), deduped by ts (live-polled rows win). Prints every extracted Amber/Red/Black/TC3/TC8 episode for warndb cross-check; optional `--warndb events.csv` (type,start,end HKT) auto-compares. TC/F3 features left blank on archived rows (train-time defaults apply). |
| P2 | `app/backfill.py --range START END --events`: event days from snapshots.csv rain-warning flags ∪ `data/event_days.txt` (fill from warndb), plus equal seeded-random quiet days. Merges/dedupes into `grid_dataset.npz`. **Label fix**: label is now real future rain — the lead-0 frame of the snapshot ~2h later — not the current snapshot's own 4th lead frame (which just taught the CNN HKO's extrapolation). |
| P3 | `train.py` trains LR *and* `HistGradientBoostingClassifier` per target, picks the val-PR-AUC winner, exports trees to `model/trees.json` (plain node lists), verifies pure-Python traversal reproduces `predict_proba` to 1e-4 (falls back to LR on any mismatch/export failure). Inference in `features.py` is dependency-free. |
| P4 | Timestamp-based labels (true 1/3/3/6-hour horizons via bisect — cadence-proof), time-ordered 80/20 split, per-target PR-AUC + Brier vs persistence and climatology, Platt calibration fitted on the val fold (dropped if it worsens Brier). A target ships only if it beats persistence on BOTH metrics; otherwise rules-only. Full report in `model/metrics.json`. |
| P5 | Per-target blend weight grid-searched (w ∈ 0…1, step .1) minimizing Brier of `(1-w)·rules + w·AI` on the val fold → `model/blend.json`; unshipped targets get w=0. `fetch.py` blends per target; legacy ramp only as fallback when blend.json is absent. Dashboard shows per-target weights. |
| P6 | New tabular features: `f3_max`/`f3_mean`/`f3_trend` (scalar summaries of the live F3 grid, fetched before row build) and `tc_dist_rate` (km/h approach speed from the row 1h back). Feature count guard skips stale models trained on the old feature set. |
| P7 | CNN input is 5 channels (4 lead frames + lead1−lead0 delta). Dataset stores `B` = max of the current ~2h lead frame; training reports CNN PR-AUC/Brier vs this advection baseline and `predict_frames` drops any threshold where the CNN failed to beat it. Backward-compatible with old 3-channel weights via `meta.in_ch`. |

New workflow `.github/workflows/backfill.yml` (workflow_dispatch only — one-off Actions minutes): modes `tabular` / `f3-events` / `f3-range` / `both` with start/end inputs, commits `data/`.

Suggested run order once merged: ① provide `data/warning_events.csv` (warndb export) → ② dispatch backfill `tabular` (start/end = years, e.g. 2022→2025) → ③ dispatch `f3-events` (dates, e.g. 2022-05-01→2025-10-31, 3 wet seasons) → ④ dispatch retrain → check `model/metrics.json` per-target `ships`/`blend_w`.

### 9.6 Correction to §9.1 (verified against the live archive API, 2026-08)

The §9.1 claim that archived `rhrread`/`warnsum` snapshots exist is **wrong**. Probes of
`api.data.gov.hk/v1/historical-archive/list-file-versions` (2021/2023/2025 windows): 0 versions for
both endpoints, in every window; not in the hk-hko json/xml/csv catalogs either. The archived RSS
warning feed (`WeatherWarningSummaryv2.xml`) gets only ~1 snapshot/day — daily even during Saola's
Signal 10 (Sep 2023) — so warning timelines cannot be reconstructed from it.

Verified working replacements (P1 as shipped):

| Need | Source (verified) |
|---|---|
| Hourly features | NOAA ISD-Lite station 450070-99999 (HK Intl Airport): temp, dewpoint→RH, pressure, 1h precip; hourly, multi-year. Station rain scaled ×15 to approximate live district-sum semantics. `ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/450070-99999-{year}.gz` |
| Warning labels | HKO warndb (issue/cancel times; Rainstorm since 1998, TC since 1946) exported to `data/warning_events.csv` (`type,start,end` HKT; types AMBER/RED/BLACK/TC1/TC3/TC8/TC9/TC10/WTS/WMSGNL). Backfill refuses to run without it (all-zero labels would poison training). **Done (2026-08-03)**: 9,276 events extracted from warndb's flat data files (`hko.gov.hk/dps/wxinfo/climat/warndb/{tc,rstorm,thunder}.dat` — browser-only endpoints; summer-time "S" flags converted to HKT) and committed. 771 Amber / 160 Red / 35 Black / 428 TC3 / 296 TC8+ / 1,248 SMS / 5,801 WTS, 1946 → 2026-08-02. Spot-checked against the Sep 2023 black rainstorm and Saola Signal 10. To refresh later, re-extract via a browser (fetch the three .dat files and rebuild the CSV). |
| F3 grids | data.gov.hk archive, 15-min snapshots, 2022-12 → present (verified 2022-12-01 = 2,966 snaps / 8GB; 2023-01-01 = 2,960 snaps / 8GB). Full 2022 wet season (May–Nov 2022) + 2023–2025 available (3 complete annual cycles), incl. Typhoon Noru (Sep 2022), Typhoon Haikui (Aug 2023), and the Sep 2023 black-rainstorm record event. |
| Bonus found | TC track XML (`weather.gov.hk/wxinfo/currwx/tc_list.xml`) is archived — candidate historical TC feature source for a future release. |
