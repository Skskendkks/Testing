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
