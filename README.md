# SkyWager

Unofficial Hong Kong weather-warning nowcast. Polls HKO Open Data hourly, predicts
warning probability (rain, Amber/Red Rainstorm, Typhoon Signal 3+) using a hybrid of
hand-written rules and a nightly-retrained logistic-regression model, emails you on
official warning changes and AI lead alerts, and renders a static dashboard.

V2 adds JTWC tropical-cyclone best-track/forecast ingestion (distance, bearing, wind,
pressure, 24h forecast distance from HK) as model features and dashboard/email content.

Runs 100% free on GitHub Actions with a student (Pro) account, in a private repo.

## Architecture

```
GitHub Actions (hourly)                     GitHub Actions (03:00 UTC daily)
app/fetch.py ──► data/snapshots.csv ──► app/train.py (scikit-learn)
   │  rules + AI blend                          │
   │  └─► email alerts (Gmail SMTP)             ▼
   │  └─► data/latest.json, history.json   model/weights.json (pure-Python inference)
   ├─► app/jtwc.py ──► data/tc_state.json (JTWC/NOAA ATCF track data)
   └─► site/data/* ──► GitHub Pages dashboard
```

- `app/fetch.py` — stdlib-only poller: fetches `rhrread` + `warnsum`, appends a snapshot,
  runs rules + AI (sigmoid over JSON weights — no ML lib needed), sends emails on
  triggers, writes dashboard JSON.
- `app/jtwc.py` — V2: scans NOAA's ATCF mirror for active Western-Pacific cyclones,
  parses best-track + official (OFCL) forecast lines, computes distance/bearing from
  Hong Kong and the 24h forecast distance; persists `data/tc_state.json`.
- `app/rules.py` — interpretable nowcast rules (rainfall trend, humidity, active
  warnings, cyclone distance/wind/approach).
- `app/train.py` — nightly: trains one balanced logistic regression per target on
  accumulated data with lookahead labels; exports coefficients + feature scaling to
  `model/weights.json`; skips quietly until ≥200 rows / ≥5 positive samples per target.
- `site/index.html` — static dashboard (GitHub Pages).

## Notification triggers

1. Official HKO warning state changes (issue / extend / escalate / downgrade / cancel) — always emailed.
2. Lead alerts: hybrid (rules + AI) probability ≥60% for Amber/Red Rainstorm or TC Signal 3+
   within the horizon — emailed, then silent for 6h per signal type (cooldown).

## Setup

1. Create a **private** repo named `skywager`, push this folder to it.
2. Create a **Gmail app password** (Google Account → Security → 2-Step Verification →
   App passwords) — needed because GitHub Actions can't use your normal Gmail login.
3. Repo → Settings → Secrets and variables → Actions → add:
   - `SMTP_USER` — your Gmail address
   - `SMTP_APP_PASSWORD` — the 16-char app password
   - `NOTIFY_TO` — the address that receives alerts (can be the same Gmail)
4. Repo → Settings → Pages → Source: **GitHub Actions**. The `pages` workflow will
   publish `site/` after the first push; you'll get a URL like
   `https://<you>.github.io/skywager/`.
5. Actions will start polling on the schedule (hourly) and retraining nightly.
   To test immediately, open Actions → `poll` → Run workflow (workflow_dispatch).
   Set the `DISABLE_EMAIL` environment variable to 1 on a run if you want to test
   without sending mail.

## Budget (private repo, student Pro = 3000 Actions min/month)

- Poll: 24 runs/day × ~1.5 min ≈ 1080 min/month (includes TC track scan)
- Retrain: 30 × ~8 min ≈ 240 min/month
- Total ≈ 1300 min/month — plenty of headroom under the 3000-min limit.

## Honest limits

- HKO provides no historical warning archive via API — the model learns from day one,
  and rules carry the load until enough labeled snapshots accumulate (weeks).
- `rhrread` has no pressure field, so pressure-drop features from the original spec
  are not used; rainfall/humidity/temperature trends power the model.
- This is not an official forecast. Always check https://www.hko.gov.hk for official
  warnings before any safety decision.

## Future

- Radar-image CNN nowcasting (HKO radar imagery) — v3
- Public-repo flip when the student discount expires (unlimited Actions minutes)
