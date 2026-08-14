# CNN Rain Nowcast Benchmark

<https://skskendkks.github.io/testing/site/index.html>

This repository is a **reproducible CNN skill-evaluation project**, not a weather-warning or public forecasting product. It asks one bounded question:

> **Do CNN features from HKO F3 gridded rainfall nowcasts add predictive skill beyond HKO F3's own two-hour advection nowcast?**

The dashboard reports a clear **completed**, **blocked**, or **no added skill** outcome. It never presents a CNN probability as an operational weather forecast and it does not send model-driven lead alerts. For weather decisions, use official [Hong Kong Observatory warnings](https://www.hko.gov.hk/).

## What is evaluated

| Item | Definition |
|---|---|
| **Input** | Four HKO F3 half-hourly gridded rainfall-nowcast frames over Hong Kong, downsampled to 32 × 32, plus one inter-frame change channel. |
| **Target** | Maximum rainfall in the first lead frame of a later F3 snapshot approximately 90–150 minutes after the input. |
| **Thresholds** | At least 15 mm, 25 mm, or 35 mm in a 30-minute grid cell around two hours ahead. |
| **Evaluation split** | Timestamp-ordered final 20% holdout. Training never sees the holdout period. |
| **Baseline** | Maximum rain in the input snapshot's own F3 two-hour advection lead. |
| **Success criterion** | The CNN must improve **both** PR-AUC and Brier score versus the baseline for a target. |

The target is a later **F3 product**, rather than independent rain-gauge ground truth. Therefore the benchmark measures incremental skill relative to the F3 proxy baseline only; it does not establish general weather-forecasting skill.

## Architecture

```text
HKO F3 archive ── app/backfill.py ── data/grid_dataset.npz
                                             │
                       time-ordered holdout ─┤
                                             ▼
                            app/cnn_benchmark.py
                              │         │
                              │         └── F3 advection baseline comparison
                              ▼
          model/cnn_evaluation.json + site/data/cnn_evaluation.json
                              │
                              ▼
                    GitHub Pages experiment report
```

| Component | Responsibility |
|---|---|
| `app/backfill.py` | Builds the version-4 CNN dataset from historical F3 archive snapshots. It pairs inputs to a later F3 proxy label and saves the advection baseline `B`. |
| `app/cnn.py` | Pure-NumPy CNN implementation and PR-AUC/Brier metric calculation. A CNN target is eligible only if it beats the baseline. |
| `app/cnn_benchmark.py` | Validates the dataset contract, runs the chronological benchmark, and publishes a machine-readable report. An incompatible dataset creates an explicit `blocked` report rather than a model. |
| `site/index.html` | Static experiment-report interface, showing the task, dataset, baseline comparison, conclusion, and data-pipeline health. |
| `.github/workflows/retrain.yml` | Manual or weekly `cnn-evaluation` run that updates only the benchmark report and an eligible CNN artifact. |
| `app/fetch.py` | Collects source-health information and official-warning changes only. It no longer publishes model probabilities or lead alerts. |

## Run an experiment

First collect a compatible data set. The existing legacy 3-channel data set is intentionally rejected because it lacks the current five-channel schema and the baseline array needed for a fair comparison.

```bash
# Example: collect event-oriented and quiet days in the desired date range.
python app/backfill.py --range 2025-05-01 2025-10-31 --events

# Validate the contract, train on the chronological training period,
# and evaluate on the untouched final 20%.
python app/cnn_benchmark.py
```

The benchmark writes these artifacts:

| File | Meaning |
|---|---|
| `data/grid_dataset.manifest.json` | Dataset schema, SHA-256, source days, archive failures, duplicate removal, input gaps, input/label time span, actual lead-time distribution and class counts. |
| `model/cnn_evaluation.json` | Full machine-readable result: task definition, dataset summary, manifest evidence, status, blocking reasons or per-target metrics. |
| `site/data/cnn_evaluation.json` | Copy rendered by the static benchmark dashboard. |
| `model/cnn_weights.json` | CNN weights from a compatible benchmark run. They are not used to issue live weather forecasts. |

The builder rejects old datasets and records archive failures rather than silently treating incomplete history as valid. A small date range is useful to verify ingestion, but it will remain `blocked` until it satisfies the minimum sample and chronological-holdout-positive requirements.

A benchmark can produce three useful outcomes.

| Status | Interpretation |
|---|---|
| `blocked` | The dataset cannot support the experiment yet, such as missing baseline values, old channels, insufficient samples, or too few holdout positives. Rebuild or extend the dataset. |
| `completed` with **added skill** | At least one threshold improves both PR-AUC and Brier score over the F3 baseline on the chronological holdout. This is evidence for that precise task only. |
| `completed` with **no added skill** | The CNN did not improve on the baseline. This is a valid result and the correct conclusion is that the CNN has not demonstrated value for this task. |

## Automation

The `cnn-evaluation` workflow runs weekly and can be triggered manually from GitHub Actions. It runs the regression suite, executes the benchmark, and commits only `cnn_evaluation.json`, `cnn_weights.json` when applicable, and the dashboard report. The `poll` workflow retains source-health and official-warning tracking but does not run model inference or send model alerts.

## Data notes and limitations

The F3 archive is accessed through the data.gov.hk historical-archive endpoint for HKO's gridded rainfall nowcast. The data represent a nowcast product, not radar imagery or ground truth. Archived coverage, missing snapshots, spatial downsampling, and label correlation with the F3 baseline can all affect results. The dashboard and report make these constraints visible so that the project remains an experiment rather than a forecasting claim.

## Development checks

```bash
python -m unittest discover -s tests -v
python app/cnn_benchmark.py --quiet
```

The test suite covers the CNN dataset contract, benchmark conclusions, model-artifact safety, rainfall schema, and dashboard health-state data.
