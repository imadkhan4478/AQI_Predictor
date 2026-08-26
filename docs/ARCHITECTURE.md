# Architecture and conventions

Orientation for anyone reading or extending this repo. For the reasoning behind
the decisions, and the experiments that failed on the way to them, see
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Requirements coverage

| # | Requirement | Where it lives |
|---|---|---|
| 1 | Fetch raw weather + pollutants from an external API | `feature_pipeline/fetch.py` (OpenWeather) |
| 2 | Compute features and targets; time features (hour, day, month) and derived features such as AQI change rate | `feature_pipeline/pipeline.py`, `training_pipeline/data.py` |
| 3 | Store features in a Feature Store | Hopsworks — `feature_pipeline/hopsworks_store.py` |
| 4 | Backfill historical (features, targets) | `backfill/` — 49k rows, 2020-11-27 onwards |
| 5 | Training pipeline: read, train, evaluate, register | `training_pipeline/` |
| 6 | scikit-learn (Random Forest, Ridge) and TensorFlow/PyTorch | `baseline_models.py`, `deep_model.py`, `statistical_model.py` |
| 7 | RMSE, MAE and R² | `evaluate.py` — all three, always |
| 8 | Model Registry | Hopsworks — `register.py` |
| 9 | CI/CD: features hourly, training daily | `.github/workflows/` |
| 10 | Web app: load model + features, descriptive dashboard | `app/app.py` (Streamlit) |
| 11 | Streamlit/Gradio **and** Flask/FastAPI | `api/main.py` + `app/app.py`, over `serving/forecast.py` |
| 12 | EDA to identify trends | `notebooks/01_eda.ipynb` — full history, trend and seasonality separated |
| 13 | Variety of models, statistical → deep learning | ARIMA, Ridge, Random Forest, HistGradientBoosting, Keras MLP — all compared in `training_pipeline/train.py`, results in `reports/model_comparison.json` |
| 14 | SHAP or LIME | `training_pipeline/explain.py`, `.github/workflows/explainability.yml` |
| 15 | Alerts for hazardous AQI | `serving/forecast.py` `_build_alert()` + `aqi_category()` |
| 16 | Detailed report | *outstanding* — to be assembled from the experiment log |

## Data flow

```
OpenWeather (current weather + pollutants)      Open-Meteo archive (historical weather)
                    │                                        │
                    └────────────────┬───────────────────────┘
                                     ▼
              feature_pipeline/  ──hourly──►  Hopsworks Feature Store
                                                      │
                              training_pipeline/  ──daily──►  Model Registry
                                                      │
                                          serving/forecast.py
                                            │              │
                                       api/main.py     app/app.py
```

## Conventions that are load-bearing

These are not style preferences. Each one was arrived at by measurement, and
"simplifying" any of them reintroduces a bug that has already happened here.

**Labels are matched by timestamp, never by row shift.** About 2% of hours are
missing (938 of 50,091); a positional shift pairs rows with the wrong future value
and reports no error.

**Splits are chronological *and* purged.** Training rows whose label falls inside
the test window must be dropped. Without the purge, ~27% of the apparent skill at
72h was leakage.

**The model predicts a delta, not a level.** The forecast is
`current_aqi + w · predicted_delta`, with `w` fitted on validation and stored in
`blend.json` beside the artifact. `w` is never defaulted to 1.0 — the unshrunk
model loses to persistence at every horizon.

**Persistence is scored alongside every model, every time.** It beats every
standalone model tried here, so a result not compared against it is not a result.

**Evaluation is walk-forward.** A single frozen split selects a different, worse
model — a stale model cannot track a falling trend, which flatters linear models.

**One serving path.** `serving/forecast.py` is the only place features are
assembled for inference and the only place the blend is applied. Neither front
end may reimplement it.

**`day` and `month` are stored but excluded from the feature set.** Requirement 2
is met by storing them; including them in training measurably hurts.

**Explicit `float()` on every double column written to the feature group.** pandas
infers dtypes per batch while the feature group schema is fixed.

**Bulk operations run on a GitHub runner, not locally.** Sustained uploads from
the development machine drop mid-transfer.

**`requirements.txt` is pinned**, and TensorFlow is deliberately not in it — see
`requirements-deep.txt` for why the two cannot be resolved together.

## The failure mode this project keeps hitting

Four times now, something has reported success while doing nothing useful:
unaligned timestamps contributing zero training rows, a frozen materialisation
job hiding 49k inserted rows, a dashboard serving a stale model, and a daily job
that failed for a day because a function signature changed under it.

Green CI is not evidence. Verify by querying the thing that should have changed.
