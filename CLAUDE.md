# Working notes for AI assistants on this repo

Read this file first, then `Project Guidelines & Requirements.pdf`, then
`docs/EXPERIMENT_LOG.md`. In that order, every session. The log is long but it is
the only record of *why* the code looks the way it does, and several obvious
"improvements" have already been tried and measured as harmful.

## What this project is

10Pearls assignment: predict a city's AQI 3 days ahead on a 100% serverless
stack. City is Lahore, Pakistan. Graded on the four Final Submissions listed at
the end of the requirements PDF, not on model score alone.

## The requirements, condensed — check work against this list

| # | Requirement | Where it lives |
|---|---|---|
| 1 | Feature pipeline: fetch raw weather + pollutants from an external API | `feature_pipeline/fetch.py` (OpenWeather) |
| 2 | Compute features **and targets**; include time features (hour, day, month) and derived features like AQI change rate | `feature_pipeline/pipeline.py`, `training_pipeline/data.py` |
| 3 | Store features in a Feature Store | Hopsworks, `feature_pipeline/hopsworks_store.py` |
| 4 | Backfill historical (features, targets) over a range of past dates | `backfill/` — 49k rows, 2020-11-27 → now |
| 5 | Training pipeline: read from the store, train and evaluate the best model, register it | `training_pipeline/` |
| 6 | Experiment with scikit-learn (Random Forest, Ridge) **and** TensorFlow/PyTorch | `baseline_models.py`, `deep_model.py`, `statistical_model.py` |
| 7 | Evaluate with RMSE, MAE **and** R² | `evaluate.py` — all three, always |
| 8 | Store the model in a Model Registry | Hopsworks registry, `register.py` |
| 9 | CI/CD: feature script hourly, training script daily | `.github/workflows/` |
| 10 | Web app: load model + features, show predictions on a descriptive dashboard | `app/app.py` (Streamlit) |
| 11 | Use Streamlit/Gradio **and** Flask/FastAPI for the web app | `api/main.py` (FastAPI) + `app/app.py` (Streamlit), over `serving/forecast.py` |
| 12 | EDA to identify trends | `notebooks/01_eda.ipynb` |
| 13 | A variety of models, statistical → deep learning | ARIMA, Ridge, RF, HistGB, Keras MLP |
| 14 | SHAP or LIME feature importance | `training_pipeline/explain.py`, `.github/workflows/explainability.yml` |
| 15 | Alerts for hazardous AQI levels | `serving/forecast.py` `_build_alert()` + `aqi_category()` |
| 16 | **A detailed report documenting everything achieved** | **not written yet** — assemble from `docs/EXPERIMENT_LOG.md` |

## Conventions that are load-bearing — do not "simplify" these

- **Labels are matched by timestamp, never by row shift.** ~5% of hours are
  missing; a positional shift silently pairs rows with the wrong future value.
- **Splits are chronological AND purged.** Training rows whose label falls in
  the test window must be dropped. Without the purge, ~27% of the apparent skill
  at 72h was leakage.
- **The model predicts a delta, not a level.** The forecast is
  `current_aqi + w * predicted_delta`, with `w` fitted on validation and stored
  in `blend.json` beside the artifact. Never default `w` to 1.0.
- **Persistence is scored alongside every model, every time.** It beats every
  standalone model tried here. A result that isn't compared to it is not a result.
- **One serving path.** `serving/forecast.py` is the only place features are
  assembled for inference and the only place the blend is applied. Neither the
  API nor the dashboard may reimplement it — that duplication is exactly how the
  dashboard came to serve a stale model.
- **Evaluation is walk-forward.** A single frozen split picks the wrong model —
  demonstrated, see Phase 4.
- **`day` and `month` are stored but excluded from model features.** Requirement 2
  is satisfied by storing them; including them measurably hurt. Explain this in
  the report rather than silently reverting it.
- **Bulk operations run on a GitHub runner, not locally.** Sustained uploads from
  the dev machine drop mid-transfer.
- Explicit `float()` on every double column written to the feature group; pandas
  infers dtypes per batch and the schema is fixed.
- `requirements.txt` is pinned. Every number in the log was measured against
  those versions.

## Failure mode to watch for

Three times now the pipeline has reported success while doing nothing useful:
unaligned timestamps contributing zero training rows, a frozen materialization
job hiding 49k inserted rows, and a dashboard serving a stale model. Green CI is
not evidence. Verify by querying the thing that should have changed.
