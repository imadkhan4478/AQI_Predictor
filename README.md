# AQI Predictor

Forecasts the US EPA Air Quality Index for Lahore, Pakistan at **24, 48 and 72 hours
ahead**. Data collection, retraining and serving all run without an administered server:
scheduled GitHub Actions jobs write to a hosted feature store, and a shared serving module
feeds both a JSON API and a Streamlit dashboard.

[![Tests](https://github.com/imadkhan4478/AQI_Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/imadkhan4478/AQI_Predictor/actions/workflows/tests.yml)
[![Feature Pipeline](https://github.com/imadkhan4478/AQI_Predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/imadkhan4478/AQI_Predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/imadkhan4478/AQI_Predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/imadkhan4478/AQI_Predictor/actions/workflows/training_pipeline.yml)

| | |
|---|---|
| **Technical report** | [`docs/AQI_Predictor_Technical_Report.pdf`](docs/AQI_Predictor_Technical_Report.pdf) — methodology, results, what failed, limitations |
| **Experiment log** | [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — dated chronological record, every measurement |
| **Exploratory analysis** | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — executed, renders on GitHub |
| **Conventions** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — read before changing anything |

## Headline result

| Horizon | Persistence baseline | **Deployed blend** |
|---|---|---|
| 24h | 0.814 | **0.832** |
| 48h | 0.709 | **0.748** |
| 72h | 0.628 | **0.715** |

R² under walk-forward evaluation — retrain at successive monthly origins, score only the
following month. The forecast beats the naive baseline on R² and RMSE at all three
horizons; on MAE it is level at 24–48h and ahead only at 72h, because the gain is in large
errors rather than typical ones. Full numbers, all six candidate models and the reasoning
are in the report, §6.

## Quickstart

```bash
git clone https://github.com/imadkhan4478/AQI_Predictor.git
cd AQI_Predictor

python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q                          # 72 tests, no credentials needed
```

The test suite substitutes the feature store and the model registry, so it runs on a fresh
clone with no configuration. Everything else needs credentials — see below.

```bash
cp .env.example .env               # then fill in the six values

uvicorn api.main:app --reload      # API on :8000, docs at /docs
streamlit run app/app.py           # dashboard on :8501
```

To point the dashboard at the running API instead of loading models into its own process:

```bash
AQI_API_URL=http://127.0.0.1:8000 streamlit run app/app.py
```

### Optional: the TensorFlow model

The neural-net entry in the model comparison needs TensorFlow, which is deliberately **not**
in `requirements.txt`:

```bash
pip install -r requirements-deep.txt
```

`hopsworks` declares `protobuf<5` while `tensorflow` requires `protobuf>=6.31`, so pip
cannot resolve both in one pass. Installing second works and prints an "incompatible"
warning about hopsworks, which is expected. Nothing deployed needs it — the pipelines, API,
dashboard and tests all run on `requirements.txt` alone.

## Configuration

Six variables, documented in `.env.example`. The same six are held as GitHub Actions
secrets; no secret has ever been committed, verified by a full history scan.

| Variable | Purpose |
|---|---|
| `OPENWEATHER_API_KEY` | Current weather, current pollution, and the free pollution archive |
| `HOPSWORKS_API_KEY` | Feature store and model registry |
| `HOPSWORKS_PROJECT_NAME` | Hopsworks project to connect to |
| `CITY_NAME` | Label stored with every row, and the dashboard title |
| `CITY_LAT` / `CITY_LON` | Coordinates passed to both APIs |

`AQI_API_URL` is optional and read only by the dashboard.

## How it runs

Six workflows in `.github/workflows/`. Three are scheduled; three are manual.

| Workflow | Trigger | What it does |
|---|---|---|
| **Feature Pipeline** | hourly | Fetch, compute the EPA AQI, write one row to the feature store |
| **Training Pipeline** | daily, 02:00 UTC | Walk-forward evaluation and blend-weight fitting per horizon, then register three models |
| **Tests** | every push and PR | pyflakes plus the full test suite. No secrets required |
| **Analysis** | manual | Runs the six-model comparison and/or executes the EDA notebook, then commits `reports/model_comparison.json` and the notebook back |
| **Explainability** | weekly + manual | Regenerates the SHAP plot from the deployed model and commits it back |
| **Backfill** | manual | Loads multi-year history in monthly chunks |

**Anything that reads the whole feature group should run on a runner, not locally.**
Sustained transfers from a development machine drop mid-stream; this was verified as a
network limitation rather than a code bug. That is what the Analysis and Backfill workflows
are for.

## API

```
GET /health      cached state only — never contacts the registry
GET /current     latest observed AQI, EPA category, dominant pollutant
GET /forecast    current + 24/48/72h forecast, blend weights, hazard alert
GET /models      registry version, held-out metrics and blend weight per horizon
```

Interactive docs at `/docs`, generated from typed response models.

```jsonc
// GET /forecast
{
  "city": "Lahore",
  "observed_at": "2026-08-15T02:00:00+00:00",
  "current": {
    "aqi": 168, "category": "Unhealthy", "severity": "serious",
    "color": "#ff0000", "dominant_pollutant": "pm2_5"
  },
  "forecast": [
    { "horizon_hours": 24, "aqi": 159, "category": "Unhealthy",
      "severity": "serious", "color": "#ff0000",
      "model_version": 3, "blend_weight": 0.35 }
    // 48h and 72h follow
  ],
  "unavailable_horizons": [],
  "alert": {
    "severity": "serious", "horizon_hours": 0, "aqi": 168,
    "category": "Unhealthy",
    "message": "Unhealthy air quality expected now (AQI 168). Everyone may begin to experience health effects. Limit prolonged outdoor exertion."
  }
}
```

Two contract details worth knowing:

- **`blend_weight` is part of the response**, not internal bookkeeping. The forecast is
  `current_aqi + blend_weight × predicted_delta`, so a consumer cannot interpret the number
  without knowing how much of it is model and how much is persistence. A weight of 0 would
  be the naive baseline.
- **A horizon with no registered weight is withheld**, and named in `unavailable_horizons`,
  rather than served with a guessed one.

`/health` reports only cached state and never contacts Hopsworks, so polling it cannot
itself generate load. Missing data returns **503**, not 500 — the service is fine, the data
is not there yet, and the client should retry.

## Architecture

```
OpenWeather (weather + pollutants + pollution archive)
Open-Meteo archive (historical weather)      ← OpenWeather's weather history is paid
        │
        ▼
feature_pipeline/   ──hourly, GitHub Actions──►   Hopsworks Feature Store
                                                          │
                        training_pipeline/  ──daily──►  Hopsworks Model Registry
                                                          │
                                              serving/forecast.py
                                                │                │
                                          api/main.py      app/app.py
```

Both front ends call one serving module rather than each assembling features and applying
the model themselves. This is the repository's most important structural rule: the dashboard
once kept its own copy of that logic, drifted out of step with the training pipeline, and
served a stale model while appearing healthy.

## Repository layout

```
feature_pipeline/       fetch → compute EPA AQI → write one row per hour
  aqi.py                EPA breakpoints, unit conversion, truncation, clamping
  pipeline.py           the hourly job
  hopsworks_store.py    feature-group definition and retrying reads
backfill/               multi-year history in monthly chunks
training_pipeline/
  data.py               features, targets, the purged chronological split
  baseline_models.py    Ridge, Random Forest, HistGradientBoosting, persistence
  deep_model.py         Keras MLP
  statistical_model.py  rolling-origin ARIMA
  train.py              the six-model comparison → reports/model_comparison.json
  register.py           walk-forward evaluation, blend weight, registration
  explain.py            SHAP
serving/forecast.py     the forecast — the only place features are assembled for inference
api/main.py             FastAPI service
app/                    Streamlit dashboard and registry loading
tests/                  72 tests, no credentials required
notebooks/              executed EDA over the full history
scripts/                offline smoke test of the slow paths
docs/                   report, experiment log, conventions
```

## Data

| | |
|---|---|
| Rows | **49,153** hourly observations |
| Span | 2020-11-27 00:00 → 2026-08-15 02:00 UTC |
| Missing hours | 938 — **1.9%** |
| Duplicates / nulls | 0 / 0 |
| Stored columns → model features | 23 → 33 |
| Mean / median AQI | 243 / 189 |
| Dominant pollutant | PM2.5 in **89.2%** of hours, ozone 10.4% |

Features: raw weather and pollutant readings, the computed AQI and its change rate,
cyclically encoded hour, day-of-week, AQI lags at 1/2/3/6/12/24h, and rolling mean and
standard deviation over 3/24/72h. `day` and `month` are computed and stored but excluded
from training, because including them measurably hurts — report §4.2.

The AQI is computed from raw concentrations using EPA breakpoints, applied to hourly
readings rather than their official averaging windows. That approximation is documented in
the `feature_pipeline/aqi.py` module docstring and in report §2.2. It is a consistent hourly
proxy suitable as a forecasting target, not a figure to publish as official EPA AQI.

## Troubleshooting

**`FlightUnavailableError` / `Could not read data using Hopsworks Query Service`** — the
Query Service is unavailable or the transfer dropped. On the Python client, Arrow Flight is
the only offline read path, so there is no client-side workaround: retry later, and run
whole-feature-group reads through the Analysis workflow rather than locally.

**`ResolutionImpossible` mentioning protobuf** — TensorFlow and hopsworks are in the same
pip pass. Install `requirements.txt` first, then `requirements-deep.txt`.

**nbconvert times out executing the notebook** — its per-cell timeout defaults to 30
seconds, which the feature-store read alone exceeds. Pass
`--ExecutePreprocessor.timeout=1800`.

**The dashboard reports a withheld horizon** — no blend weight is registered for that model.
Re-run the Training Pipeline. This is deliberate: a guessed weight would produce a
plausible-looking but wrong forecast.

**A registry write fails with `errorCode 110043`** — a server-side fault inside Hopsworks'
filesystem metadata layer, mid-upload. `register.py` retries the write; if all attempts
fail, the loader falls back to the newest version that still loads.

## Contributing

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first. It lists the conventions that are
load-bearing rather than stylistic — timestamp-matched labels, the purged split, the
never-defaulted blend weight, one serving path — each arrived at by measurement, and each of
which reintroduces a real bug if "simplified".

One rule from that document bears repeating here: **green CI is not evidence.** Four times
in this project's history, something reported success while doing nothing useful. Verify by
querying the thing that should have changed.
