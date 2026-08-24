# AQI Predictor

A serverless MLOps pipeline that forecasts the Air Quality Index for Lahore, Pakistan
24, 48 and 72 hours ahead. Data collection, retraining and serving all run without a
server: scheduled GitHub Actions jobs write to a hosted feature store, and a Streamlit
dashboard reads the latest registered model.

The forecast beats the naive baseline at every horizon — which took more work than it
sounds like, and is the reason the project is documented the way it is.

## Results

Evaluated **walk-forward**: retrain at successive monthly origins, score only the
following month, roll forward. 12 retrains, ~7,578 predictions per horizon, with
persistence scored on exactly the same rows.

| Horizon | Persistence baseline | Model alone | **Deployed blend** |
|---|---|---|---|
| 24h | 0.814 | 0.780 | **0.832** |
| 48h | 0.709 | 0.643 | **0.748** |
| 72h | 0.628 | 0.613 | **0.715** |

R² on held-out months, as measured by the daily training job against the live
49k-row feature group. Persistence — "AQI in H hours = AQI now" — is a genuinely
strong baseline for air quality, and no standalone model tried here beat it. The
deployed forecast blends the two.

## What the model actually predicts

The model predicts the **change** in AQI, and the forecast is anchored to the most
recent reading:

```
prediction = current_aqi + blend_weight × predicted_delta
```

`blend_weight = 0` is exactly persistence; `1` is the unshrunk model. The weight is
fitted on validation data and registered alongside the model artifact, so inference
cannot silently fall back to a default.

This is algebraically the same as averaging the model with persistence —
`(1-w)·aqi + w·(aqi + delta) = aqi + w·delta` — so deployment needs no separate
persistence branch at inference time.

Two reasons it is built this way:

- **Lahore's air has improved ~58% since 2020** (yearly mean AQI 387 → 162). Tree
  models cannot extrapolate below the levels they trained on, so a model predicting
  the *level* systematically over-predicts. Anchoring to today's reading tracks the
  trend for free.
- **Persistence is hard to beat and the blend is how you beat it.** The two make
  partly different errors, so a weighted average beats either alone.

## Architecture

```
OpenWeather (current weather + pollutant concentrations)
Open-Meteo archive (historical weather)        ← OpenWeather's weather history is paid
        │
        ▼
Feature pipeline (feature_pipeline/)  ──hourly, GitHub Actions──►  Hopsworks Feature Store
                                                                          │
                                                                          ▼
                                              Training pipeline (training_pipeline/)
                                                     daily, GitHub Actions
                                                              │
                                                              ▼
                                                  Hopsworks Model Registry
                                                              │
                                                              ▼
                                              Serving core (serving/forecast.py)
                                                    │                    │
                                                    ▼                    ▼
                                         FastAPI (api/)          Streamlit (app/)
                                                    └────────────────────┘
                                                     3-day AQI forecast
```

Both front ends call one serving module rather than each assembling features and
applying the model themselves. That is deliberate: the dashboard previously kept
its own copy of that logic, drifted out of step with the training pipeline, and
served a stale model while looking healthy.

## Stack

- **Data:** OpenWeather (weather + air pollution, including free pollution history back
  to 2020-11-27) and Open-Meteo's archive API for historical weather
- **AQI:** US EPA breakpoints (2024-revised PM2.5 table), computed from raw
  concentrations in `feature_pipeline/aqi.py` — see the module docstring for the
  averaging-window approximation this makes
- **Feature store + model registry:** [Hopsworks](https://www.hopsworks.ai/) (free tier)
- **Models:** scikit-learn (HistGradientBoosting — deployed — plus Ridge and Random
  Forest), TensorFlow/Keras MLP, statsmodels ARIMA, and an explicit persistence baseline
- **Explainability:** SHAP
- **Automation:** GitHub Actions — feature pipeline hourly, training pipeline daily,
  tests on every push, explainability weekly
- **Serving:** FastAPI (JSON API, OpenAPI docs at `/docs`) and Streamlit + Plotly
  (dashboard), over a shared serving core

## API

```
GET /health     cached state only — never triggers a registry connection
GET /current    latest observed AQI, EPA category, dominant pollutant
GET /forecast   current + 24/48/72h forecast, blend weights, hazard alert
GET /models     registry version and held-out metrics per horizon
```

```bash
uvicorn api.main:app --reload        # docs at http://127.0.0.1:8000/docs
```

`GET /forecast` returns the blend weight actually used alongside each number,
because a consumer cannot interpret the forecast without knowing how much of it
is the model and how much is persistence. A horizon whose model has no
registered weight is listed in `unavailable_horizons` rather than being served
with a guessed one.

Point the dashboard at the service with `AQI_API_URL=http://127.0.0.1:8000`;
without it, Streamlit calls the same serving core in-process.

## Data

49,115 hourly rows spanning **2020-11-27 → present**, no duplicate timestamps, no nulls.
33 features: raw weather and pollutant readings, computed AQI, cyclical hour encoding,
day-of-week, AQI lags (1/2/3/6/12/24h) and rolling mean/std (3/24/72h).

Day-of-month and month are deliberately **excluded** — with only 3 months of data they
let the model memorise "late July looks like this", and with 5.7 years they measurably
hurt.

## Repo layout

- `feature_pipeline/` — fetches raw data, computes the EPA AQI and features, writes one
  row per hour to the feature store
- `backfill/` — loads multi-year history in monthly chunks (runs on a GitHub runner)
- `training_pipeline/` — model candidates, walk-forward evaluation, blend-weight
  fitting, registration, SHAP
- `serving/forecast.py` — the forecast, shared by both front ends
- `api/` — FastAPI service
- `app/` — Streamlit dashboard
- `notebooks/01_eda.ipynb` — exploratory analysis
- `tests/` — 57 tests: the AQI calculation, the blend arithmetic, and the API contract
- `docs/EXPERIMENT_LOG.md` — **the running record of what was tried, what failed and
  what the numbers were**. Five of seven modelling hypotheses failed; they are kept in
  the log deliberately, because the two that worked are only trustworthy given the ones
  that didn't.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: the TensorFlow entry in the model comparison. Second step on purpose
# -- hopsworks declares protobuf<5 and tensorflow needs protobuf>=6.31, so pip
# cannot resolve both at once. Expect an "incompatible" warning about hopsworks;
# the project has always run in that state. Nothing deployed needs it.
pip install -r requirements-deep.txt
```

Copy `.env.example` to `.env` and fill in the six variables (OpenWeather key, Hopsworks
key + project, city name/lat/lon). The same six exist as GitHub Actions secrets so the
scheduled jobs authenticate without a secret ever entering the repository.

## Running it

```bash
python -m feature_pipeline.pipeline              # write one hourly feature row
python -m backfill.backfill                     # load historical data (prefer the runner)
python -m training_pipeline.train               # compare all candidates vs persistence
python -m training_pipeline.register_forecast_models   # train + register 24/48/72h
python -m training_pipeline.explain             # regenerate the SHAP plot
uvicorn api.main:app --reload                   # JSON API + /docs
streamlit run app/app.py                        # dashboard
pytest                                          # 57 tests, no credentials needed
```

Bulk operations are best run through **Actions → workflow_dispatch** rather than
locally: sustained uploads from the development machine drop mid-transfer, which cost
enough debugging time to be worth writing down.

## Known limitations

- The EPA AQI is computed from instantaneous hourly readings rather than the official
  averaging windows, making it a consistent hourly proxy rather than official AQI.
  Details and consequences in `feature_pipeline/aqi.py`.
- The blend weight is refit only at registration time, not continuously.
- Single city. Nothing in the pipeline is Lahore-specific except the configured
  coordinates, but nothing has been validated elsewhere either.
