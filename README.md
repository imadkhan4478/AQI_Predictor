# AQI Predictor

Predicts the Air Quality Index (AQI) for a city 3 days ahead, using a 100% serverless MLOps stack.

## Architecture

```
OpenWeather API (weather + pollutants)
        │
        ▼
Feature pipeline (feature_pipeline/) ──► Hopsworks Feature Store
        │                                        │
   (hourly, GitHub Actions)                       │
                                                    ▼
                                    Training pipeline (training_pipeline/)
                                                    │
                                          (daily, GitHub Actions)
                                                    ▼
                                          Hopsworks Model Registry
                                                    │
                                                    ▼
                                     Streamlit app (app/) ──► 3-day AQI forecast
```

## Stack

- **Data source:** OpenWeather (weather) + OpenWeather Air Pollution (pollutant concentrations); AQI computed from concentrations using EPA breakpoints.
- **Feature store / model registry:** [Hopsworks](https://www.hopsworks.ai/) (free tier)
- **Modeling:** scikit-learn (Random Forest, Ridge) and TensorFlow/PyTorch, evaluated on RMSE/MAE/R²
- **Explainability:** SHAP
- **Automation:** GitHub Actions (feature pipeline hourly, training pipeline daily)
- **Dashboard:** Streamlit

## Repo layout

- `feature_pipeline/` — fetches raw data, computes features + targets, writes to the Hopsworks feature store
- `backfill/` — runs the feature pipeline over a range of historical dates to build training data
- `training_pipeline/` — reads features from Hopsworks, trains/evaluates models, registers the best one
- `app/` — Streamlit dashboard that loads the latest model + features and shows the 3-day forecast
- `notebooks/` — exploratory data analysis
- `.github/workflows/` — scheduled CI/CD jobs

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in API keys (OpenWeather, Hopsworks) — see that file for the full list.

## Status

Project scaffolding in progress. See commit history for what's implemented so far.
