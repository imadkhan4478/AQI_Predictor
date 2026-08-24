"""The forecast, computed once and shared by every front end.

Both the FastAPI service (`api/main.py`) and the Streamlit dashboard
(`app/app.py`) call into this module rather than each assembling features and
applying the blend themselves.

That is not tidiness for its own sake. The last serious bug in this project was
exactly this kind of duplication: the training pipeline moved to a delta target
with lag features and the dashboard kept its own copy of the feature assembly,
so it silently served a stale model and would have rendered a predicted *change*
of -4 as "AQI 4 - Good" for a city sitting at 170. Two front ends multiply that
risk. One serving path does not.
"""

import numpy as np

from app.load_model import load_latest_model
from feature_pipeline.aqi import aqi_category
from feature_pipeline.hopsworks_store import read_features_df
from training_pipeline.data import FEATURE_COLUMNS, add_lag_features, add_time_features

HORIZONS_HOURS = (24, 48, 72)
MODEL_NAME_TEMPLATE = "aqi_forecast_{}h"

# AQI is defined on 0-500. A blended prediction is arithmetic and has no such
# bound, so clamp before anything renders it.
AQI_MIN, AQI_MAX = 0, 500

# Ordered so "the worst thing in the forecast" is well defined.
SEVERITY_RANK = {None: 0, "warning": 1, "serious": 2, "critical": 3}

HEALTH_ADVICE = {
    "warning": (
        "Sensitive groups (children, elderly, respiratory/heart conditions) should "
        "limit prolonged outdoor exertion."
    ),
    "serious": (
        "Everyone may begin to experience health effects. Limit prolonged outdoor "
        "exertion."
    ),
    "critical": (
        "Health alert: everyone may experience serious health effects. Avoid outdoor "
        "activity."
    ),
}


class ForecastUnavailable(RuntimeError):
    """Raised when a forecast genuinely cannot be produced.

    Distinct from a missing single horizon, which is reported in the payload
    rather than raised: one absent model should not blank the whole dashboard.
    """


def load_feature_row(city_name):
    """The newest feature row, built through the *same* transformations the
    training pipeline uses.

    The model's features include AQI lags and rolling statistics, which are
    functions of the preceding hours -- so the live row cannot be assembled from
    a single feature-store record. Reusing add_time_features and
    add_lag_features here, rather than recomputing them, is what keeps training
    and serving in step.
    """
    df = read_features_df()
    city_rows = df[df["city_name"] == city_name]
    if city_rows.empty:
        raise ForecastUnavailable(f"No feature rows stored for city {city_name!r}")

    city_rows = add_time_features(city_rows)
    city_rows = add_lag_features(city_rows)

    # Name the columns explicitly rather than letting pandas raise a bare
    # KeyError. If the feature group's schema and FEATURE_COLUMNS ever diverge
    # again -- the failure this whole module exists to prevent -- the error
    # should say which columns are missing.
    missing = [column for column in FEATURE_COLUMNS if column not in city_rows.columns]
    if missing:
        raise ForecastUnavailable(
            f"Feature store is missing columns the model was trained on: {missing}"
        )

    complete = city_rows.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        raise ForecastUnavailable(
            "No feature row yet has a complete set of lag features -- the store "
            "needs at least 72 contiguous hours of history."
        )
    return complete.sort_values("timestamp").iloc[-1]


def load_forecast_models(horizons=HORIZONS_HOURS):
    """One registered model per horizon, each with its own blend weight."""
    models = {}
    for horizon_hours in horizons:
        model, version, metrics, blend_weight = load_latest_model(
            MODEL_NAME_TEMPLATE.format(horizon_hours)
        )
        models[horizon_hours] = {
            "model": model,
            "version": version,
            "metrics": metrics,
            "blend_weight": blend_weight,
        }
    return models


def blend_forecast(model_info, features, current_aqi):
    """Turn a predicted *change* in AQI into a forecast level.

        prediction = current_aqi + w * predicted_delta

    w = 0 is exactly the persistence baseline, w = 1 the unshrunk model. The
    weight comes from walk-forward validation and is stored with the artifact.

    Returns None when no weight is registered. Deliberately not defaulting to
    1.0: that would deploy the unshrunk model, which loses to persistence at
    every horizon, and it would look exactly like a working forecast.
    """
    weight = model_info.get("blend_weight")
    if weight is None:
        return None
    predicted_delta = float(model_info["model"].predict(features)[0])
    blended = current_aqi + weight * predicted_delta
    return int(round(min(max(blended, AQI_MIN), AQI_MAX)))


def _describe(aqi_value):
    name, severity, color = aqi_category(aqi_value)
    return {"aqi": int(aqi_value), "category": name, "severity": severity, "color": color}


def build_forecast(city_name, models, feature_row):
    """Assemble the full payload both front ends render.

    Returns plain JSON-serialisable types only -- no numpy scalars, no
    DataFrames -- so FastAPI can return it unchanged and Streamlit can read the
    same structure whether it came from this function or over HTTP.
    """
    features = feature_row[FEATURE_COLUMNS].to_frame().T.astype(float)
    current_aqi = int(feature_row["aqi"])

    horizons, unavailable = [], []
    for horizon_hours, info in sorted(models.items()):
        predicted = blend_forecast(info, features, current_aqi)
        if predicted is None:
            unavailable.append(horizon_hours)
            continue
        horizons.append(
            {
                "horizon_hours": horizon_hours,
                **_describe(predicted),
                "model_version": info["version"],
                "blend_weight": info["blend_weight"],
            }
        )

    current = _describe(current_aqi)
    # Rows on interpolated hours carry no dominant pollutant: the lag features
    # reindex onto a continuous hourly grid, and a text column cannot be
    # interpolated across a gap the way a numeric one can.
    dominant = feature_row.get("dominant_pollutant")
    current["dominant_pollutant"] = dominant if isinstance(dominant, str) else None

    return {
        "city": city_name,
        "observed_at": feature_row["timestamp"].isoformat(),
        "current": current,
        "forecast": horizons,
        "unavailable_horizons": unavailable,
        "alert": _build_alert(current, horizons),
    }


def _build_alert(current, horizons):
    """The single worst point in the forecast, if it warrants an alert.

    Keyed off aqi_category() rather than its own thresholds, so the alert
    boundaries and the AQI scale cannot disagree -- they read the same table.
    """
    points = [{"horizon_hours": 0, **current}] + list(horizons)
    worst = max(points, key=lambda point: SEVERITY_RANK[point["severity"]])
    if not worst["severity"]:
        return None

    when = "now" if worst["horizon_hours"] == 0 else f"in {worst['horizon_hours']}h"
    return {
        "severity": worst["severity"],
        "horizon_hours": worst["horizon_hours"],
        "aqi": worst["aqi"],
        "category": worst["category"],
        "message": (
            f"{worst['category']} air quality expected {when} (AQI {worst['aqi']}). "
            f"{HEALTH_ADVICE[worst['severity']]}"
        ),
    }


def model_details(models):
    """Registry metadata per horizon, without the model objects."""
    return [
        {
            "horizon_hours": horizon_hours,
            "model_name": MODEL_NAME_TEMPLATE.format(horizon_hours),
            "version": info["version"],
            "blend_weight": info["blend_weight"],
            "metrics": {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in (info["metrics"] or {}).items()
            },
        }
        for horizon_hours, info in sorted(models.items())
    ]
