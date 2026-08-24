"""FastAPI service exposing the registered AQI forecast over HTTP.

Run locally:

    uvicorn api.main:app --reload

Interactive docs at /docs, generated from the response models below.

Why an API at all, when Streamlit could load the model itself: it separates
"produce the forecast" from "draw the forecast". The dashboard becomes one
client among several, the JSON is consumable by anything, and the expensive part
-- a Hopsworks login and three model downloads -- happens once per process
instead of once per browser session.

Nothing here computes a forecast. All of that lives in serving/forecast.py,
shared with the dashboard, because two independent copies of the feature
assembly is precisely the bug this project already shipped once.
"""

import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.forecast import (
    HORIZONS_HOURS,
    ForecastUnavailable,
    build_forecast,
    load_feature_row,
    load_forecast_models,
    model_details,
)

load_dotenv()

# The feature store gains one row per hour, so re-reading it more often than
# that cannot produce new information -- and it is a multi-second Arrow Flight
# transfer. Ten minutes keeps the dashboard responsive without hammering it.
FEATURE_CACHE_SECONDS = 600

app = FastAPI(
    title="AQI Predictor API",
    version="1.0.0",
    description=(
        "3-day Air Quality Index forecast for a single city, served from models "
        "registered in the Hopsworks Model Registry. Each model predicts the "
        "*change* in AQI; the forecast returned is `current_aqi + blend_weight * "
        "predicted_delta`, which is algebraically a blend of the model with a "
        "persistence baseline."
    ),
)


class AqiReading(BaseModel):
    aqi: int = Field(..., description="US EPA AQI, 0-500", examples=[168])
    category: str = Field(..., examples=["Unhealthy"])
    severity: Optional[str] = Field(
        None, description="null, 'warning', 'serious' or 'critical'", examples=["serious"]
    )
    color: str = Field(..., description="Official EPA/AirNow colour", examples=["#ff0000"])
    dominant_pollutant: Optional[str] = Field(None, examples=["pm2_5"])


class ForecastPoint(BaseModel):
    horizon_hours: int = Field(..., examples=[24])
    aqi: int = Field(..., examples=[152])
    category: str = Field(..., examples=["Unhealthy"])
    severity: Optional[str] = None
    color: str = Field(..., examples=["#ff0000"])
    model_version: int = Field(..., description="Registry version actually used", examples=[7])
    blend_weight: float = Field(
        ...,
        description="0 would be the naive persistence baseline; 1 the unshrunk model",
        examples=[0.5],
    )


class Alert(BaseModel):
    severity: str = Field(..., examples=["serious"])
    horizon_hours: int = Field(..., description="0 means the alert is for right now")
    aqi: int
    category: str
    message: str


class ForecastResponse(BaseModel):
    city: str = Field(..., examples=["Lahore"])
    observed_at: str = Field(..., description="Timestamp of the latest feature row, UTC")
    current: AqiReading
    forecast: list[ForecastPoint]
    unavailable_horizons: list[int] = Field(
        default_factory=list,
        description=(
            "Horizons withheld because no blend weight is registered for them. A "
            "forecast is never produced with a guessed weight."
        ),
    )
    alert: Optional[Alert] = None


class CurrentResponse(BaseModel):
    city: str
    observed_at: str
    current: AqiReading


class ModelInfo(BaseModel):
    horizon_hours: int
    model_name: str
    version: int
    blend_weight: Optional[float]
    metrics: dict


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' once models and features are cached, else 'starting'")
    models_loaded: list[int] = Field(default_factory=list)
    feature_row_age_seconds: Optional[float] = None
    city: Optional[str] = None


class _Cache:
    """Process-wide cache. A Hopsworks login plus three model downloads takes
    tens of seconds, so it happens once, lazily, guarded by a lock.

    Lazily rather than at startup on purpose: loading in a startup hook means a
    transient Hopsworks outage stops the service from booting at all, instead of
    degrading a single request that can be retried.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.models = None
        self.feature_row = None
        self.feature_row_read_at = None

    def get_models(self):
        with self.lock:
            if self.models is None:
                self.models = load_forecast_models(HORIZONS_HOURS)
            return self.models

    def get_feature_row(self, city_name):
        with self.lock:
            fresh = (
                self.feature_row is not None
                and time.monotonic() - self.feature_row_read_at < FEATURE_CACHE_SECONDS
            )
            if not fresh:
                self.feature_row = load_feature_row(city_name)
                self.feature_row_read_at = time.monotonic()
            return self.feature_row

    def feature_row_age(self):
        if self.feature_row_read_at is None:
            return None
        return round(time.monotonic() - self.feature_row_read_at, 1)


CACHE = _Cache()


def _city_name():
    city = os.environ.get("CITY_NAME")
    if not city:
        raise HTTPException(status_code=500, detail="CITY_NAME is not configured")
    return city


def _payload():
    """Shared by /forecast and /current so both see identical numbers."""
    city = _city_name()
    try:
        return build_forecast(city, CACHE.get_models(), CACHE.get_feature_row(city))
    except ForecastUnavailable as error:
        # 503 rather than 500: the service is fine, the data it needs is not yet
        # there, and a retry later is the correct client behaviour.
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ConnectionError as error:
        raise HTTPException(
            status_code=503, detail=f"Could not reach the model registry: {error}"
        ) from error


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Cheap liveness check -- reports cached state without touching Hopsworks,
    so an orchestrator polling it cannot itself cause load."""
    loaded = sorted(CACHE.models) if CACHE.models else []
    return {
        "status": "ok" if loaded and CACHE.feature_row is not None else "starting",
        "models_loaded": loaded,
        "feature_row_age_seconds": CACHE.feature_row_age(),
        "city": os.environ.get("CITY_NAME"),
    }


@app.get("/current", response_model=CurrentResponse, tags=["forecast"])
def current():
    """The latest observed AQI, with its EPA category and dominant pollutant."""
    payload = _payload()
    return {
        "city": payload["city"],
        "observed_at": payload["observed_at"],
        "current": payload["current"],
    }


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast():
    """Current AQI plus the 24h, 48h and 72h forecast, and a hazard alert if the
    worst point in that window warrants one."""
    return _payload()


@app.get("/models", response_model=list[ModelInfo], tags=["meta"])
def models():
    """Which registry version is serving each horizon, with its held-out metrics
    and blend weight. Useful for confirming the daily retrain is landing."""
    try:
        return model_details(CACHE.get_models())
    except ConnectionError as error:
        raise HTTPException(
            status_code=503, detail=f"Could not reach the model registry: {error}"
        ) from error
