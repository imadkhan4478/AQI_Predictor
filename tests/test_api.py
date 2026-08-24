"""Tests for the FastAPI service.

The registry and feature store are replaced by pre-populated caches, so these
tests check what the API contributes -- routing, status codes and response
shape -- rather than re-testing the forecast arithmetic covered in
test_forecast.py.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import CACHE, app
from serving import forecast as fc
from serving.forecast import ForecastUnavailable


class StubModel:
    def __init__(self, delta):
        self.delta = delta

    def predict(self, features):
        return np.array([self.delta] * len(features))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CITY_NAME", "Lahore")

    values = {column: 1.0 for column in fc.FEATURE_COLUMNS}
    values.update(
        {
            "aqi": 170,
            "timestamp": pd.Timestamp("2026-08-24 06:00:00+00:00"),
            "dominant_pollutant": "pm2_5",
        }
    )

    # Pre-populate the cache so no request attempts a Hopsworks connection.
    monkeypatch.setattr(
        CACHE,
        "models",
        {
            horizon: {
                "model": StubModel(delta),
                "version": 5,
                "metrics": {"R2": 0.83},
                "blend_weight": 0.5,
            }
            for horizon, delta in ((24, -20), (48, 10), (72, 30))
        },
    )
    monkeypatch.setattr(CACHE, "feature_row", pd.Series(values))
    monkeypatch.setattr(CACHE, "feature_row_read_at", 0.0)
    monkeypatch.setattr(CACHE, "get_feature_row", lambda city: CACHE.feature_row)

    return TestClient(app)


def test_forecast_returns_all_three_horizons(client):
    body = client.get("/forecast").json()

    assert body["city"] == "Lahore"
    assert [point["horizon_hours"] for point in body["forecast"]] == [24, 48, 72]
    assert body["forecast"][0]["aqi"] == 160  # 170 + 0.5 * -20
    assert body["current"]["aqi"] == 170
    assert body["unavailable_horizons"] == []


def test_forecast_includes_the_blend_weight_actually_used(client):
    """The weight is part of the contract, not an implementation detail: a
    consumer cannot interpret the number without knowing how much of it is the
    model and how much is persistence."""
    for point in client.get("/forecast").json()["forecast"]:
        assert point["blend_weight"] == 0.5
        assert point["model_version"] == 5


def test_current_matches_the_forecast_payload(client):
    """Both endpoints read one cached feature row, so they cannot disagree."""
    assert client.get("/current").json()["current"] == client.get("/forecast").json()["current"]


def test_alert_is_present_when_the_forecast_is_hazardous(client):
    alert = client.get("/forecast").json()["alert"]
    assert alert["severity"] == "serious"
    assert alert["category"] == "Unhealthy"


def test_models_endpoint_lists_every_horizon(client):
    body = client.get("/models").json()
    assert [entry["horizon_hours"] for entry in body] == [24, 48, 72]
    assert body[0]["model_name"] == "aqi_forecast_24h"
    assert "model" not in body[0]  # the artifact itself must never be serialised


def test_health_is_ok_once_warm(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == [24, 48, 72]
    assert body["city"] == "Lahore"


def test_health_does_not_touch_the_registry(monkeypatch):
    """An orchestrator polling /health must not be able to trigger a Hopsworks
    login -- otherwise the health check becomes the load."""
    monkeypatch.setattr(CACHE, "models", None)
    monkeypatch.setattr(CACHE, "feature_row", None)
    monkeypatch.setattr(CACHE, "feature_row_read_at", None)

    def explode(*args, **kwargs):
        raise AssertionError("/health must not load models")

    monkeypatch.setattr("api.main.load_forecast_models", explode)

    body = TestClient(app).get("/health").json()
    assert body["status"] == "starting"
    assert body["models_loaded"] == []


def test_missing_data_is_503_not_500(client, monkeypatch):
    """The service is healthy; the data it needs is not there yet. A client
    should retry, which 503 says and 500 does not."""
    monkeypatch.setattr(
        CACHE, "get_feature_row", lambda city: (_ for _ in ()).throw(ForecastUnavailable("no rows"))
    )
    response = client.get("/forecast")
    assert response.status_code == 503
    assert "no rows" in response.json()["detail"]


def test_unreachable_registry_is_503(client, monkeypatch):
    monkeypatch.setattr(
        CACHE, "get_models", lambda: (_ for _ in ()).throw(ConnectionError("hopsworks down"))
    )
    assert client.get("/models").status_code == 503
