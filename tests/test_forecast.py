"""Tests for the serving layer -- the blend arithmetic and the payload both the
API and the dashboard render.

No network and no Hopsworks: the feature store and the registered models are
substituted, because what needs testing here is the arithmetic and the shape of
the response, not whether Hopsworks is reachable.
"""

import numpy as np
import pandas as pd
import pytest

from serving import forecast as fc


class StubModel:
    """Stands in for a registered model: returns a fixed predicted delta."""

    def __init__(self, delta):
        self.delta = delta

    def predict(self, features):
        return np.array([self.delta] * len(features))


def model_info(delta, weight, version=3, metrics=None):
    return {
        "model": StubModel(delta),
        "version": version,
        "metrics": metrics if metrics is not None else {"R2": 0.83, "MAE": 21.4},
        "blend_weight": weight,
    }


def feature_row(aqi=170, dominant="pm2_5"):
    values = {column: 1.0 for column in fc.FEATURE_COLUMNS}
    values.update(
        {
            "aqi": aqi,
            "timestamp": pd.Timestamp("2026-08-24 06:00:00+00:00"),
            "dominant_pollutant": dominant,
        }
    )
    return pd.Series(values)


# --- blend arithmetic -------------------------------------------------------


def test_zero_weight_is_exactly_persistence():
    features = feature_row()[fc.FEATURE_COLUMNS].to_frame().T.astype(float)
    assert fc.blend_forecast(model_info(-40, 0.0), features, 170) == 170


def test_full_weight_applies_the_whole_delta():
    features = feature_row()[fc.FEATURE_COLUMNS].to_frame().T.astype(float)
    assert fc.blend_forecast(model_info(-40, 1.0), features, 170) == 130


def test_half_weight_shrinks_the_delta():
    features = feature_row()[fc.FEATURE_COLUMNS].to_frame().T.astype(float)
    assert fc.blend_forecast(model_info(-40, 0.5), features, 170) == 150


def test_missing_weight_withholds_the_forecast():
    """A weight of None must never be silently treated as 1.0: the unshrunk
    model loses to persistence at every horizon."""
    features = feature_row()[fc.FEATURE_COLUMNS].to_frame().T.astype(float)
    assert fc.blend_forecast(model_info(-40, None), features, 170) is None


@pytest.mark.parametrize(
    "current,delta,expected",
    [(20, -500, fc.AQI_MIN), (480, 500, fc.AQI_MAX)],
)
def test_forecast_is_clamped_to_the_aqi_scale(current, delta, expected):
    features = feature_row()[fc.FEATURE_COLUMNS].to_frame().T.astype(float)
    assert fc.blend_forecast(model_info(delta, 1.0), features, current) == expected


# --- payload ----------------------------------------------------------------


def test_payload_contains_every_horizon_with_its_version_and_weight():
    models = {24: model_info(-20, 0.5, version=7), 48: model_info(10, 0.4, version=7)}
    payload = fc.build_forecast("Lahore", models, feature_row(aqi=170))

    assert [point["horizon_hours"] for point in payload["forecast"]] == [24, 48]
    assert payload["forecast"][0]["aqi"] == 160
    assert payload["forecast"][0]["model_version"] == 7
    assert payload["forecast"][0]["blend_weight"] == 0.5
    assert payload["unavailable_horizons"] == []
    assert payload["current"]["aqi"] == 170
    assert payload["current"]["dominant_pollutant"] == "pm2_5"
    assert payload["observed_at"].startswith("2026-08-24T06:00:00")


def test_horizon_without_a_weight_is_reported_not_guessed():
    models = {24: model_info(-20, 0.5), 72: model_info(-5, None)}
    payload = fc.build_forecast("Lahore", models, feature_row())

    assert [point["horizon_hours"] for point in payload["forecast"]] == [24]
    assert payload["unavailable_horizons"] == [72]


def test_interpolated_row_reports_no_dominant_pollutant():
    """Lag features reindex onto a continuous hourly grid; a text column cannot
    be interpolated across a gap, so it arrives as NaN rather than a string."""
    payload = fc.build_forecast("Lahore", {24: model_info(0, 0.5)}, feature_row(dominant=np.nan))
    assert payload["current"]["dominant_pollutant"] is None


# --- alerts -----------------------------------------------------------------


def test_alert_fires_for_the_worst_point_in_the_window():
    models = {24: model_info(150, 1.0), 48: model_info(20, 1.0)}
    payload = fc.build_forecast("Lahore", models, feature_row(aqi=170))

    assert payload["alert"]["horizon_hours"] == 24
    assert payload["alert"]["aqi"] == 320
    assert payload["alert"]["severity"] == "critical"
    assert "Hazardous" in payload["alert"]["message"]


def test_alert_can_be_about_right_now():
    payload = fc.build_forecast("Lahore", {24: model_info(-120, 1.0)}, feature_row(aqi=260))
    assert payload["alert"]["horizon_hours"] == 0
    assert "now" in payload["alert"]["message"]


def test_no_alert_when_air_is_acceptable():
    payload = fc.build_forecast("Lahore", {24: model_info(5, 1.0)}, feature_row(aqi=60))
    assert payload["alert"] is None


def test_alert_thresholds_come_from_the_aqi_table():
    """101 is the first Unhealthy-for-Sensitive-Groups value in aqi.py. If that
    table moves, this must move with it -- alerts and the AQI scale must not
    disagree."""
    assert fc.build_forecast("Lahore", {}, feature_row(aqi=100))["alert"] is None
    assert fc.build_forecast("Lahore", {}, feature_row(aqi=101))["alert"]["severity"] == "warning"


# --- feature assembly -------------------------------------------------------


def test_load_feature_row_raises_when_the_city_is_absent(monkeypatch):
    monkeypatch.setattr(
        fc, "read_features_df", lambda: pd.DataFrame({"city_name": ["Karachi"], "aqi": [100]})
    )
    with pytest.raises(fc.ForecastUnavailable, match="No feature rows"):
        fc.load_feature_row("Lahore")


def _store_frame(hours):
    """A feature-store frame with every stored column, so the only thing under
    test is how many hours of history it holds."""
    index = pd.date_range("2026-08-01", periods=hours, freq="h", tz="UTC")
    stored = {
        "city_name": "Lahore",
        "timestamp": index,
        "hour": index.hour,
        "day": index.day,
        "month": index.month,
        "dominant_pollutant": "pm2_5",
        "aqi": np.linspace(100, 180, hours),
        "aqi_change_rate": 1.0,
    }
    for column in (
        "temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_deg",
        "clouds", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    ):
        stored[column] = 1.0
    return pd.DataFrame(stored)


def test_load_feature_row_raises_when_lags_are_incomplete(monkeypatch):
    """Fewer than 72 contiguous hours means the rolling features cannot be
    computed for any row -- serving a forecast from NaN-padded features would be
    worse than serving none."""
    monkeypatch.setattr(fc, "read_features_df", lambda: _store_frame(10))
    with pytest.raises(fc.ForecastUnavailable, match="complete set of lag features"):
        fc.load_feature_row("Lahore")


def test_load_feature_row_returns_the_newest_complete_row(monkeypatch):
    monkeypatch.setattr(fc, "read_features_df", lambda: _store_frame(200))
    row = fc.load_feature_row("Lahore")

    assert not row[fc.FEATURE_COLUMNS].isna().any()
    assert row["timestamp"] == pd.Timestamp("2026-08-09 07:00:00+00:00")


def test_schema_drift_names_the_missing_columns(monkeypatch):
    """The bug this module exists to prevent is training and serving disagreeing
    on the feature set. If it happens anyway, say which columns."""
    frame = _store_frame(200).drop(columns=["pm10", "humidity"])
    monkeypatch.setattr(fc, "read_features_df", lambda: frame)

    with pytest.raises(fc.ForecastUnavailable, match="missing columns"):
        fc.load_feature_row("Lahore")


def test_model_details_excludes_the_model_object():
    details = fc.model_details({24: model_info(-10, 0.5, version=2)})
    assert details == [
        {
            "horizon_hours": 24,
            "model_name": "aqi_forecast_24h",
            "version": 2,
            "blend_weight": 0.5,
            "metrics": {"R2": 0.83, "MAE": 21.4},
        }
    ]
