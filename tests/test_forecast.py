"""Tests for the serving layer: the blend arithmetic and the payload shape.

The feature store and registered models are substituted, so no network or
Hopsworks credentials are needed.
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
    """None must never be treated as 1.0: the unshrunk model loses to
    persistence at every horizon."""
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
    """A text column cannot be interpolated across a gap in the hourly grid, so
    it arrives as NaN rather than a string."""
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
    """101 is the first Unhealthy-for-Sensitive-Groups value in aqi.py; alerts and
    the AQI scale must not disagree."""
    assert fc.build_forecast("Lahore", {}, feature_row(aqi=100))["alert"] is None
    assert fc.build_forecast("Lahore", {}, feature_row(aqi=101))["alert"]["severity"] == "warning"


# --- feature assembly -------------------------------------------------------


def test_load_feature_row_raises_when_the_city_is_absent(monkeypatch):
    monkeypatch.setattr(
        fc, "read_recent_features_df",
        lambda city: pd.DataFrame(
            {
                "city_name": ["Karachi"] * 96,
                "aqi": [100] * 96,
                "timestamp": pd.date_range("2026-08-01", periods=96, freq="h", tz="UTC"),
            }
        ),
    )
    with pytest.raises(fc.ForecastUnavailable, match="No feature rows"):
        fc.load_feature_row("Lahore")


def _store_frame(hours):
    """Every stored column, so only the amount of history varies."""
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
    """Under 72 contiguous hours, no row has complete rolling features."""
    monkeypatch.setattr(fc, "read_recent_features_df", lambda city: _store_frame(10))
    monkeypatch.setattr(fc, "read_features_df", lambda: _store_frame(10))
    with pytest.raises(fc.ForecastUnavailable, match="complete set of lag features"):
        fc.load_feature_row("Lahore")


def test_load_feature_row_returns_the_newest_complete_row(monkeypatch):
    monkeypatch.setattr(fc, "read_recent_features_df", lambda city: _store_frame(200))
    row = fc.load_feature_row("Lahore")

    assert not row[fc.FEATURE_COLUMNS].isna().any()
    assert row["timestamp"] == pd.Timestamp("2026-08-09 07:00:00+00:00")


def test_schema_drift_names_the_missing_columns(monkeypatch):
    """If training and serving disagree on the feature set, say which columns."""
    frame = _store_frame(200).drop(columns=["pm10", "humidity"])
    monkeypatch.setattr(fc, "read_recent_features_df", lambda city: frame)

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


class TestFeatureHistory:
    """The bounded read is the fast path; the full read is the safety net."""

    def test_bounded_read_is_used_when_it_returns_enough_rows(self, monkeypatch):
        recent = pd.DataFrame(
            {
                "city_name": ["Lahore"] * fc.MIN_HISTORY_ROWS,
                "timestamp": pd.date_range(
                    "2026-08-01", periods=fc.MIN_HISTORY_ROWS, freq="h", tz="UTC"
                ),
            }
        )
        monkeypatch.setattr(fc, "read_recent_features_df", lambda city: recent)
        monkeypatch.setattr(fc, "read_features_df", _must_not_be_called)
        assert len(fc.load_feature_history("Lahore")) == fc.MIN_HISTORY_ROWS

    def test_short_bounded_read_falls_back_to_full_history(self, monkeypatch):
        """Happens when the offline store has fallen behind the lookback window."""
        monkeypatch.setattr(
            fc,
            "read_recent_features_df",
            lambda city: pd.DataFrame(
                {"city_name": ["Lahore"], "timestamp": pd.to_datetime(["2026-08-01"], utc=True)}
            ),
        )
        monkeypatch.setattr(
            fc,
            "read_features_df",
            lambda: pd.DataFrame(
                {"city_name": ["full"], "timestamp": pd.to_datetime(["2026-08-01"], utc=True)}
            ),
        )
        assert fc.load_feature_history("Lahore")["city_name"].tolist() == ["full"]

def _must_not_be_called():
    raise AssertionError("full feature-group read should not happen on the fast path")


def test_failed_bounded_read_is_not_escalated_to_a_full_read(monkeypatch):
    """Retrying a refusing Query Service with a 200x larger request only fails
    again, slower. The caller degrades instead."""

    def boom(city):
        raise ConnectionError("Flight unavailable")

    monkeypatch.setattr(fc, "read_recent_features_df", boom)
    monkeypatch.setattr(fc, "read_features_df", _must_not_be_called)
    with pytest.raises(fc.ForecastUnavailable, match="ConnectionError"):
        fc.load_feature_history("Lahore")


class TestObservedHistory:
    """Three future numbers cannot show whether a forecast continues the recent
    trend or breaks from it; the observations behind them can."""

    def frame(self, hours, end="2026-08-27 12:00"):
        index = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=hours, freq="h")
        return pd.DataFrame({"timestamp": index, "aqi": np.linspace(100, 150, hours)})

    def test_it_returns_records_oldest_first(self):
        history = fc.observed_history(self.frame(5))
        assert len(history) == 5
        assert history[0]["timestamp"] < history[-1]["timestamp"]
        assert history[-1]["aqi"] == 150

    def test_it_is_bounded_to_the_history_window(self):
        history = fc.observed_history(self.frame(200))
        assert len(history) == fc.HISTORY_HOURS + 1  # inclusive of the cutoff hour

    def test_an_empty_frame_gives_an_empty_history(self):
        assert fc.observed_history(pd.DataFrame({"timestamp": [], "aqi": []})) == []

    def test_missing_readings_are_skipped_not_rendered_as_nan(self):
        frame = self.frame(5)
        frame.loc[2, "aqi"] = np.nan
        assert len(fc.observed_history(frame)) == 4

    def test_the_payload_carries_the_history(self):
        payload = fc.build_forecast(
            "Lahore",
            {24: model_info(-20, 0.5)},
            feature_row(),
            history=[{"timestamp": "2026-08-24T06:00:00+00:00", "aqi": 170}],
        )
        assert payload["history"] == [{"timestamp": "2026-08-24T06:00:00+00:00", "aqi": 170}]

    def test_history_defaults_to_empty_so_the_api_contract_is_stable(self):
        payload = fc.build_forecast("Lahore", {24: model_info(-20, 0.5)}, feature_row())
        assert payload["history"] == []
