"""Tests for the dashboard's presentation helpers.

Streamlit is imported but never run: only the pure functions are exercised, so
these need no server and no credentials.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app import app as dash
from feature_pipeline.aqi import _CATEGORIES


def payload_observed_hours_ago(hours):
    observed = datetime.now(timezone.utc) - timedelta(hours=hours)
    return {"observed_at": observed.replace(microsecond=0).isoformat()}


class TestContrast:
    def test_yellow_gets_dark_text(self):
        assert dash.readable_text_color("#ffff00") == "#111111"

    def test_dark_red_gets_light_text(self):
        assert dash.readable_text_color("#7e0023") == "#ffffff"

    def test_every_epa_category_is_legible(self):
        """Not a snapshot of today's palette -- an assertion that whatever colour a
        category carries, the text on it is the higher-contrast of black or white."""
        for *_, color in _CATEGORIES:
            chosen = dash.readable_text_color(color)
            assert chosen in ("#111111", "#ffffff")
            assert contrast_ratio(color, chosen) >= contrast_ratio(
                color, "#ffffff" if chosen == "#111111" else "#111111"
            )


def contrast_ratio(background_hex, text_hex):
    lighter, darker = sorted((_luminance(background_hex), _luminance(text_hex)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _luminance(hex_color):
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


class TestObservationAge:
    def test_age_is_measured_from_now(self):
        age_hours, _, _ = dash.observation_age(payload_observed_hours_ago(5))
        assert 4.9 < age_hours < 5.1

    def test_timestamp_is_rendered_in_local_time(self):
        _, local_text, tz_label = dash.observation_age(
            {"observed_at": "2026-08-26T02:00:00+00:00"}
        )
        # Asia/Karachi is UTC+5, so 02:00 UTC is 07:00 the same day.
        assert "07:00" in local_text
        assert "26 August 2026" in local_text
        assert tz_label

    @pytest.mark.parametrize(
        "hours, expected",
        [(0.2, "less than an hour ago"), (1.0, "less than an hour ago"),
         (3.0, "3 hours ago"), (30.0, "30 hours ago"), (264.0, "11 days ago")],
    )
    def test_relative_phrasing(self, hours, expected):
        assert dash.relative_age(hours) == expected

    def test_the_eleven_day_incident_reads_as_days(self):
        """2026-08-15 02:00 UTC served on 2026-08-26 -- the reading that went
        unnoticed because the page printed a raw ISO timestamp."""
        assert dash.relative_age(265.0) == "11 days ago"


class TestDegradation:
    """The page must never answer a remote failure with a traceback."""

    def test_successful_payload_is_retained(self, monkeypatch):
        dash.last_good_payload().clear()
        monkeypatch.setattr(dash, "fetch_payload", lambda city: {"observed_at": "x"})
        payload, live = dash.fetch_payload_or_last_good("Lahore")
        assert live is True
        assert dash.last_good_payload()["payload"] == payload

    def test_retained_payload_is_served_when_the_read_fails(self, monkeypatch):
        dash.last_good_payload().clear()
        monkeypatch.setattr(dash, "fetch_payload", lambda city: {"observed_at": "good"})
        dash.fetch_payload_or_last_good("Lahore")

        def boom(city):
            raise RuntimeError("Query Service refused the read")

        monkeypatch.setattr(dash, "fetch_payload", boom)
        payload, live = dash.fetch_payload_or_last_good("Lahore")
        assert live is False
        assert payload["observed_at"] == "good"

    def test_failure_with_nothing_retained_propagates(self, monkeypatch):
        dash.last_good_payload().clear()

        def boom(city):
            raise RuntimeError("Query Service refused the read")

        monkeypatch.setattr(dash, "fetch_payload", boom)
        with pytest.raises(RuntimeError):
            dash.fetch_payload_or_last_good("Lahore")


class TestEvaluationPanel:
    """The walk-forward comparison is the project's strongest claim; it has to
    reach the page correctly and it has to be honest about partial wins."""

    def details(self, metrics, horizon=24, version=7, weight=0.35):
        return [
            {
                "horizon_hours": horizon,
                "model_name": f"aqi_forecast_{horizon}h",
                "version": version,
                "blend_weight": weight,
                "metrics": metrics,
            }
        ]

    def test_a_clear_win_lists_every_metric(self):
        wins = dash.beats_baseline(
            {
                "R2": 0.715, "persistence_R2": 0.628,
                "MAE": 24.0, "persistence_MAE": 30.0,
                "RMSE": 40.0, "persistence_RMSE": 52.0,
            }
        )
        assert wins == ["R2", "MAE", "RMSE"]

    def test_the_real_24h_result_does_not_claim_an_mae_win(self):
        """Production 24h: R2 0.832 vs 0.814, but MAE 30.2 against 29.8 -- worse.
        The panel must not round that into "beats the baseline"."""
        wins = dash.beats_baseline(
            {
                "R2": 0.832, "persistence_R2": 0.814,
                "MAE": 30.2, "persistence_MAE": 29.8,
                "RMSE": 45.0, "persistence_RMSE": 48.0,
            }
        )
        assert wins == ["R2", "RMSE"]
        assert "MAE" not in wins

    def test_a_missing_baseline_is_never_counted_as_a_win(self):
        """persistence_RMSE does not exist on models registered before it was
        added; absent evidence is not evidence."""
        wins = dash.beats_baseline({"R2": 0.8, "persistence_R2": 0.7, "RMSE": 40.0})
        assert wins == ["R2"]

    def test_losing_on_everything_says_so(self):
        wins = dash.beats_baseline(
            {
                "R2": 0.5, "persistence_R2": 0.7,
                "MAE": 40.0, "persistence_MAE": 30.0,
                "RMSE": 60.0, "persistence_RMSE": 50.0,
            }
        )
        assert wins == []
        row = dash.evaluation_rows(self.details({
            "R2": 0.5, "persistence_R2": 0.7,
            "MAE": 40.0, "persistence_MAE": 30.0,
            "RMSE": 60.0, "persistence_RMSE": 50.0,
        }))[0]
        assert row["Beats baseline on"] == "nothing"

    def test_row_carries_the_baseline_beside_the_model(self):
        row = dash.evaluation_rows(self.details({
            "R2": 0.832, "persistence_R2": 0.814,
            "MAE": 30.2, "persistence_MAE": 29.8,
            "RMSE": 45.0, "persistence_RMSE": 48.0,
        }))[0]
        assert row["Horizon"] == "24h"
        assert row["R2"] == "0.832" and row["R2 baseline"] == "0.814"
        assert row["MAE"] == "30.2" and row["MAE baseline"] == "29.8"
        assert row["Blend weight"] == "0.35"
        assert row["Version"] == 7

    def test_missing_numbers_render_as_a_dash_not_a_zero(self):
        row = dash.evaluation_rows(self.details({}))[0]
        assert row["R2"] == "—" and row["RMSE baseline"] == "—"

    def test_an_unregistered_weight_renders_as_a_dash(self):
        row = dash.evaluation_rows(self.details({}, weight=None))[0]
        assert row["Blend weight"] == "—"
