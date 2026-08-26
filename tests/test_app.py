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
