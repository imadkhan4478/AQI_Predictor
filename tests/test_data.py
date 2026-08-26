"""Tests for the future-row guard.

Written after a backfill wrote eleven forecast hours into the feature store as
though they were observations.
"""

import pandas as pd

from training_pipeline.data import drop_future_rows


NOW = pd.Timestamp("2026-08-26 12:00:00+00:00")


def frame(hours):
    return pd.DataFrame({"timestamp": pd.to_datetime(hours, utc=True), "aqi": range(len(hours))})


def test_past_and_current_hours_are_kept():
    df = frame(["2026-08-26 10:00", "2026-08-26 11:00", "2026-08-26 12:00"])
    assert len(drop_future_rows(df, now=NOW)) == 3


def test_the_eleven_forecast_hours_are_dropped():
    hours = [f"2026-08-26 {h:02d}:00" for h in range(13, 24)]
    df = frame(["2026-08-26 12:00"] + hours)
    kept = drop_future_rows(df, now=NOW)
    assert len(kept) == 1
    assert kept["timestamp"].max() == NOW


def test_it_says_how_many_it_dropped(capsys):
    hours = [f"2026-08-26 {h:02d}:00" for h in range(13, 24)]
    drop_future_rows(frame(hours), now=NOW)
    assert "Dropped 11 rows" in capsys.readouterr().out


def test_a_clean_frame_is_left_alone(capsys):
    drop_future_rows(frame(["2026-08-26 11:00"]), now=NOW)
    assert "Dropped" not in capsys.readouterr().out
