"""Tests for the ARIMA baseline.

It is the only candidate that needs a continuous series rather than independent
rows, which makes it the only one that breaks when the series has holes.
"""

import numpy as np
import pandas as pd
import pytest

from training_pipeline import statistical_model as sm


def context(hours=600, test_hours=80, gaps=((200, 207), (400, 404)), horizon_hours=24):
    """A gap-free hourly series plus the split boundaries predict_delta needs.

    `gaps` removes hours from the *source* readings, mirroring the hours missing
    in the feature store, before the series is completed the way train.py
    completes it.
    """
    index = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    values = 180 + 30 * np.sin(2 * np.pi * np.arange(hours) / 24) + np.linspace(0, -20, hours)

    observed = pd.Series(values, index=index)
    for start, end in gaps:
        observed = observed.drop(observed.index[start:end])

    # Exactly what train.py does: reindex onto a complete hourly range so the
    # series can be extended incrementally.
    series = observed.reindex(
        pd.date_range(observed.index.min(), observed.index.max(), freq="h")
    ).interpolate(limit_direction="both")

    split_at = len(series) - test_hours
    train_timestamps = pd.Series(series.index[:split_at])
    test_timestamps = pd.Series(series.index[split_at:])

    return {
        "aqi_series": series,
        "train_timestamps": train_timestamps,
        "test_timestamps": test_timestamps,
        "current_aqi_test": series.iloc[split_at:].to_numpy(dtype=float),
        "horizon_hours": horizon_hours,
    }


def test_predicts_a_delta_for_every_test_row():
    ctx = context()
    deltas = sm.predict_delta(ctx)

    assert len(deltas) == len(ctx["test_timestamps"])
    assert np.isfinite(deltas).all()


def test_survives_gaps_in_the_source_readings():
    """A series with holes cannot be extended incrementally -- ARIMA requires
    each appended block to continue exactly where the model's data ends. This
    failed in production with "Given `endog` does not have an index that extends
    the index of the model"."""
    deltas = sm.predict_delta(context(gaps=((100, 118), (300, 307), (450, 452))))
    assert np.isfinite(deltas).all()


def test_forecast_updates_as_the_origin_advances():
    """If the origin never moved, every test row would share one forecast and
    the whole rolling-origin design would be silently pointless."""
    deltas = sm.predict_delta(context())
    assert len(np.unique(np.round(deltas, 6))) > 1


@pytest.mark.parametrize("horizon_hours", [24, 72])
def test_runs_at_every_deployed_horizon(horizon_hours):
    deltas = sm.predict_delta(context(horizon_hours=horizon_hours))
    assert np.isfinite(deltas).all()


def test_stride_bounds_how_stale_a_forecast_can_be():
    """Rows between origins are served by the origin at the start of their
    block, so consecutive rows repeat at most ORIGIN_STRIDE_HOURS times."""
    deltas = sm.predict_delta(context())
    longest_run = max(len(list(group)) for _, group in __import__("itertools").groupby(deltas.round(6)))
    assert longest_run <= sm.ORIGIN_STRIDE_HOURS
