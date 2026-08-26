"""Tests for the hourly pipeline's freshness assertion.

The offline read is substituted, so no network or Hopsworks credentials are
needed. These exist because the insert reporting success while the offline table
stood still for eleven days went undetected by every other check in the project.
"""

import pandas as pd
import pytest

from feature_pipeline import pipeline as pl


OBSERVED_AT = pd.Timestamp("2026-08-26 09:00:00+00:00")


def patch_offline(monkeypatch, result):
    """Substitute the offline read: a Timestamp, None, or an exception to raise."""

    def fake(city_name):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pl, "offline_max_timestamp", fake)


def test_current_offline_store_passes(monkeypatch):
    patch_offline(monkeypatch, OBSERVED_AT - pd.Timedelta(hours=1))
    assert pl.verify_offline_freshness("Lahore", OBSERVED_AT) is not None


def test_lag_at_the_tolerance_is_accepted(monkeypatch):
    patch_offline(monkeypatch, OBSERVED_AT - pd.Timedelta(hours=pl.STALE_AFTER_HOURS))
    assert pl.verify_offline_freshness("Lahore", OBSERVED_AT) is not None


def test_lag_past_the_tolerance_fails(monkeypatch):
    patch_offline(monkeypatch, OBSERVED_AT - pd.Timedelta(hours=pl.STALE_AFTER_HOURS + 1))
    with pytest.raises(RuntimeError, match="behind"):
        pl.verify_offline_freshness("Lahore", OBSERVED_AT)


def test_the_eleven_day_stall_fails(monkeypatch):
    """The exact incident: inserts succeeding, offline frozen at 2026-08-15."""
    patch_offline(monkeypatch, pd.Timestamp("2026-08-15 02:00:00+00:00"))
    with pytest.raises(RuntimeError) as error:
        pl.verify_offline_freshness("Lahore", OBSERVED_AT)
    assert pl.MATERIALIZATION_JOB in str(error.value)


def test_empty_lookback_window_fails(monkeypatch):
    patch_offline(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no rows"):
        pl.verify_offline_freshness("Lahore", OBSERVED_AT)


def test_unreadable_offline_store_warns_rather_than_failing(monkeypatch, capsys):
    """An Arrow Flight outage must not cost an hour of collection."""
    patch_offline(monkeypatch, ConnectionError("Flight unavailable"))
    assert pl.verify_offline_freshness("Lahore", OBSERVED_AT) is None
    assert "unverified" in capsys.readouterr().out


def test_failure_names_the_row_as_kept(monkeypatch):
    """The message must not read as data loss -- the insert already succeeded."""
    patch_offline(monkeypatch, pd.Timestamp("2026-08-15 02:00:00+00:00"))
    with pytest.raises(RuntimeError, match="not lost"):
        pl.verify_offline_freshness("Lahore", OBSERVED_AT)
