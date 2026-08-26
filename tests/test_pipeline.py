"""Tests for the hourly pipeline's freshness assertion.

The offline read is substituted, so no network or Hopsworks credentials are
needed. These exist because the insert reporting success while the offline table
stood still for eleven days went undetected by every other check in the project --
and because the first version of this assertion, which looked only at the newest
timestamp, passed while eleven days were missing behind it.
"""

import pandas as pd
import pytest

from feature_pipeline import pipeline as pl
from feature_pipeline.hopsworks_store import OFFLINE_LOOKBACK_HOURS


OBSERVED_AT = pd.Timestamp("2026-08-26 09:00:00+00:00")


def hourly_series(hours, end=OBSERVED_AT):
    """A contiguous run of hourly timestamps ending at `end`."""
    return pd.Series(pd.date_range(end=end, periods=hours, freq="h"))


def patch_offline(monkeypatch, result):
    def fake(city_name):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pl, "offline_timestamps", fake)


EMPTY = pd.Series([], dtype="datetime64[ns, UTC]")


class TestHealthy:
    def test_full_window_passes(self, monkeypatch):
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS))
        assert pl.verify_offline_freshness("Lahore", OBSERVED_AT) is not None

    def test_the_expected_two_percent_of_missing_hours_is_tolerated(self, monkeypatch):
        """1.9% of hours are absent across the whole dataset; that is normal."""
        stamps = hourly_series(OFFLINE_LOOKBACK_HOURS).drop([5, 17]).reset_index(drop=True)
        patch_offline(monkeypatch, stamps)
        assert pl.verify_offline_freshness("Lahore", OBSERVED_AT) is not None

    def test_verdict_reports_coverage(self, monkeypatch, capsys):
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS))
        pl.verify_offline_freshness("Lahore", OBSERVED_AT)
        assert f"{OFFLINE_LOOKBACK_HOURS}/{OFFLINE_LOOKBACK_HOURS} hours present" in capsys.readouterr().out


class TestStale:
    def test_lag_past_the_tolerance_fails(self, monkeypatch):
        end = OBSERVED_AT - pd.Timedelta(hours=pl.STALE_AFTER_HOURS + 1)
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS, end=end))
        with pytest.raises(RuntimeError, match="behind"):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)

    def test_the_eleven_day_stall_fails(self, monkeypatch):
        """The original incident: inserts succeeding, offline frozen at 2026-08-15."""
        end = pd.Timestamp("2026-08-15 02:00:00+00:00")
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS, end=end))
        with pytest.raises(RuntimeError) as error:
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)
        assert pl.MATERIALIZATION_JOB in str(error.value)

    def test_empty_window_fails(self, monkeypatch):
        patch_offline(monkeypatch, EMPTY)
        with pytest.raises(RuntimeError, match="no rows"):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)

    def test_failure_names_the_row_as_kept(self, monkeypatch):
        """The message must not read as data loss -- the insert already succeeded."""
        end = pd.Timestamp("2026-08-15 02:00:00+00:00")
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS, end=end))
        with pytest.raises(RuntimeError, match="not lost"):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)


class TestGapBehindCurrentRows:
    """The failure the first version of this check could not see."""

    def test_a_handful_of_current_rows_over_a_gap_fails(self, monkeypatch):
        """Exactly what 2026-08-26 looked like once materialisation was unblocked:
        three fresh rows, eleven days missing behind them, newest row one hour old."""
        patch_offline(monkeypatch, hourly_series(3))
        with pytest.raises(RuntimeError, match="only 3 of the last"):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)

    def test_the_message_says_to_backfill_not_to_restart_the_job(self, monkeypatch):
        """A gap is not a stalled job, and pointing at the job would waste an hour."""
        patch_offline(monkeypatch, hourly_series(3))
        with pytest.raises(RuntimeError) as error:
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)
        assert "Backfill" in str(error.value)
        assert pl.MATERIALIZATION_JOB not in str(error.value)

    def test_a_gap_is_annotated_as_a_gap_not_as_staleness(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        patch_offline(monkeypatch, hourly_series(3))
        with pytest.raises(RuntimeError):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)
        assert "::error title=Offline freshness::GAP" in capsys.readouterr().out


class TestUnreadable:
    def test_unreadable_offline_store_warns_rather_than_failing(self, monkeypatch, capsys):
        """An Arrow Flight outage must not cost an hour of collection."""
        patch_offline(monkeypatch, ConnectionError("Flight unavailable"))
        assert pl.verify_offline_freshness("Lahore", OBSERVED_AT) is None
        assert "UNVERIFIED" in capsys.readouterr().out


def test_annotation_is_emitted_only_inside_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    pl.announce("CURRENT: all good")
    assert "::notice title=Offline freshness::CURRENT: all good" in capsys.readouterr().out

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    pl.announce("CURRENT: all good")
    assert "::notice" not in capsys.readouterr().out
