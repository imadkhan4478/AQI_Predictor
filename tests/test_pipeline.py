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


class TestFutureRows:
    """The backfill of 2026-08-26 wrote 11 hours that had not happened yet, and the
    coverage check reported 59/48 with a negative lag rather than failing."""

    def test_rows_dated_after_the_insert_fail(self, monkeypatch):
        stamps = hourly_series(48, end=OBSERVED_AT + pd.Timedelta(hours=11))
        patch_offline(monkeypatch, stamps)
        with pytest.raises(RuntimeError, match="in the future"):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)

    def test_future_rows_are_annotated_as_future(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        patch_offline(monkeypatch, hourly_series(48, end=OBSERVED_AT + pd.Timedelta(hours=11)))
        with pytest.raises(RuntimeError):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)
        assert "::error title=Offline freshness::FUTURE" in capsys.readouterr().out

    def test_more_rows_than_hours_fails(self, monkeypatch):
        """Duplicate primary keys, with no future dates to explain them."""
        stamps = pd.concat([hourly_series(48), hourly_series(10)], ignore_index=True)
        patch_offline(monkeypatch, stamps.sort_values().reset_index(drop=True))
        with pytest.raises(RuntimeError, match="More rows than hours"):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)


class TestFailureIsLegible:
    """Run #443 failed with 'exit code 1' and no annotation naming a cause, which
    left the reason readable only in a log panel that would not render."""

    def test_an_unexpected_failure_is_annotated_with_its_cause(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")

        def boom():
            raise KeyError("OPENWEATHER_API_KEY")

        monkeypatch.setattr(pl, "run", boom)
        with pytest.raises(KeyError):
            pl.main()
        printed = capsys.readouterr().out
        assert "::error title=Feature pipeline::KeyError" in printed
        assert "OPENWEATHER_API_KEY" in printed

    def test_the_exception_still_propagates_so_the_run_goes_red(self, monkeypatch):
        """Annotating must not swallow the failure."""

        def boom():
            raise RuntimeError("insert rejected")

        monkeypatch.setattr(pl, "run", boom)
        with pytest.raises(RuntimeError, match="insert rejected"):
            pl.main()

    def test_a_successful_run_is_not_annotated_as_a_failure(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setattr(pl, "run", lambda: None)
        pl.main()
        assert "::error" not in capsys.readouterr().out


class FakeExecution:
    def __init__(self, state, duration_minutes):
        self.state = state
        self.duration = duration_minutes * 60000
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeJob:
    def __init__(self, executions):
        self._executions = executions
        self.runs = 0

    def get_executions(self):
        return self._executions

    def run(self, await_termination=True):
        self.runs += 1


class FakeFeatureGroup:
    def __init__(self, job):
        self.materialization_job = job


class TestReviveMaterialization:
    """The job has stalled twice in two days. Detecting it is worth little if the
    remedy is always a person in a web UI."""

    def test_a_wedged_execution_is_stopped_and_the_job_restarted(self):
        wedged = FakeExecution("INITIALIZING", duration_minutes=16000)  # the 11-day stall
        job = FakeJob([wedged])
        assert pl.revive_materialization(FakeFeatureGroup(job)) is True
        assert wedged.stopped is True
        assert job.runs == 1

    def test_a_job_that_is_merely_slow_is_left_alone(self):
        """A healthy pass takes about four minutes. Killing one at ten would turn
        a working system into a broken one."""
        working = FakeExecution("RUNNING", duration_minutes=10)
        job = FakeJob([working])
        assert pl.revive_materialization(FakeFeatureGroup(job)) is False
        assert working.stopped is False
        assert job.runs == 0

    def test_a_healthy_job_is_not_restarted_and_the_real_cause_is_named(self, monkeypatch, capsys):
        """2026-08-27: every execution green, and the store still behind -- because
        the workflow had fired three times in seventeen hours. Re-running a healthy
        job would have looked like a fix and changed nothing."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        job = FakeJob([FakeExecution("FINISHED", duration_minutes=3)])
        assert pl.revive_materialization(FakeFeatureGroup(job)) is False
        assert job.runs == 0
        printed = capsys.readouterr().out
        assert "no wedged execution" in printed
        assert "how often the workflow is actually firing" in printed

    def test_an_api_failure_warns_and_names_the_manual_remedy(self, monkeypatch, capsys):
        """This runs on the already-broken path; it must not replace a clear
        diagnosis with a stack trace."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")

        class Broken:
            @property
            def materialization_job(self):
                raise RuntimeError("job metadata unavailable")

        assert pl.revive_materialization(Broken()) is False
        printed = capsys.readouterr().out
        assert "::warning title=Materialisation" in printed
        assert "Hopsworks Jobs UI" in printed

    def test_staleness_triggers_a_revival_and_still_fails_the_run(self, monkeypatch):
        """Self-healing must not hide the incident: the run stays red."""
        wedged = FakeExecution("INITIALIZING", duration_minutes=1000)
        job = FakeJob([wedged])
        end = pd.Timestamp("2026-08-27 02:00:00+00:00")
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS, end=end))

        with pytest.raises(RuntimeError, match="behind"):
            pl.verify_offline_freshness(
                "Lahore", OBSERVED_AT + pd.Timedelta(hours=25), feature_group=FakeFeatureGroup(job)
            )
        assert job.runs == 1

    def test_no_feature_group_means_no_revival_attempt(self, monkeypatch):
        """Callers that only want the verdict, and every existing test, stay valid."""
        end = pd.Timestamp("2026-08-15 02:00:00+00:00")
        patch_offline(monkeypatch, hourly_series(OFFLINE_LOOKBACK_HOURS, end=end))
        with pytest.raises(RuntimeError):
            pl.verify_offline_freshness("Lahore", OBSERVED_AT)
