"""Tests for the optional error reporting.

The contract that matters is that everything works with no account configured: a
fresh clone, this test suite and a local run must never need SENTRY_DSN, and a
missing sentry-sdk must never take down the thing it was meant to watch.
"""

import sys

import monitoring


class FakeScope:
    def __init__(self):
        self.extras = {}

    def set_extra(self, key, value):
        self.extras[key] = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSentry:
    """Stands in for sentry_sdk, recording what it was asked to send."""

    def __init__(self):
        self.init_kwargs = None
        self.tags = {}
        self.exceptions = []
        self.messages = []
        self.scope = FakeScope()

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def set_tag(self, key, value):
        self.tags[key] = value

    def new_scope(self):
        return self.scope

    def capture_exception(self, error):
        self.exceptions.append(error)

    def capture_message(self, message, level=None):
        self.messages.append((message, level))


def install(monkeypatch, fake=None):
    fake = fake or FakeSentry()
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    monkeypatch.setattr(monitoring, "_enabled", False)
    return fake


class TestDisabledByDefault:
    def test_no_dsn_means_no_monitoring(self, monkeypatch):
        install(monkeypatch)
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        assert monitoring.start_monitoring("dashboard") is False

    def test_a_blank_dsn_counts_as_unset(self, monkeypatch):
        """Streamlit secrets and Actions both hand through empty strings for a
        secret that was never filled in."""
        install(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "   ")
        assert monitoring.start_monitoring("dashboard") is False

    def test_reporting_without_monitoring_is_a_no_op(self, monkeypatch):
        install(monkeypatch)
        assert monitoring.report(ValueError("boom")) is False
        assert monitoring.note("something odd") is False


class TestEnabled:
    def test_init_tags_the_component(self, monkeypatch):
        fake = install(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        assert monitoring.start_monitoring("feature-pipeline") is True
        assert fake.tags["component"] == "feature-pipeline"

    def test_traces_are_off_and_pii_is_not_sent(self, monkeypatch):
        """The free allowance is spent on errors, and nothing here needs personal
        data to be useful."""
        fake = install(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monitoring.start_monitoring("api")
        assert fake.init_kwargs["traces_sample_rate"] == 0.0
        assert fake.init_kwargs["send_default_pii"] is False

    def test_a_handled_exception_is_sent_with_context(self, monkeypatch):
        """The dashboard swallows read failures on purpose; swallowed must not
        mean unnoticed."""
        fake = install(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monitoring.start_monitoring("dashboard")

        error = ConnectionError("Query Service refused the read")
        assert monitoring.report(error, city="Lahore", served_retained_payload=True) is True
        assert fake.exceptions == [error]
        assert fake.scope.extras["city"] == "Lahore"
        assert fake.scope.extras["served_retained_payload"] is True

    def test_a_condition_that_did_not_raise_can_still_be_sent(self, monkeypatch):
        fake = install(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")
        monitoring.start_monitoring("feature-pipeline")

        monitoring.note("Offline freshness unverified", city="Lahore")
        assert fake.messages == [("Offline freshness unverified", "warning")]


class TestMissingDependency:
    def test_a_missing_sdk_does_not_break_the_caller(self, monkeypatch, capsys):
        """A monitoring dependency taking down the monitored job would be worse
        than no monitoring."""
        monkeypatch.setattr(monitoring, "_enabled", False)
        monkeypatch.setitem(sys.modules, "sentry_sdk", None)
        monkeypatch.setenv("SENTRY_DSN", "https://key@example.ingest.sentry.io/1")

        import builtins

        real_import = builtins.__import__

        def fail_on_sentry(name, *args, **kwargs):
            if name == "sentry_sdk":
                raise ImportError("No module named 'sentry_sdk'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_sentry)
        assert monitoring.start_monitoring("api") is False
        assert "running unmonitored" in capsys.readouterr().out


class TestScheduledJobCheckIns:
    """Error reporting only fires when something throws. For eleven days nothing
    threw -- the job ran, reported success, and the store stood still. A check-in
    turns 'stopped running' from silence into a notification."""

    def test_the_job_runs_unchanged_when_monitoring_is_off(self, monkeypatch):
        install(monkeypatch)
        calls = []

        @monitoring.monitored_job("aqi-feature-pipeline", crontab="0 * * * *")
        def job():
            calls.append(1)
            return "done"

        assert job() == "done"
        assert calls == [1]

    def test_the_check_in_margin_tolerates_github_schedule_drift(self):
        """Observed firings on 2026-08-26 were up to five hours apart. An alert
        must mean the job stopped, not that GitHub queued it."""
        assert monitoring.CHECKIN_MARGIN_MINUTES >= 300

    def test_a_failing_job_still_raises(self, monkeypatch):
        install(monkeypatch)

        @monitoring.monitored_job("aqi-feature-pipeline", crontab="0 * * * *")
        def job():
            raise RuntimeError("insert rejected")

        import pytest

        with pytest.raises(RuntimeError, match="insert rejected"):
            job()
