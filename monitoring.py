"""Optional error reporting, shared by both front ends and the scheduled jobs.

Every entry point calls start_monitoring(); nothing changes when SENTRY_DSN is
unset, so a fresh clone, the test suite and local runs never need an account.

This exists because of how the failures in this project were actually found. The
offline store stood still for eleven days, and 264 green pipeline runs, a green
test suite and a live dashboard all agreed nothing was wrong -- it was noticed by
a person reading the page. Annotations made the state visible to anyone who opens
a run; this makes a failure arrive without anyone looking.
"""

import functools
import os

# Traces off. What this project needs is exception reports, not latency profiles,
# and the free allowance is worth more spent on errors.
TRACES_SAMPLE_RATE = 0.0

_enabled = False


def start_monitoring(component):
    """Initialise error reporting for one entry point. Returns True if enabled.

    `component` tags every event, so a failing hourly job is distinguishable from
    a failing dashboard before opening the stack trace.
    """
    global _enabled
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    try:
        import sentry_sdk
    except ImportError:
        # A missing optional dependency must not take down the thing being
        # monitored, so this is reported and ignored rather than raised.
        print("SENTRY_DSN is set but sentry-sdk is not installed; running unmonitored.")
        return False

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=TRACES_SAMPLE_RATE,
        send_default_pii=False,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
    )
    sentry_sdk.set_tag("component", component)
    _enabled = True
    return True


def report(error, **context):
    """Send an exception that was already handled locally. Returns True if sent.

    The dashboard deliberately swallows feature-store read failures so a reader
    sees the last good reading rather than a traceback. Swallowed is not the same
    as unnoticed: without this, the degraded path would be the one failure mode
    nobody ever hears about.
    """
    if not _enabled:
        return False

    import sentry_sdk

    with sentry_sdk.new_scope() as scope:
        for key, value in context.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_exception(error)
    return True


def note(message, level="warning", **context):
    """Send a message for a condition that is wrong but did not raise.

    An unverifiable freshness check is the example: the row was written, nothing
    threw, and the one thing the check exists to confirm went unconfirmed.
    """
    if not _enabled:
        return False

    import sentry_sdk

    with sentry_sdk.new_scope() as scope:
        for key, value in context.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_message(message, level=level)
    return True


# GitHub does not run scheduled workflows on time. Observed firings on
# 2026-08-26 were 12:27, 13:43, 14:32, 16:06, 18:31, 21:21 and 02:18 UTC --
# an "hourly" job with gaps of up to five hours. A tight check-in margin would
# alert on GitHub's queueing rather than on this project's health, so the margin
# is set well beyond the worst observed delay: an alert here should mean the job
# has stopped, not that it is late.
CHECKIN_MARGIN_MINUTES = 360
MAX_RUNTIME_MINUTES = 15


def monitored_job(slug, crontab):
    """Report each run of a scheduled job to Sentry, so a job that stops running
    raises an alarm.

    This is the piece that addresses the incident directly. Error reporting only
    fires when something throws; between 2026-08-15 and 2026-08-26 nothing threw
    -- the pipeline ran, reported success, and the offline store stood still. A
    check-in tells Sentry the job ran and finished, which means a job that stops
    checking in becomes a notification instead of a silence.

    Without SENTRY_DSN the decorator returns the function unchanged.
    """

    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return function(*args, **kwargs)

            from sentry_sdk.crons import monitor

            config = {
                "schedule": {"type": "crontab", "value": crontab},
                "checkin_margin": CHECKIN_MARGIN_MINUTES,
                "max_runtime": MAX_RUNTIME_MINUTES,
                "timezone": "UTC",
            }
            return monitor(monitor_slug=slug, monitor_config=config)(function)(*args, **kwargs)

        return wrapper

    return decorate
