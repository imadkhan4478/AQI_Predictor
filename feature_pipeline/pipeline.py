"""Fetch weather + pollution data, compute AQI, and write one feature row to Hopsworks."""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError
from dotenv import load_dotenv

from feature_pipeline.aqi import compute_aqi
from feature_pipeline.fetch import get_pollution, get_pollution_at, get_weather
from feature_pipeline.hopsworks_store import (
    OFFLINE_LOOKBACK_HOURS,
    get_feature_group,
    offline_timestamps,
)
from monitoring import monitored_job, note, start_monitoring

load_dotenv()

MAX_INSERT_ATTEMPTS = 3
STALE_AFTER_HOURS = 6

# An hourly pipeline should leave roughly one row per hour in the window. Well
# under that means a gap, which the newest timestamp alone cannot reveal.
MIN_WINDOW_COVERAGE = 0.8

# More rows than hours means duplicate keys or, as on 2026-08-26, rows dated in
# the future. Either way the window is not what it claims to be.
MAX_WINDOW_COVERAGE = 1.1

MATERIALIZATION_JOB = "aqi_features_2_offline_fg_materialization"

# An execution still non-terminal after this long is wedged, not working. The
# stall of 2026-08-15 sat in Initializing for eleven days; a healthy run of this
# job takes about four minutes.
WEDGED_AFTER_MINUTES = 45
NON_TERMINAL_STATES = {"INITIALIZING", "RUNNING", "STARTING_APP_MASTER", "ACCEPTED", "NEW"}


def build_feature_row(city_name, weather, pollution, previous_aqi=None):
    components = pollution["list"][0]["components"]
    aqi, dominant_pollutant, _ = compute_aqi(components)
    # Floor to the hour. OpenWeather's "dt" is the observation time (11:19:16),
    # while the backfill wrote hour-aligned timestamps; labels are built by
    # matching t+24/48/72h exactly, so an unaligned row can never match one. It
    # also lets the (city_name, timestamp) key deduplicate two runs in one hour.
    timestamp = datetime.fromtimestamp(weather["dt"], tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )

    return {
        "city_name": city_name,
        "timestamp": timestamp,
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        # Explicit float() on every 'double' column: with a single-row insert,
        # pandas types a whole number (OpenWeather returns "no": 0) as int64 and
        # the insert fails against the schema the backfill locked in.
        "temp": float(weather["main"]["temp"]),
        "feels_like": float(weather["main"]["feels_like"]),
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": float(weather["wind"]["speed"]),
        "wind_deg": weather["wind"].get("deg", 0),
        "clouds": weather["clouds"]["all"],
        "co": float(components["co"]),
        "no": float(components["no"]),
        "no2": float(components["no2"]),
        "o3": float(components["o3"]),
        "so2": float(components["so2"]),
        "pm2_5": float(components["pm2_5"]),
        "pm10": float(components["pm10"]),
        "nh3": float(components["nh3"]),
        "dominant_pollutant": dominant_pollutant,
        "aqi": aqi,
        "aqi_change_rate": float(aqi - previous_aqi) if previous_aqi is not None else 0.0,
    }


def get_previous_hour_aqi(lat, lon, api_key, observed_at):
    """AQI one hour before `observed_at`, recomputed from the pollution archive.

    Returns None if that hour is not published yet; the caller then records a
    change rate of 0.0."""
    components = get_pollution_at(lat, lon, api_key, observed_at - timedelta(hours=1))
    if components is None:
        return None
    aqi, _, _ = compute_aqi(components)
    return aqi


def announce(message, level="notice", title="Offline freshness"):
    """Report to the log, and to the run summary when running in Actions.

    A number buried in a collapsed log step is not monitoring. As an annotation it
    appears on the run page and in the checks API, so the state can be read without
    expanding anything -- which also happens to be the only way to read it when the
    log viewer will not render.
    """
    print(message, flush=True)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title={title}::{message}", flush=True)


def revive_materialization(feature_group):
    """Stop a wedged materialisation execution and start a fresh one.

    Detecting the stall is worth little if the remedy is always a person in a web
    UI. It has now happened twice in two days, and each time the offline store --
    and therefore the dashboard, the API and training -- stood still until someone
    noticed. So the pipeline attempts its own remedy and still fails the run: the
    failure stays visible, and the next hourly insert finds the store current.

    Everything here is best-effort. This runs only on the already-broken path, so
    a failure to repair must not replace a clear diagnosis with a stack trace.
    """
    try:
        job = feature_group.materialization_job
        stopped = 0
        for execution in job.get_executions():
            state = str(getattr(execution, "state", "")).upper()
            if state not in NON_TERMINAL_STATES:
                continue
            minutes = (getattr(execution, "duration", 0) or 0) / 60000
            if minutes < WEDGED_AFTER_MINUTES:
                announce(
                    f"{MATERIALIZATION_JOB} is {state} after {minutes:.0f}m; leaving it alone",
                    title="Materialisation",
                )
                return False
            execution.stop()
            stopped += 1

        if not stopped:
            # Nothing was wedged, so materialisation is not the reason the store is
            # behind -- the inserts are. On 2026-08-27 the workflow fired three
            # times in seventeen hours and the newest offline row was simply the
            # last successful insert. Re-running a healthy job would have looked
            # like a fix and changed nothing.
            announce(
                f"{MATERIALIZATION_JOB} has no wedged execution, so it is not the cause. "
                "The offline store is behind because inserts have not been reaching it -- "
                "check how often the workflow is actually firing.",
                level="warning",
                title="Materialisation",
            )
            return False

        job.run(await_termination=False)
    except Exception as error:
        announce(
            f"Could not revive {MATERIALIZATION_JOB} ({type(error).__name__}: {error}). "
            "Stop its execution in the Hopsworks Jobs UI and run it again.",
            level="warning",
            title="Materialisation",
        )
        return False

    announce(
        f"Revived {MATERIALIZATION_JOB} (stopped {stopped} wedged execution(s)); "
        "the next run should find the store current",
        title="Materialisation",
    )
    return True


def verify_offline_freshness(city_name, observed_at, feature_group=None):
    """Raise unless the offline store both reaches `observed_at` and covers the
    hours before it.

    Checking only the newest timestamp is not enough, and this is the second
    version of this function for exactly that reason. After the materialisation
    stall of 2026-08-15 was cleared, the newest offline row was an hour old and
    this check passed -- while eleven days were missing behind it. Serving needs
    72 contiguous hours to compute its rolling features, so it silently fell back
    to the last complete row, from before the gap, and the dashboard went on
    showing an eleven-day-old reading through a green pipeline.

    A read that cannot complete is a warning, not a failure: Arrow Flight has its
    own outages, and losing an hour of collection to a monitoring call would be a
    worse trade than an unverified insert.
    """
    try:
        stamps = offline_timestamps(city_name)
    except Exception as error:
        announce(
            f"UNVERIFIED: could not read the offline store ({type(error).__name__}: {error}). "
            "Row inserted; freshness not checked.",
            level="warning",
        )
        note(
            "Offline freshness unverified",
            city=city_name,
            error=f"{type(error).__name__}: {error}",
        )
        return None

    if stamps.empty:
        announce(
            f"STALE: offline store has no rows in the last {OFFLINE_LOOKBACK_HOURS}h",
            level="error",
        )
        raise RuntimeError(_remedy(f"has no rows in the last {OFFLINE_LOOKBACK_HOURS}h", observed_at))

    newest = stamps.max()
    lag_hours = (observed_at - newest).total_seconds() / 3600
    coverage = len(stamps) / OFFLINE_LOOKBACK_HOURS

    if lag_hours < 0:
        announce(
            f"FUTURE: offline store holds rows up to {newest}, "
            f"{abs(lag_hours):.1f}h ahead of the observation just inserted",
            level="error",
        )
        raise RuntimeError(
            f"The offline store contains rows dated {abs(lag_hours):.1f}h in the future "
            f"(newest {newest}). Both source APIs return the rest of the calendar day as "
            "forecast when a range ends today, so a backfill can write hours that have "
            "not happened. Those rows are not observations and must not reach training."
        )

    if coverage > MAX_WINDOW_COVERAGE:
        announce(
            f"DUPLICATED: {len(stamps)} rows in the last {OFFLINE_LOOKBACK_HOURS} hours "
            f"({coverage:.0%} of one row per hour)",
            level="error",
        )
        raise RuntimeError(
            f"The offline store holds {len(stamps)} rows for the last "
            f"{OFFLINE_LOOKBACK_HOURS} hours. More rows than hours means duplicate "
            "primary keys or future-dated rows."
        )

    if lag_hours > STALE_AFTER_HOURS:
        announce(f"STALE: newest offline row {newest} is {lag_hours:.1f}h behind", level="error")
        if feature_group is not None:
            revive_materialization(feature_group)
        raise RuntimeError(_remedy(f"is {lag_hours:.1f}h behind", observed_at))

    if coverage < MIN_WINDOW_COVERAGE:
        announce(
            f"GAP: newest offline row {newest} is current, but only {len(stamps)} of the "
            f"last {OFFLINE_LOOKBACK_HOURS} hours are present ({coverage:.0%})",
            level="error",
        )
        raise RuntimeError(
            f"The offline store holds only {len(stamps)} of the last "
            f"{OFFLINE_LOOKBACK_HOURS} hours. Recent rows exist but the history "
            "behind them does not, so serving cannot compute its 72h rolling "
            "features and will fall back to the newest complete row -- which is on "
            "the far side of the gap. Backfill the missing range."
        )

    announce(
        f"CURRENT: newest offline row {newest}, {lag_hours:.1f}h behind the insert, "
        f"{len(stamps)}/{OFFLINE_LOOKBACK_HOURS} hours present"
    )
    return newest


def _remedy(problem, observed_at):
    """Name both causes rather than guessing between them.

    An earlier version asserted a stalled materialisation job, which sent a person
    to the Hopsworks Jobs UI to find every execution green: the store was behind
    because GitHub had fired the "hourly" workflow three times in seventeen hours.
    Two causes produce the same symptom, and the run summary should say so instead
    of picking the one that happened last time.
    """
    return (
        f"Offline store {problem} while this insert succeeded. Two things cause this: "
        f"the materialisation job {MATERIALIZATION_JOB} is wedged (check the Hopsworks "
        "Jobs UI for a non-terminal execution), or the workflow has not been firing "
        "often enough to keep the store fed (check the run history for gaps). The "
        f"Materialisation annotation on this run says which. The row for {observed_at} "
        "was inserted and is not lost."
    )


def run():
    api_key = os.environ["OPENWEATHER_API_KEY"]
    lat = os.environ["CITY_LAT"]
    lon = os.environ["CITY_LON"]
    city_name = os.environ["CITY_NAME"]

    weather = get_weather(lat, lon, api_key)
    pollution = get_pollution(lat, lon, api_key)

    # Change rate is defined against the immediately preceding hour, matching the
    # backfill; 0.0 if that hour is unavailable, rather than a delta spanning an
    # arbitrary gap. The previous hour is recomputed from the pollution archive
    # rather than read back from Hopsworks: the hourly job should not need the
    # feature store to be queryable in order to write to it, and AQI is a
    # deterministic function of the concentrations.
    observed_at = datetime.fromtimestamp(weather["dt"], tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    previous_aqi = get_previous_hour_aqi(lat, lon, api_key, observed_at)
    row = build_feature_row(city_name, weather, pollution, previous_aqi)
    df = pd.DataFrame([row])

    fg = get_feature_group()
    for attempt in range(1, MAX_INSERT_ATTEMPTS + 1):
        try:
            fg.insert(df)
            break
        except RequestsConnectionError:
            if attempt == MAX_INSERT_ATTEMPTS:
                raise
            wait_seconds = attempt * 2
            print(
                f"Hopsworks insert connection failed (attempt {attempt}/{MAX_INSERT_ATTEMPTS}). "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
    print(f"Inserted feature row for {city_name} at {row['timestamp']} (AQI={row['aqi']})")
    verify_offline_freshness(city_name, row["timestamp"], feature_group=fg)


@monitored_job("aqi-feature-pipeline", crontab="0 * * * *")
def _run_checked_in():
    """The pipeline, wrapped so Sentry hears about every run.

    Separate from run() so the check-in covers exactly the work, and so run()
    stays callable from a test or a shell without a monitor."""
    run()


def main():
    """Run the pipeline, and make any failure legible from the run summary.

    A red run whose cause is only in the log tells you that something broke, not
    what. Annotating the exception means the reason is on the run page beside the
    freshness verdict, in the same place and the same format.
    """
    start_monitoring("feature-pipeline")
    try:
        _run_checked_in()
    except Exception as error:
        announce(f"{type(error).__name__}: {error}", level="error", title="Feature pipeline")
        raise


if __name__ == "__main__":
    main()
