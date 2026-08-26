"""Fetch weather + pollution data, compute AQI, and write one feature row to Hopsworks."""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError
from dotenv import load_dotenv

from feature_pipeline.aqi import compute_aqi
from feature_pipeline.fetch import get_pollution, get_pollution_at, get_weather
from feature_pipeline.hopsworks_store import get_feature_group, offline_max_timestamp

load_dotenv()

MAX_INSERT_ATTEMPTS = 3
STALE_AFTER_HOURS = 6
MATERIALIZATION_JOB = "aqi_features_2_offline_fg_materialization"


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


def announce(message, level="notice"):
    """Report to the log, and to the run summary when running in Actions.

    A number buried in a collapsed log step is not monitoring. As an annotation it
    appears on the run page and in the checks API, so freshness can be read without
    opening anything.
    """
    print(message, flush=True)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title=Offline freshness::{message}", flush=True)


def verify_offline_freshness(city_name, observed_at):
    """Raise if the offline store has not kept up with the inserts.

    A clean return from `fg.insert` means the row was accepted, not that anything
    can read it. Between 2026-08-15 and 2026-08-26 the materialisation job sat in
    Initializing, 264 hourly runs each reported success, and the dashboard served
    an 11-day-old reading throughout. Green has to mean the offline table is
    current, so this runs after the insert and fails the job when it is not --
    the row is already written either way.

    A read that cannot complete is a warning, not a failure: Arrow Flight has its
    own outages, and losing an hour of collection to a monitoring call would be a
    worse trade than an unverified insert.
    """
    try:
        newest = offline_max_timestamp(city_name)
    except Exception as error:
        announce(
            f"UNVERIFIED: could not read the offline store ({type(error).__name__}: {error}). "
            "Row inserted; freshness not checked.",
            level="warning",
        )
        return None

    lag_hours = None if newest is None else (observed_at - newest).total_seconds() / 3600
    if newest is not None and lag_hours <= STALE_AFTER_HOURS:
        announce(f"CURRENT: newest offline row {newest}, {lag_hours:.1f}h behind the insert")
        return newest

    behind = "has no rows in the lookback window" if newest is None else f"is {lag_hours:.1f}h behind"
    announce(f"STALE: offline store {behind}; newest row {newest}", level="error")
    raise RuntimeError(
        f"Offline store {behind} while inserts are succeeding. The materialisation "
        f"job {MATERIALIZATION_JOB} has almost certainly stalled -- stop its current "
        "execution in the Hopsworks Jobs UI and run it again. The row for "
        f"{observed_at} was inserted and is not lost."
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
    verify_offline_freshness(city_name, row["timestamp"])


if __name__ == "__main__":
    run()
